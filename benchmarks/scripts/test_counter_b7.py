"""Unit tests for counter_b7 (CPU only; SYNTHETIC plan and 'measurements')."""

import unittest

import counter_b7 as cb

L2 = 64 * 1024
H = 4096


def synthetic_plan(sizes=(0.25, 0.5, 2.0)):
    """Mimics torch_opchain's allocation pattern: t1's block is reused for y in eager."""
    launches, base = [], 1 << 30
    flush = [base, max(4 * L2, 16 << 20)]
    for f in sizes:
        m = max(1, round(f * L2 / (4 * H)))
        B = m * H * 4
        x, r, b, t1, t2, yc = (base + (k + 1) * (64 << 20) for k in range(6))
        for variant, mode, calls in (("eager", "isolated", 1), ("compiled", "isolated", 1),
                                     ("eager", "steady", 3), ("compiled", "steady", 3)):
            launches.append({"l2x": f, "M": m, "variant": variant, "mode": mode, "kernel": "flush",
                             "reads": [flush], "writes": []})
            for _ in range(calls):
                if variant == "eager":
                    launches += [
                        {"l2x": f, "M": m, "variant": variant, "mode": mode, "kernel": "eager_add1",
                         "reads": [[x, B], [b, H * 4]], "writes": [[t1, B]]},
                        {"l2x": f, "M": m, "variant": variant, "mode": mode, "kernel": "eager_gelu",
                         "reads": [[t1, B]], "writes": [[t2, B]]},
                        {"l2x": f, "M": m, "variant": variant, "mode": mode, "kernel": "eager_add2",
                         "reads": [[t2, B], [r, B]], "writes": [[t1, B]]}]
                else:
                    launches.append({"l2x": f, "M": m, "variant": variant, "mode": mode, "kernel": "inductor_fused",
                                     "reads": [[x, B], [r, B], [b, H * 4]], "writes": [[yc, B]]})
    return {"meta": {"l2_bytes": L2, "H": H, "flush_bytes": flush[1]}, "launches": launches}


NAMES = {"flush": "flush_l2_read", "eager_add1": "void at::elementwise_kernel<128, 2>",
         "eager_gelu": "void at::vectorized_elementwise_kernel<4, at::GeluCUDAKernelImpl>",
         "eager_add2": "void at::vectorized_elementwise_kernel<4, at::CUDAFunctor_add<float>>",
         "inductor_fused": "triton_poi_fused_add_gelu_0"}


def fake_rows(plan, model, prefix=3):
    sim = cb.simulate(plan["launches"], L2, model)
    rows = [{"id": k, "kernel": "void at::vectorized_elementwise_kernel<4, at::FillFunctor<float>>", "read": 0, "write": 0}
            for k in range(prefix)]  # correctness-phase launches before the first flush are skipped
    for k, (L, s) in enumerate(zip(plan["launches"], sim)):
        rows.append({"id": prefix + k, "kernel": NAMES[L["kernel"]], **s})
    return rows


class TestB7(unittest.TestCase):
    def setUp(self):
        self.plan = synthetic_plan()
        self.pred = cb.predictions(self.plan)

    def run_world(self, model):
        runs = [cb.align(fake_rows(self.plan, model), self.plan["launches"])] * 3
        return cb.score(self.pred, runs, self.plan["meta"]["flush_bytes"], log=lambda *_: None)

    def test_v4_world(self):
        v = self.run_world("v4")
        self.assertTrue(v["R-0"])
        self.assertTrue(v["R-1"])
        self.assertIn(v["R-2"], ("v4 better", "n/a"))
        self.assertTrue(v["R-3"])

    def test_v3_world_loses_r2_when_models_differ(self):
        v = self.run_world("v3")
        if v["R-2"] != "n/a":
            self.assertEqual(v["R-2"], "v3 better")

    def test_misaligned_raises(self):
        rows = fake_rows(self.plan, "v4")
        rows[5]["kernel"] = "triton_poi_other"
        with self.assertRaises(ValueError):
            cb.align(rows, self.plan["launches"])

    def test_far_sizes_stream_from_dram(self):
        fused_steady = [p for p in self.pred
                        if p["l2x"] == 2.0 and p["kernel"] == "inductor_fused" and p["mode"] == "steady"]
        self.assertEqual(len(fused_steady), 3)
        for p in fused_steady[1:]:  # the 1st call after the flush leaves its dirty tail for the next one
            if True:
                bias = H * 4 / p["B"]  # the bias is evicted by the streams and re-read every launch
                self.assertAlmostEqual(p["v4_read"] / p["B"], 2.0 + bias, delta=0.05)  # x, r, b
                self.assertAlmostEqual(p["v4_write"] / p["B"], 1.0, delta=0.05)  # y


if __name__ == "__main__":
    unittest.main()
