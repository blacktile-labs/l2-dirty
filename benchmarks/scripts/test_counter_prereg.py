"""Unit tests for counter_prereg (CPU only; SYNTHETIC fixtures, no measured data)."""

import tempfile
import unittest
from pathlib import Path

import cache_sim
import cache_sim_b5
import counter_prereg as cp

L2 = 64 * 1024  # small cache keeps the tests fast; ratios scale with l2x


class TestAgreesWithLockedSimulators(unittest.TestCase):
    """For LRU, replaying the whole launch sequence (flush included) must give the same window
    totals as the locked per-window simulators, which assume an empty cache at window start."""

    def check(self, bench, ref_fn):
        for l2x in (0.25, 0.5625, 0.75, 1.0, 2.0):
            n = cp.n_for(bench, l2x, L2)
            got = cp.window_sums(cp.simulate_launches(bench, n, L2, "lru"))
            ref = ref_fn(n, L2, "lru")
            for key, b in ref.items():
                with self.subTest(l2x=l2x, key=key):
                    self.assertEqual(got[key], b)

    def test_fusion(self):
        self.check("fusion", cache_sim.simulate_fusion)

    def test_stencil(self):
        self.check("stencil", cache_sim_b5.simulate_stencil)


class TestMechanisms(unittest.TestCase):
    def setUp(self):
        n = cp.n_for("fusion", 0.25, L2)
        self.B = 4 * n
        self.L = {d["label"]: d for d in cp.simulate_launches("fusion", n, L2, "lru")}

    def test_small_window_defers_writes(self):
        # y fits in L2: no DRAM write inside the fused window ...
        self.assertEqual(self.L["fused|isolated|k1"]["write"], 0)
        # ... and the flush that follows stage2_cold pays for all of y
        self.assertEqual(self.L["stage2_warm|isolated|flush"]["write"], self.B)

    def test_flush_reads_whole_buffer(self):
        self.assertEqual(self.L["fused|isolated|flush"]["read"], cp.flush_bytes(L2))
        self.assertFalse(self.L["fused|isolated|flush"]["scored_write"])
        self.assertTrue(self.L["materialized|isolated|flush"]["scored_write"])

    def test_flush_bytes_matches_timer(self):
        self.assertEqual(cp.flush_bytes(3 << 20), 16 << 20)    # 4 x 3 MiB < 16 MiB floor
        self.assertEqual(cp.flush_bytes(24 << 20), 96 << 20)


def fake_measured(bench, n_sizes, noise_kernels=("fill_u32_kernel", "fused_kernel")):
    """Correctness-phase launches (never a flush) before every size, then the timed plan."""
    rows, i = [], 1
    for _ in range(n_sizes):
        for k in noise_kernels:
            rows.append({"id": i, "kernel": k, "read": 0.0, "write": 0.0})
            i += 1
        for _, k, _, _ in cp.launch_plan(bench):
            rows.append({"id": i, "kernel": k, "read": 1.0, "write": 2.0})
            i += 1
    return rows


class TestAlignAndLoad(unittest.TestCase):
    def test_align_ok(self):
        segs = cp.align(fake_measured("stencil", 3, ("fill_u32_kernel", "sweep13_kernel")), "stencil", 3)
        self.assertEqual(len(segs), 3)
        self.assertTrue(all(s[0]["kernel"] == cp.FLUSH for s in segs))

    def test_align_rejects_mismatch(self):
        rows = fake_measured("fusion", 2)
        rows[-1]["kernel"] = "fused_kernel"  # last launch should be stage2_kernel
        with self.assertRaises(ValueError):
            cp.align(rows, "fusion", 2)

    def test_load_units_and_commas(self):
        txt = ('"ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum"\n'
               '"","","Kbyte","byte"\n'
               '"1","stage1_kernel(const float *, float *, unsigned long, int)","1,024","2,048"\n'
               '"0","fill_u32_kernel","0","0"\n')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.csv"
            p.write_text("==PROF== Disconnected\n" + txt)
            rows = cp.load_ncu_raw(p)
        self.assertEqual([r["id"] for r in rows], [0, 1])
        self.assertEqual(rows[1]["kernel"], "stage1_kernel")
        self.assertEqual(rows[1]["read"], 1024e3)
        self.assertEqual(rows[1]["write"], 2048.0)


if __name__ == "__main__":
    unittest.main()
