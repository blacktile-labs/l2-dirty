"""Unit tests for counter_b7r (CPU only; SYNTHETIC data)."""

import unittest

import counter_b7r as r7

SIZES = (0.125, 0.25, 0.3125, 0.375, 0.5, 1.0, 2.0, 3.0)
SEQ = (("eager", "isolated", 3), ("compiled", "isolated", 1), ("eager", "steady", 9), ("compiled", "steady", 3))
B = 1000.0


def make_pred():
    pred = []
    for f in SIZES:
        for var, mode, n in SEQ:
            pred.append({"l2x": f, "variant": var, "mode": mode, "kernel": "flush", "B": B,
                         "v4_read": 16 * B, "v4_write": 0.0})
            for _ in range(n):
                pred.append({"l2x": f, "variant": var, "mode": mode, "kernel": "k", "B": B,
                             "v4_read": 1.0 * B, "v4_write": 1.0 * B})
    return pred


def world(k_err, g_err, bg=1000.0, bg_w=0.0, enq=500.0, gate_iters=None, gate_max=None):
    """k_err / g_err: per-launch error (B) of kernel / range mode at edge sizes, 0 elsewhere."""
    pred = make_pred()
    segs = r7.segments(pred)
    kruns = [[{"read": p["v4_read"] + (k_err * B if p["kernel"] != "flush" and p["l2x"] in r7.EDGE else 0),
               "write": p["v4_write"] + (k_err * B if p["kernel"] != "flush" and p["l2x"] in r7.EDGE else 0)}
              for p in pred] for _ in range(3)]
    ranges, rows = [], []
    for s in segs:
        if s["variant"] == "eager" and s["mode"] == "isolated":
            ranges.append({"kind": "control", "l2x": s["l2x"], **({"gate_iters": 100} if gate_max else {})})
            rows.append({"read": bg, "write": bg_w})
        ranges.append({"kind": "segment", "l2x": s["l2x"], "variant": s["variant"], "mode": s["mode"],
                       "first_launch": s["first"], "n_launches": s["n"], "enqueue_us": enq,
                       **({"gate_iters": gate_iters} if gate_max else {})})
        e = g_err * B * s["n"] if s["l2x"] in r7.EDGE else 0.0
        rows.append({"read": s["P"]["read"] + e + bg, "write": s["P"]["write"] + e + bg_w})
    r7.check_range_meta(segs, ranges)
    return r7.score(segs, kruns, [rows] * 3, ranges, [1.9e6] * 8, {"plans": True, "correctness": True},
                    log=lambda *_: None, gate_max=gate_max)


class TestB7r(unittest.TestCase):
    def test_segments(self):
        segs = r7.segments(make_pred())
        self.assertEqual(len(segs), 4 * len(SIZES))
        self.assertEqual([s["n"] for s in segs[:4]], [3, 1, 9, 3])
        self.assertEqual(segs[0]["P"]["read"], 3 * B)

    def test_measurement_artifact(self):
        v = world(k_err=-0.2, g_err=0.0)
        self.assertTrue(v["Y-0"])
        self.assertEqual(v["Y-1"], "reads change with replay mode")
        self.assertEqual(v["Y-2"], "write error mostly a measurement artifact")
        self.assertEqual(v["Y-3"], "MEASUREMENT")

    def test_model_error(self):
        v = world(k_err=-0.2, g_err=-0.2)
        self.assertTrue(v["Y-0"])
        self.assertEqual(v["Y-1"], "reads unchanged by replay mode")
        self.assertEqual(v["Y-2"], "write error not a measurement artifact")
        self.assertEqual(v["Y-3"], "MODEL")

    def test_partial(self):
        v = world(k_err=-0.2, g_err=-0.13)
        self.assertEqual(v["Y-3"], "PARTIAL")

    def test_background_subtracted(self):
        v = world(k_err=-0.2, g_err=-0.2, bg=9000.0)
        self.assertTrue(v["Y-0"])
        self.assertEqual(v["Y-3"], "MODEL")

    def test_validity_failures(self):
        self.assertFalse(world(-0.2, -0.2, bg=20000.0)["Y-0 parts"]["b background"])
        self.assertFalse(world(-0.2, -0.2, bg_w=5000.0)["Y-0 parts"]["b background"])
        self.assertFalse(world(-0.2, -0.2, enq=1700.0)["Y-0 parts"]["c back-to-back"])

    def test_flag_gate(self):
        # slow enqueue is fine with the flag gate; only a gate timeout fails Y-0c
        self.assertTrue(world(-0.2, -0.2, enq=5000.0, gate_iters=20000, gate_max=2_000_000)["Y-0 parts"]["c back-to-back"])
        self.assertFalse(world(-0.2, -0.2, gate_iters=2_000_000, gate_max=2_000_000)["Y-0 parts"]["c back-to-back"])


if __name__ == "__main__":
    unittest.main()
