"""Unit tests for queue_drain_score.py (B18b) on synthetic data."""

import unittest

import queue_drain_score as S

KERNELS = ("gemv4", "e_large", "t_small")
MODES = ("pr256", "prL2", "pr2L2", "zero256", "zeroL2", "zero256_wait")
BASE = {"gemv4": 57.0, "e_large": 46.0, "t_small": 10.0}


def build(fn, runs=3, fails=0, pad_us=300.0):
    metas = []
    for _ in range(runs):
        ev = []
        for k in KERNELS:
            for c in ("plain", "pad"):
                for m in MODES:
                    t = fn(k, c, m)
                    ev.append({"kind": "timing", "kernel": k, "cond": c, "mode": m, "times_us": [t, t * 1.001, t * 0.999]})
        metas.append({"device": "RTX 3060 Laptop", "l2_bytes": 3 << 20, "events": ev, "correctness_failures": fails,
                      "pad_us": pad_us})
    return metas


def queue_drain_world(k, c, m):
    t = BASE[k] * (1.10 if m.startswith("zero") else 1.0)
    if c == "plain" and m in ("prL2", "pr2L2", "zeroL2", "zero256_wait"):
        t += 30.0                                            # host latency exposed when the queue drains
    if m == "zero256_wait" and c == "pad":
        t *= 2.0                                             # clocks
    return t


def no_drain_world(k, c, m):
    t = BASE[k] * (1.10 if m.startswith("zero") else 1.0)
    if m in ("prL2", "pr2L2", "zeroL2"):
        t *= 1.5                                             # inflation that padding does not remove
    return t


class TestScore(unittest.TestCase):
    def test_queue_drain_world(self):
        v = S.score(build(queue_drain_world), log=lambda *_: None)
        self.assertTrue(v["Q-0"])
        self.assertEqual(v["Q-1"], "queue drain explains the inflation")
        self.assertEqual(v["Q-2"], "a 256 MiB flush already keeps the queue full")
        self.assertEqual(v["Q-3"], "the anomaly is queue drain")
        self.assertEqual(v["Q-4"], "the wait slowdown is something else (e.g. clocks)")

    def test_no_drain_world(self):
        v = S.score(build(no_drain_world), log=lambda *_: None)
        self.assertTrue(v["Q-0"])
        self.assertEqual(v["Q-1"], "not queue drain")
        self.assertEqual(v["Q-3"], "not queue drain")

    def test_gate(self):
        self.assertFalse(S.score(build(queue_drain_world, fails=1), log=lambda *_: None)["Q-0"])
        self.assertFalse(S.score(build(queue_drain_world, pad_us=50.0), log=lambda *_: None)["Q-0"])
        flat = S.score(build(lambda k, c, m: BASE[k]), log=lambda *_: None)
        self.assertFalse(flat["Q-0"])
        self.assertTrue(flat["Q-1"].startswith("not scored"))

    def test_no_anomaly_branch(self):
        def world(k, c, m):
            t = queue_drain_world(k, c, m)
            return t - 30.0 if (k == "t_small" and c == "plain" and m == "zeroL2") else t
        v = S.score(build(world), log=lambda *_: None)
        self.assertEqual(v["Q-3"], "no L2-sized-write anomaly on this GPU")


if __name__ == "__main__":
    unittest.main()
