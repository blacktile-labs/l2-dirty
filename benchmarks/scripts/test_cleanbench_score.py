"""Unit tests for cleanbench_score.py (B18c) and cleanbench's statistics helpers (no GPU needed)."""

import unittest

import cleanbench as cb
import cleanbench_score as S

BASE = {"gemv4": 58.0, "e_large": 46.0, "e_mid": 7.0, "t_small": 12.0}


def build(fn, runs=3, calls=5, fails=0, drop=None):
    metas = []
    for _ in range(runs):
        ev = []
        for k in BASE:
            for c in ("cold_2L2", "cold_256", "steady"):
                if drop == (k, c):
                    continue
                for _ in range(calls):
                    t = fn(k, c)
                    ev.append({"kind": "bench", "kernel": k, "config": c, "mean_us": t, "median_us": t, "p25_us": t * 0.99,
                               "p75_us": t * 1.01, "pad_us": 300.0, "host_dispatch_us": 80.0})
            ev.append({"kind": "do_bench", "kernel": k, "median_us": BASE[k] * 1.1})
        metas.append({"device": "RTX 3060 Laptop", "l2_bytes": 3 << 20, "events": ev, "correctness_failures": fails})
    return metas


def predicted(k, c):
    if c != "steady":
        return BASE[k] * (1.01 if c == "cold_2L2" else 1.0)
    return BASE[k] * {"gemv4": 1.0, "e_large": 1.13, "e_mid": 1.6, "t_small": 1.3}[k]


class TestScore(unittest.TestCase):
    def test_predicted(self):
        v = S.score(build(predicted), log=lambda *_: None)
        self.assertTrue(v["T-0"])
        self.assertEqual(v["T-1"], "cold timing does not depend on the flush size")
        self.assertEqual(v["T-2"], "steady mode charges the deferred write-back as predicted")

    def test_flush_size_dependence_and_bad_steady(self):
        def world(k, c):
            if c == "cold_2L2":
                return BASE[k] * 1.5
            if c == "steady":
                return BASE[k]
            return BASE[k]
        v = S.score(build(world), log=lambda *_: None)
        self.assertEqual(v["T-1"], "depends")
        self.assertTrue(v["T-2"].startswith("not as predicted"))
        self.assertIn("e_mid", v["T-2"])

    def test_gate(self):
        self.assertFalse(S.score(build(predicted, fails=1), log=lambda *_: None)["T-0"])
        v = S.score(build(predicted, drop=("t_small", "steady")), log=lambda *_: None)
        self.assertFalse(v["T-0"])
        self.assertTrue(v["T-2"].startswith("not scored"))


class TestHelpers(unittest.TestCase):
    def test_tick_fraction(self):
        self.assertEqual(cb.tick_fraction([1.024, 2.048, 3.072]), 1.0)
        self.assertEqual(cb.tick_fraction([1.5, 2.5]), 0.0)
        self.assertEqual(cb.tick_fraction([]), 0.0)

    def test_quantiles(self):
        p25, p50, p75, p95 = cb.quantiles([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual((p25, p50, p75), (2.0, 3.0, 4.0))
        self.assertAlmostEqual(p95, 4.8)

    def test_warnings(self):
        w = cb.warnings_for([10.24] * 20, "cold")
        self.assertTrue(any("timer ticks" in x for x in w))
        self.assertTrue(any("cold mode" in x for x in w))
        self.assertFalse(any("IQR" in x for x in cb.warnings_for([100.0] * 20, "steady")))
        self.assertTrue(any("IQR" in x for x in cb.warnings_for([80.0, 100.0, 120.0, 140.0], "steady")))


if __name__ == "__main__":
    unittest.main()
