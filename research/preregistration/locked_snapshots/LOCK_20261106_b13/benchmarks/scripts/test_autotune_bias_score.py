import unittest

import autotune_bias_score as s


def meta(times: dict, n_cand=None, shape=(4096, 4096)):
    """times: {cand: (zero_ms, read_ms)}"""
    rows = [{"shape": list(shape), "cand": c, "zero_ms": z, "read_ms": r} for c, (z, r) in times.items()]
    return {"device": "X", "n_candidates": n_cand or len(times), "rows": rows}


class AutotuneBiasScore(unittest.TestCase):
    def sc(self, times, runs=3, **k):
        return s.score([meta(times, **k)] * runs, log=lambda *_: None)

    def test_additive_bias_is_harmless(self):
        times = {"cublas": (1.30, 1.00), "a": (1.40, 1.10), "b": (1.50, 1.20)}
        v = self.sc(times)
        self.assertTrue(v["T-0"])
        self.assertEqual(v["T-1"], "harmless")
        self.assertEqual(v["T-2"], "no flip")

    def test_bias_that_reorders_is_harmful_and_flips(self):
        # cuBLAS truly fastest (read), but the write flush penalizes it most
        times = {"cublas": (1.60, 1.00), "a": (1.20, 1.10), "b": (1.50, 1.20)}
        v = self.sc(times)
        self.assertEqual(v["T-1"], "harmful")
        self.assertAlmostEqual(v["T-1 max regret"], 0.10, places=3)
        self.assertEqual(v["T-2"], "flips")

    def test_minor(self):
        times = {"cublas": (1.60, 1.00), "a": (1.20, 1.03), "b": (1.50, 1.20)}
        self.assertEqual(self.sc(times)["T-1"], "minor")

    def test_sanity_needs_three_runs_and_valid_candidates(self):
        times = {"cublas": (1.3, 1.0), "a": (1.4, 1.1)}
        self.assertFalse(self.sc(times, runs=2)["T-0"])
        self.assertFalse(self.sc(times, n_cand=37)["T-0"])

    def test_spearman(self):
        self.assertAlmostEqual(s.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(s.spearman([1, 2, 3], [30, 20, 10]), -1.0)


if __name__ == "__main__":
    unittest.main()
