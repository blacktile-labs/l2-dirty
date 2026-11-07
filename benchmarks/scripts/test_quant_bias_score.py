import unittest

import quant_bias_score as s


def meta(times: dict, n_cand=None):
    """times: {cand: (format, zero_ms, read_ms)}"""
    rows = [{"shape": [4096, 4096], "cand": c, "format": f, "zero_ms": z, "read_ms": r}
            for c, (f, z, r) in times.items()]
    return {"device": "X", "n_candidates": n_cand or len(times), "rows": rows}


class QuantBiasScore(unittest.TestCase):
    def sc(self, times, runs=3, **k):
        return s.score([meta(times, **k)] * runs, log=lambda *_: None)

    def test_additive_bias_underestimates_speedup(self):
        # true: fp16 40, w8 20, w4 10 us; the write flush adds 10 us to everyone
        times = {"cublas": ("fp16", 0.050, 0.040), "a": ("w8", 0.030, 0.020), "b": ("w4", 0.020, 0.010)}
        v = self.sc(times)
        self.assertTrue(v["U-0"])
        self.assertEqual(v["U-1"], "distorted")
        self.assertEqual(v["U-1 sign"], "speedup under-estimated")
        self.assertAlmostEqual(v["U-1 worst D"], (50 / 20) / (40 / 10) - 1, places=4)
        self.assertEqual(v["U-2"], "no flip")

    def test_proportional_bias_is_faithful(self):
        times = {"cublas": ("fp16", 0.044, 0.040), "a": ("w8", 0.022, 0.020), "b": ("w4", 0.011, 0.010)}
        self.assertEqual(self.sc(times)["U-1"], "faithful")

    def test_flip(self):
        # truly w8 fastest, but under the write flush fp16 looks fastest
        times = {"cublas": ("fp16", 0.030, 0.029), "a": ("w8", 0.032, 0.025), "b": ("w4", 0.040, 0.030)}
        self.assertEqual(self.sc(times)["U-2"], "flips")

    def test_sanity(self):
        times = {"cublas": ("fp16", 0.05, 0.04), "a": ("w8", 0.03, 0.02), "b": ("w4", 0.02, 0.01)}
        self.assertFalse(self.sc(times, runs=2)["U-0"])
        self.assertFalse(self.sc(times, n_cand=37)["U-0"])


if __name__ == "__main__":
    unittest.main()
