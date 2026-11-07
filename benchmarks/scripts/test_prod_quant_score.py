import unittest

import prod_quant_score as s


def meta(times, shape=(4096, 4096)):
    rows = [{"shape": list(shape), "cand": c, "zero_ms": z, "read_ms": r} for c, (z, r) in times.items()]
    return {"device": "X", "l2_bytes": 64 << 20, "rows": rows}


BASE = {"cublas_fp16": (0.070, 0.050), "tinygemm_k2": (0.040, 0.020), "tinygemm_k8": (0.038, 0.018),
        "bnb_nf4": (0.045, 0.025)}


class ProdQuantScore(unittest.TestCase):
    def sc(self, times, runs=3):
        return s.score([meta(times)] * runs, log=lambda *_: None)

    def test_additive_bias_distorts_and_underestimates(self):
        v = self.sc(BASE)
        self.assertTrue(v["P-0"])
        self.assertEqual(v["V-1"], "distorted")
        self.assertEqual(v["V-1 sign"], "speedup under-estimated")
        self.assertEqual(v["V-2"], "no flip")
        self.assertEqual(v["V-3 max regret"], 0.0)

    def test_faithful_proportional(self):
        t = {"cublas_fp16": (0.055, 0.050), "tinygemm_k8": (0.0198, 0.018), "bnb_nf4": (0.0275, 0.025)}
        self.assertEqual(self.sc(t)["V-1"], "faithful")

    def test_flip_and_pick_regret(self):
        t = dict(BASE)
        t["bnb_nf4"] = (0.036, 0.025)          # looks fastest under zero, truly slower than tinygemm
        t["tinygemm_k2"] = (0.037, 0.021)      # tinygemm pick under zero is k2, truly k8 is best
        v = self.sc(t)
        self.assertEqual(v["V-2"], "flips")
        self.assertGreater(v["V-3 max regret"], 0.05)

    def test_missing_candidate_fails_p0(self):
        t = {k: v for k, v in BASE.items() if k != "bnb_nf4"}
        self.assertFalse(self.sc(t)["P-0"])
        self.assertFalse(self.sc(BASE, runs=2)["P-0"])


if __name__ == "__main__":
    unittest.main()
