"""Unit tests for e2e_score.py (B21) on synthetic data."""

import unittest

import e2e_score as S

MIB = 1 << 20
SHAPES = ("qkv", "o", "gate_up", "down")
KT = {"qkv": 60.0, "o": 20.0, "gate_up": 110.0, "down": 55.0}       # cold per-kernel times, fp16


def build(l2=96 * MIB, step_scale=1.0, write_bias=0.25, int4_speed=3.0, write_int4_bias=0.6, fails=0, steps=("fp16", "int4")):
    metas = []
    for _ in range(3):
        ev = []
        for v in ("fp16", "int4"):
            sp = 1.0 if v == "fp16" else int4_speed
            if v in steps:
                t = 4 * sum(KT.values()) / sp * step_scale
                ev.append({"kind": "step", "variant": v, "times_us": [t * 0.99, t, t * 1.01]})
            for s in SHAPES:
                c = KT[s] / sp
                wb = write_bias if v == "fp16" else write_int4_bias
                ev.append({"kind": "kernel", "variant": v, "shape": s, "write_us": c * (1 + wb), "cold_us": c,
                           "steady_us": c, "l2w_us": c * (1 + wb)})
        metas.append({"device": "NVIDIA L40S", "l2_bytes": l2, "layers": 4, "events": ev, "correctness_failures": fails,
                      "notes": {}})
    return metas


class T(unittest.TestCase):
    def q(self, **kw):
        return S.score(build(**kw), log=lambda *_: None)

    def test_predicted(self):
        v = self.q()
        self.assertTrue(v["U-0"])
        self.assertEqual(v["U-1"], "cold predicts the in-model step")
        self.assertEqual(v["U-2"], "the shipped write flush over-predicts the step")
        self.assertEqual(v["U-3"], "cold predicts the int4 speedup better than the write flush")

    def test_small_l2_and_misses(self):
        self.assertEqual(self.q(l2=3 * MIB)["U-2"], "n/a (small L2)")
        self.assertEqual(self.q(step_scale=1.3)["U-1"], "not")
        self.assertEqual(self.q(write_bias=0.02)["U-2"], "not")
        self.assertEqual(self.q(write_int4_bias=0.25)["U-3"], "not")      # write bias identical for both: D_write = 0

    def test_gate(self):
        self.assertFalse(self.q(fails=1)["U-0"])
        self.assertFalse(self.q(steps=("fp16",))["U-0"])

    def test_all(self):
        vs = [self.q(), self.q(l2=3 * MIB)]
        out = S.score_all(vs, log=lambda *_: None)
        self.assertEqual(out["U-1"], "2/2")
        self.assertEqual(out["U-2"], "1/1")


if __name__ == "__main__":
    unittest.main()
