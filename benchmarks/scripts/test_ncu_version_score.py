"""Unit tests for ncu_version_score.py (B20)."""

import unittest

import ncu_version_score as S


def sc(q0, l75, l1, l15=0.0):
    return {"Q-0": q0, "Q-1 loss": {"0.75": l75, "1": l1, "1.5": l15}}


class T(unittest.TestCase):
    def test_persists_unchanged(self):
        v = S.score(sc(True, 0.089, 0.067), sc(True, 0.085, 0.07), log=lambda *_: None)
        self.assertEqual(v["N-1"], "persists in the new version")
        self.assertEqual(v["N-2"], "unchanged between versions")

    def test_fixed_changed(self):
        v = S.score(sc(True, 0.089, 0.067), sc(True, 0.005, 0.01), log=lambda *_: None)
        self.assertEqual(v["N-1"], "fixed in the new version")
        self.assertEqual(v["N-2"], "changed")

    def test_wrapped_input(self):
        v = S.score({"verdict": sc(True, 0.089, 0.067)}, {"verdict": sc(True, 0.085, 0.07)}, log=lambda *_: None)
        self.assertEqual(v["N-1"], "persists in the new version")

    def test_partial_and_gate(self):
        self.assertEqual(S.score(sc(True, 0.09, 0.07), sc(True, 0.06, 0.03), log=lambda *_: None)["N-1"], "partial")
        self.assertTrue(S.score(sc(True, 0.09, 0.07), sc(False, 0.06, 0.03), log=lambda *_: None)["N-1"].startswith("not scored"))


if __name__ == "__main__":
    unittest.main()
