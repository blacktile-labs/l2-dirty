import unittest

import absorb_score as s

L2 = 64 << 20


def build(cfgs, n_cand=None, timing_runs=3):
    """cfgs: {name: (R_bytes, t_read_ms, E_bytes, dt_ms)} for one shape."""
    shape = [4096, 14336]
    rows, rrows, trows = [], [], []
    for c, (R, t, E, dt) in cfgs.items():
        for rep in range(3):
            for mode in ("zero", "read"):
                rows.append({"kind": "range", "shape": shape, "cand": c, "mode": mode, "rep": rep})
                rrows.append({"read": R, "write": E if mode == "zero" else 0.0})
        trows.append({"kind": "timing", "shape": shape, "cand": c, "zero_ms": t + dt, "read_ms": t})
    rmeta = {"device": "X", "l2_bytes": L2, "n_candidates": n_cand or len(cfgs), "rows": rows}
    return rmeta, rrows, [{"rows": trows}] * timing_runs


class AbsorbScore(unittest.TestCase):
    def sc(self, *a, **k):
        return s.score(*build(*a, **k), log=lambda *_: None)

    def test_absorption_supported(self):
        # same residue E for all; faster (higher-BW) configs pay more
        E = 16e6
        cfgs = {"a": (28e6, 0.054, E, 0.0215), "b": (28e6, 0.060, E, 0.016),
                "c": (28e6, 0.067, E, 0.007), "d": (28e6, 0.095, E, 0.002)}
        v = self.sc(cfgs)
        self.assertTrue(v["K-0"])
        self.assertEqual(v["K-1"], "penalty grows with bandwidth use")
        self.assertEqual(v["K-2"], "same write-back bytes")

    def test_no_relation_and_bytes_differ(self):
        cfgs = {"a": (28e6, 0.054, 4e6, 0.001), "b": (28e6, 0.060, 16e6, 0.010),
                "c": (28e6, 0.067, 8e6, 0.020), "d": (28e6, 0.095, 20e6, 0.030)}
        v = self.sc(cfgs)
        self.assertEqual(v["K-1"], "no")
        self.assertEqual(v["K-2"], "bytes differ")

    def test_sanity(self):
        cfgs = {"a": (28e6, 0.05, 1e6, 0.01), "b": (28e6, 0.06, 1e6, 0.01)}
        self.assertFalse(self.sc(cfgs, timing_runs=2)["K-0"])
        self.assertFalse(self.sc(cfgs, n_cand=36)["K-0"])

    def test_range_count_mismatch(self):
        rmeta, rrows, tm = build({"a": (1, 1, 1, 1)})
        with self.assertRaises(ValueError):
            s.score(rmeta, rrows[:-1], tm, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
