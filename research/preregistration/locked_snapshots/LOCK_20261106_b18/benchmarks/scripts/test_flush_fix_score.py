"""Unit tests for flush_fix_score.py (B18) on synthetic data."""

import unittest

import flush_fix_score as S

L2 = 3 << 20
KINFO = {"gemv4": {"in_bytes": 4 * L2, "out_bytes": 3072, "copies": 2},
         "e_tiny": {"in_bytes": L2 // 128, "out_bytes": L2 // 128, "copies": 256},
         "e_small": {"in_bytes": L2 // 8, "out_bytes": L2 // 8, "copies": 16},
         "e_large": {"in_bytes": 2 * L2, "out_bytes": 2 * L2, "copies": 2}}
MODES = ("none", "zero256", "zero256_wait", "zero256_read", "zeroL2", "pr256", "sum256", "prL2", "pr2L2", "rotate")


def predicted_cells():
    """(read, write) bytes per (kernel, mode) following the researcher's predictions."""
    c = {}
    for m in MODES:
        c[("gemv4", m)] = (4 * L2, 0.0)
        c[("e_large", m)] = (2 * L2, 0.8 * 2 * L2)
        c[("e_tiny", m)] = (L2 / 128, 0.0)
        c[("e_small", m)] = (L2 / 8, 0.0)
    for m in ("zero256", "zero256_wait", "zeroL2"):
        c[("gemv4", m)] = (4 * L2, 0.4 * L2)
    c[("gemv4", "zero256_wait")] = (4 * L2, 0.38 * L2)
    c[("e_tiny", "prL2")] = (0.3 * L2 / 128, 0.0)
    c[("e_tiny", "none")] = (0.0, 0.0)
    c[("e_small", "rotate")] = (L2 / 8, 0.95 * L2 / 8)
    return c


def build(cells, runs=3, reps=5, fails=0, ctl_write=0.0):
    metas, rows = [], []
    for _ in range(runs):
        ev, rw = [], []
        for _ in range(2):
            ev.append({"kind": "control"})
            rw.append({"read": 1000.0, "write": ctl_write})
        for (k, m), (rd, wr) in cells.items():
            for rep in range(reps):
                ev.append({"kind": "call", "kernel": k, "mode": m, "rep": rep})
                rw.append({"read": rd * (1 + 0.001 * rep), "write": wr})
        metas.append({"l2_bytes": L2, "device": "NVIDIA GeForce RTX 3060 Laptop GPU", "kernels": KINFO,
                      "events": ev, "correctness_failures": fails,
                      "pr_flush_check": {"buffer_unchanged": True, "st_global": 0}})
        rows.append(rw)
    tmetas = [{"events": [{"kind": "timing", "kernel": "gemv4", "mode": m, "times_us": [40.0, 41.0, 42.0]}
                          for m in MODES], "correctness_failures": 0,
               "pr_flush_check": {"buffer_unchanged": True, "st_global": 0}}]
    return metas, S.range_rows(metas, rows), tmetas


class TestScore(unittest.TestCase):
    def run_score(self, cells, **kw):
        m, rr, t = build(cells, **kw)
        return S.score(m, rr, t, log=lambda *_: None)

    def test_predicted_pattern(self):
        v = self.run_score(predicted_cells())
        self.assertTrue(v["H-0"])
        self.assertTrue(v["H-0e"])
        self.assertEqual(v["H-1"], "waiting does not remove the residue")
        self.assertEqual(v["H-2"], "write-then-read removes the residue")
        self.assertEqual(v["H-3a"], "an L2-sized read flush leaves a tiny kernel warm")
        self.assertEqual(v["H-3b"], "sufficient at footprint 0.25 L2")
        self.assertEqual(v["H-3c"], "a 2 x L2 read flush suffices")
        self.assertEqual(v["H-4a"], "a read flush keeps a small kernel's own write-back out of its window")
        self.assertEqual(v["H-4b"], "rotation charges about one output per call")
        self.assertEqual(v["H-4c"], "a kernel with output > L2/2 pays most of its own write-back")

    def test_opposite_pattern(self):
        c = predicted_cells()
        c[("gemv4", "zero256_wait")] = (4 * L2, 0.01 * L2)
        c[("gemv4", "zero256_read")] = (4 * L2, 0.3 * L2)
        c[("e_tiny", "prL2")] = (0.95 * L2 / 128, 0.0)
        c[("e_small", "pr256")] = (L2 / 8, 0.9 * L2 / 8)
        c[("e_small", "rotate")] = (L2 / 8, 0.1 * L2 / 8)
        c[("e_large", "pr256")] = (2 * L2, 0.2 * 2 * L2)
        v = self.run_score(c)
        self.assertEqual(v["H-1"], "waiting removes the residue")
        self.assertEqual(v["H-2"], "write-then-read does not")
        self.assertEqual(v["H-3a"], "sufficient")
        self.assertEqual(v["H-4a"], "charges it")
        self.assertEqual(v["H-4b"], "does not")
        self.assertEqual(v["H-4c"], "does not")

    def test_no_residue_makes_h1_h2_unscorable(self):
        c = predicted_cells()
        for m in ("zero256", "zero256_wait", "zero256_read", "zeroL2"):
            c[("gemv4", m)] = (4 * L2, 0.01 * L2)
        v = self.run_score(c)
        self.assertFalse(v["H-0e"])
        self.assertTrue(v["H-1"].startswith("not scorable"))
        self.assertTrue(v["H-2"].startswith("not scorable"))

    def test_gate_fails_on_correctness_and_control(self):
        self.assertFalse(self.run_score(predicted_cells(), fails=1)["H-0"])
        v = self.run_score(predicted_cells(), ctl_write=float(128 << 10))
        self.assertFalse(v["H-0"])
        self.assertTrue(v["H-1"].startswith("not scored"))

    def test_h3_unscorable_when_big_read_flush_does_not_cool(self):
        c = predicted_cells()
        c[("e_tiny", "pr256")] = (0.5 * L2 / 128, 0.0)
        v = self.run_score(c)
        self.assertTrue(v["H-3a"].startswith("not scorable"))

    def test_partial_band(self):
        c = predicted_cells()
        c[("gemv4", "zero256_wait")] = (4 * L2, 0.2 * L2)          # ratio 0.5
        c[("e_tiny", "prL2")] = (0.75 * L2 / 128, 0.0)
        v = self.run_score(c)
        self.assertEqual(v["H-1"], "partial")
        self.assertEqual(v["H-3a"], "partial")

    def test_misaligned_rows_raise(self):
        m, _, _ = build(predicted_cells(), runs=1)
        with self.assertRaises(ValueError):
            S.range_rows(m, [[{"read": 0.0, "write": 0.0}]])


if __name__ == "__main__":
    unittest.main()
