"""Unit tests for counter_v4 (CPU only; SYNTHETIC 'measurements' built from the predictions)."""

import unittest

import counter_prereg as cp
import counter_v4 as cv

L2 = 64 * 1024
SIZES = (0.375, 2.0, 3.0)


def prereg(bench):
    rows = []
    for f in SIZES:
        n = cp.n_for(bench, f, L2)
        cells = cv.cells_of(cv.simulate(bench, n, L2, "v3"), cv.simulate(bench, n, L2, "v4"), 4 * n)
        rows.append({"l2x": f, "n": n, "array_bytes": 4 * n, "cells": cells})
    return {"sizes_l2x": list(SIZES), "rows": rows}


def fake_runs(bench, model, n_runs=3):
    """Measured launches == the chosen model's per-launch prediction (flush reads included)."""
    run = []
    for f in SIZES:
        n = cp.n_for(bench, f, L2)
        run.append([{"kernel": L["kernel"], "read": L["read"], "write": L["write"]}
                    for L in cv.simulate(bench, n, L2, model)])
    return [run] * n_runs


class TestScoring(unittest.TestCase):
    def test_v4_world(self):
        for bench in ("fusion", "stencil"):
            v = cv.score_bench(prereg(bench), fake_runs(bench, "v4"), cp.flush_bytes(L2), log=lambda *_: None)
            self.assertTrue(v["S-0"])
            self.assertEqual(v["K-1"], "v4 better", bench)
            self.assertTrue(v["K-2"])
            self.assertTrue(v["K-3"])

    def test_v3_world(self):
        v = cv.score_bench(prereg("fusion"), fake_runs("fusion", "v3"), cp.flush_bytes(L2), log=lambda *_: None)
        self.assertEqual(v["K-1"], "v3 better")
        self.assertFalse(v["K-2"])
        self.assertFalse(v["K-3"])

    def test_discriminating_cells_exist_only_below_the_cliff(self):
        rec = prereg("fusion")
        disc = {row["l2x"]: sum(c["disc"] for c in row["cells"]) for row in rec["rows"]}
        self.assertGreater(disc[0.375], 0)
        self.assertEqual(disc[2.0], 0)
        self.assertEqual(disc[3.0], 0)


if __name__ == "__main__":
    unittest.main()
