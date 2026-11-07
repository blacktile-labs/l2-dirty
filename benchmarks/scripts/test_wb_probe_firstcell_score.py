import unittest

import wb_probe_firstcell_score as fc


def cell(r, w, wb, wbr):
    return {"r": r, "w": w, "WB": wb, "WBR": wbr, "RB": 0.0, "RD": None, "RDR": None, "WARM": None}


def sector(dummy_wbr, ctrl_wbr, test_wbr, anchor=(0.97, 0.99)):
    cells = [cell(0, 0.25, 0.0, dummy_wbr), cell(0, 0.375, 0.0, ctrl_wbr), cell(0, 2, *anchor)]
    cells += [cell(0, w, 0.9, x) for w, x in zip((0.625, 0.75, 0.875), test_wbr)]
    return {fc.k(c["r"], c["w"]): c for c in cells}


class FirstCellScore(unittest.TestCase):
    def run_score(self, *a, fails=0):
        return fc.score(sector(*a), fails, log=lambda *_: None)

    def test_cells_are_b6b_sector_set_with_dummy_first(self):
        import wb_probe_b6b_score as b6b
        self.assertEqual(fc.CELLS, "0:0.25," + b6b.SECTOR_CELLS)

    def test_artifact_supported_and_lines(self):
        v = self.run_score(0.14, 0.01, (0.57, 0.95, 0.97))
        self.assertTrue(v["F-0"])
        self.assertEqual(v["F-1"], "first-cell artifact supported")
        self.assertEqual(v["F-2"], "inflation present in the first cell")
        self.assertEqual(v["V-4'"], "counts lines")

    def test_control_still_active(self):
        v = self.run_score(0.01, 0.14, (0.57, 0.95, 0.97))
        self.assertEqual(v["F-1"], "control active without first-cell position")
        self.assertEqual(v["F-2"], "no first-cell inflation")
        self.assertTrue(v["V-4'"].startswith("control failed"))

    def test_inconclusive_band_and_bytes(self):
        self.assertEqual(self.run_score(0.0, 0.07, (0.0, 0.0, 0.0))["F-1"], "inconclusive")
        self.assertEqual(self.run_score(0.0, 0.0, (0.0, 0.01, 0.0))["V-4'"], "counts bytes/sectors")
        self.assertEqual(self.run_score(0.0, 0.0, (0.3, 0.5, 0.6))["V-4'"], "mixed")

    def test_sanity_fails_on_correctness_or_anchor(self):
        self.assertFalse(self.run_score(0.0, 0.0, (1, 1, 1), fails=1)["F-0"])
        self.assertFalse(fc.score(sector(0.0, 0.0, (1, 1, 1), anchor=(0.5, 0.99)), 0, log=lambda *_: None)["F-0"])


if __name__ == "__main__":
    unittest.main()
