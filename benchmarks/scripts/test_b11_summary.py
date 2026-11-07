import unittest
from pathlib import Path

import b11_summary as s

RES = Path(__file__).resolve().parent.parent / "results"
GA102 = RES / "b10_rtx3090_vm_20261106-0353" / "benchmarks" / "results"


def one(pattern: str, base: Path) -> Path:
    found = sorted(base.glob(pattern))
    assert len(found) == 1, (pattern, found)
    return found[0]


def row(name, c5n_state, c6_state, p0=True):
    return {"gpu": name, "P-0": p0, "C5n": c5n_state, "C6": c6_state}


class B11Summary(unittest.TestCase):
    def test_ga106_reference_matches_itself(self):
        r = s.gpu("GA106", RES / "wb_probe_b6b_20261105-063839", RES / "wb_probe_b8_20261105-093216",
                  RES / "queued_wb_20261105-140557", None)
        self.assertTrue(r["P-0"])
        self.assertEqual(r["n_match"], 9)
        self.assertTrue(r["C5n"].startswith("present"))
        self.assertEqual(r["C6"], "present")

    def test_ga102_reproduces_b10(self):
        r = s.gpu("GA102", one("wb_probe_b6b_*", GA102), one("wb_probe_b8_*", GA102),
                  one("queued_wb_*", GA102), None)
        self.assertTrue(r["P-0"])
        self.assertEqual(r["n_match"], 7)
        self.assertEqual({x for x, ok in r["match"].items() if not ok}, {"V-3", "V-4"})
        self.assertTrue(r["C5n"].startswith("present"))
        self.assertEqual(r["C6"], "present")

    def test_c5n_branches(self):
        cells = lambda lo, hi, ab: [{"r": 0, "w": 0.4375, "WBR": lo}, {"r": 0, "w": 0.625, "WBR": hi},
                                    {"r": 0, "w": 0.6875, "WBR": ab}]
        self.assertTrue(s.c5n(cells(0.05, 0.9, 1.0))[0].startswith("present"))
        self.assertTrue(s.c5n(cells(0.0, 0.1, 0.2))[0].startswith("absent"))
        self.assertEqual(s.c5n(cells(0.3, 0.9, 1.0))[0], "other")

    def test_first_cell_branches(self):
        sec = lambda c1: [{"r": 0, "w": 0.375, "WBR": c1}]
        fcv = lambda c2: {"F-1 control WBR": c2}
        self.assertTrue(s.first_cell(sec(0.01), fcv(0.2))[0].startswith("nothing to test"))
        self.assertEqual(s.first_cell(sec(0.143), fcv(0.01))[0], "first-cell artifact confirmed")
        self.assertEqual(s.first_cell(sec(0.143), fcv(0.14))[0], "not a first-cell artifact")
        self.assertEqual(s.first_cell(sec(0.143), fcv(0.07))[0], "inconclusive")

    def test_cross_rules(self):
        p, a = "present (transition in 0.44-0.63 L2)", "absent up to 0.69 L2"
        rows = [row("T", p, "present"), row("A", p, "present"), row("B", a, "absent")]
        self.assertTrue(s.cross(rows, "C5n", "present").startswith("generalizes (present on 2/3)"))
        rows = [row("T", a, "absent"), row("A", a, "absent")]
        self.assertTrue(s.cross(rows, "C5n", "present").startswith("Ampere-specific"))
        rows = [row("T", p, "present"), row("A", a, "absent")]
        self.assertTrue(s.cross(rows, "C5n", "present").startswith("mixed"))
        rows = [row("T", p, "present"), row("A", p, "present", p0=False)]
        self.assertTrue(s.cross(rows, "C5n", "present").startswith("insufficient"))


if __name__ == "__main__":
    unittest.main()
