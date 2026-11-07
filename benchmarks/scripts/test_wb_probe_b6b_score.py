"""Unit tests for wb_probe_b6b_score (CPU only; SYNTHETIC per-cell summaries, no measured data)."""

import unittest

import wb_probe_b6b_score as b6b
import wb_probe_score as ws


def cells(spec: str, fn, rb=lambda r, w, wb: 0.0, kernel_extra=0.0):
    """Synthetic summarize() output. fn(r, w) -> back-to-back write fraction (WBR)."""
    out = {}
    for tok in spec.split(","):
        r, w = (float(x) for x in tok.split(":"))
        over = r + w > 1
        wbr = (1.0 if over else fn(r, w)) if w else None
        wb = None if wbr is None else min(1.0, wbr + (0 if over else kernel_extra))
        out[ws.key(r, w)] = {"r": r, "w": w, "WB": wb, "WBR": wbr,
                             "RB": None if wb is None else rb(r, w, wb),
                             "RD": (1.0 if over else 0.0) if r else None,
                             "RDR": (1.0 if over else 0.0) if r else None,
                             "WARM": 1.0 if r else None}
    return out


def step(r, w):  # the B6 picture: dirty-only step at w ~ 0.5
    return 0.0 if w < 0.5 else (0.19 if w == 0.5 else 0.97)


def run(full_fn, sector_fn, **kw):
    return b6b.score(cells(b6b.FULL_CELLS, full_fn, **kw), cells(b6b.SECTOR_CELLS, sector_fn),
                     log=lambda *_: None)


class TestRules(unittest.TestCase):
    def test_dirty_only_lines_world(self):
        v = run(step, step)
        self.assertTrue(v["V-0"])
        self.assertTrue(v["V-1 replicates B6"])
        self.assertTrue(v["V-2"].startswith("H-D"))
        self.assertEqual(v["V-3"], "sharp step")
        self.assertEqual(v["V-4"], "counts lines")
        self.assertEqual(v["V-5"], "resident")

    def test_total_footprint_bytes_world(self):
        v = run(lambda r, w: 0.0 if r + w < 0.8 else 0.4, lambda r, w: 0.0)
        self.assertFalse(v["V-1 replicates B6"])
        self.assertTrue(v["V-2"].startswith("H-E"))
        self.assertEqual(v["V-4"], "counts bytes/sectors")

    def test_kernel_mode_noise_does_not_decide_v2(self):
        v = run(step, step, kernel_extra=0.3)  # only per-kernel WB is inflated
        self.assertTrue(v["V-2"].startswith("H-D"))

    def test_soft_step_and_eviction(self):
        v = run(lambda r, w: 0.0 if w < 0.4 else (0.3 if w < 0.6 else 0.9), step,
                rb=lambda r, w, wb: wb)
        self.assertEqual(v["V-3"], "not sharp")
        self.assertEqual(v["V-5"], "evicted")


if __name__ == "__main__":
    unittest.main()
