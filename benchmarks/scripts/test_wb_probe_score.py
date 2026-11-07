"""Unit tests for wb_probe_score (CPU only; SYNTHETIC fixtures, no measured data)."""

import unittest

import wb_probe_score as ws

L2 = 3 * 2**20
STEADY = 5


def grid():
    cells = [(ri / 8, wi / 8) for ri in range(8) for wi in range(8) if ri + wi <= 7 and (ri or wi)]
    return cells + [(2.0, 0.0), (0.0, 2.0)]


def meta_for(cells):
    n = lambda f: int(f * L2 / 4) // 256 * 256 if f else 0
    return {"steady": STEADY, "cells": [{"r": r, "w": w, "nc": n(r), "nd": n(w)} for r, w in cells]}


def launches(meta, wb_fn, rb_fn=lambda r, w, wb: 0.0, gap=1.0):
    """One synthetic kernel-mode run and one range-mode run. wb_fn(r, w) = written fraction."""
    rows, rng, i = [], [], 0

    def add(kernel, read, write):
        nonlocal i
        rows.append({"id": i, "kernel": kernel, "read": read, "write": write})
        i += 1

    add("fill_u32_kernel", 0, 1e6)  # utility launch before the first flush must be skipped
    for c in meta["cells"]:
        br, bw = 4 * c["nc"], 4 * c["nd"]
        over = c["r"] + c["w"] > 1
        wb = 1.0 if over else wb_fn(c["r"], c["w"])
        rd = 1.0 if over else 0.0
        add("fill_u32_kernel", 0, 1e6)
        add("flush_l2_read_kernel", 16 * 2**20, 0)
        add("rw_kernel", br, 0)
        for _ in range(STEADY):
            add("rw_kernel", rd * br, wb * bw)
        add("readback_kernel", rb_fn(c["r"], c["w"], wb) * bw, 0)
        rng.append({"id": len(rng), "kernel": "range", "read": rd * br * (STEADY - 1),
                    "write": gap * wb * bw * (STEADY - 1)})
    return rows, rng


def run(wb_fn, **kw):
    meta = meta_for(grid())
    k, r = launches(meta, wb_fn, **kw)
    cells = ws.summarize(meta, [ws.align_kernel(k, len(meta["cells"]), STEADY)],
                         [ws.align_range(r, len(meta["cells"]))])
    return ws.score(cells, log=lambda *_: None)


class TestRules(unittest.TestCase):
    def test_lru_world(self):
        v = run(lambda r, w: 0.0)
        self.assertTrue(v["W-0"])
        self.assertFalse(v["W-1 LRU rejected"])
        self.assertTrue(v["W-2"].startswith("H-D"))  # no write-back anywhere: r has no effect
        self.assertEqual(v["W-4"], "n/a (no active cell)")

    def test_dirty_cap_world(self):  # H-D: depends on w only
        v = run(lambda r, w: 0.0 if w <= 0.375 else 0.6)
        self.assertTrue(v["W-1 LRU rejected"])
        self.assertTrue(v["W-2"].startswith("H-D"))
        self.assertEqual(v["W-3"], "flat rows (H-D)")
        self.assertEqual(v["W-4"], "cleaned, stays resident")
        self.assertEqual(v["W-5"], "not gap-driven")

    def test_total_footprint_world(self):  # H-E: depends on r + w only
        v = run(lambda r, w: 0.0 if r + w <= 0.5 else 0.5)
        self.assertTrue(v["W-2"].startswith("H-E"))
        self.assertEqual(v["W-3"], "rising rows (H-E)")

    def test_eviction_and_gaps(self):
        v = run(lambda r, w: 0.5 if w >= 0.5 else 0.0, rb_fn=lambda r, w, wb: wb, gap=0.2)
        self.assertEqual(v["W-4"], "evicted")
        self.assertEqual(v["W-5"], "gap-driven")

    def test_misaligned_sequence_raises(self):
        meta = meta_for([(0.25, 0.25)])
        rows, _ = launches(meta, lambda r, w: 0.0)
        del rows[4]  # drop one steady launch
        with self.assertRaises(ValueError):
            ws.align_kernel(rows, 1, STEADY)


if __name__ == "__main__":
    unittest.main()
