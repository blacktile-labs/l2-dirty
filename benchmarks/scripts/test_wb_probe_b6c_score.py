"""Unit tests for cache_sim_cf and wb_probe_b6c_score (CPU only; SYNTHETIC data)."""

import random
import unittest

import wb_probe_b6c_score as b6c
import wb_probe_score as ws
from cache_sim import Cache
from cache_sim_cf import CleanFirstDirtyCap


class TestCleanFirstEqualsLRUWhenItCannotChoose(unittest.TestCase):
    def test_read_only_and_write_only(self):
        rng = random.Random(3)
        for cap in (4, 64):
            for write in (False, True):
                a, b = Cache(cap, "lru"), CleanFirstDirtyCap(cap, dirty_cap_lines=cap)
                for _ in range(20000):
                    line = rng.randrange(3 * cap)
                    a.access(line, write)
                    b.access(line, write)
                self.assertEqual((a.reads, a.writebacks), (b.reads, b.writebacks))

    def test_prefers_clean_victim(self):
        c = CleanFirstDirtyCap(2, dirty_cap_lines=2)
        c.access(1, True)    # oldest, dirty
        c.access(2, False)   # clean
        c.access(3, False)   # miss: LRU would evict dirty 1; clean-first evicts 2
        self.assertEqual(c.writebacks, 0)
        c.access(1, False)                # dirty 1 survived: hit
        self.assertEqual(c.reads, 2)      # only the read misses of 2 and 3

    def test_victim_bookkeeping(self):
        c = CleanFirstDirtyCap(2, dirty_cap_lines=2)
        c.access(1, True)
        c.access(2, False)
        c.access(3, False)
        hits_before = c.reads
        c.access(1, False)   # still resident
        self.assertEqual(c.reads, hits_before)


PRED = {"cells": [{"r": r, "w": w,
                   "v4": {"WB": 1.0 if (r + w > 1 and w) else 0.0, "RB": 1.0, "RD": 1.0},
                   "cf": {"WB": 0.0 if (r + w > 1 and w < 0.5) else (1.0 if r + w > 1 else 0.0), "RB": 0.0, "RD": 1.0}}
                  for r, w in b6c.CELLS]}


def summary(wb_mixed, rb_mixed=0.0, rd_mixed=1.0):
    out = {}
    for r, w in b6c.CELLS:
        over = r + w > 1
        mixed = (r, w) in b6c.MIXED
        wb = (wb_mixed if mixed else (1.0 if over else 0.0)) if w else None
        rd = ((rd_mixed if mixed else 1.0) if over else 0.0) if r else None
        out[ws.key(r, w)] = {"r": r, "w": w, "WB": wb, "WBR": wb, "RB": None if not w else (rb_mixed if mixed else 1.0),
                             "RD": rd, "RDR": rd, "WARM": 1.0 if r else None}
    return out


class TestRules(unittest.TestCase):
    def q(self, *a, **k):
        return b6c.score(summary(*a, **k), PRED, log=lambda *_: None)

    def test_clean_first_world(self):
        v = self.q(0.0, rb_mixed=0.0)
        self.assertTrue(v["X-0"])
        self.assertTrue(v["X-1"].startswith("clean-first"))
        self.assertEqual(v["X-2"], "full protection")
        self.assertEqual(v["X-3"], "resident")
        self.assertTrue(v["X-4"])

    def test_lru_world(self):
        v = self.q(1.0, rb_mixed=1.0)
        self.assertTrue(v["X-1"].startswith("LRU"))
        self.assertEqual(v["X-2"], "no protection")
        self.assertEqual(v["X-3"], "evicted")

    def test_partial_world(self):
        v = self.q(0.4, rb_mixed=0.5, rd_mixed=0.6)
        self.assertTrue(v["X-1"].startswith("clean-first"))  # 0.4 is closer to 0 than to 1
        self.assertEqual(v["X-2"], "partial protection")
        self.assertEqual(v["X-3"], "partial")
        self.assertFalse(v["X-4"])


if __name__ == "__main__":
    unittest.main()
