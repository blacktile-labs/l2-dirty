"""Unit tests for cache_sim_v5cand and wb_probe_b8_score (CPU only; SYNTHETIC data)."""

import random
import unittest

import wb_probe_b8_score as b8
import wb_probe_score as ws
from cache_sim_v4 import DirtyCapLRU
from cache_sim_v5cand import CandidateL2


class TestCandidates(unittest.TestCase):
    def test_lru_policy_equals_v4(self):
        rng = random.Random(5)
        for cap in (8, 64):
            a, b = DirtyCapLRU(cap), CandidateL2(cap, "lru")
            for _ in range(30000):
                line, write = rng.randrange(3 * cap), rng.random() < 0.4
                a.access(line, write)
                b.access(line, write)
            self.assertEqual((a.reads, a.writebacks), (b.reads, b.writebacks))

    def test_dvsc_life_equals_lru_without_writes(self):
        rng = random.Random(9)
        for cap in (8, 64):
            a, b = CandidateL2(cap, "lru"), CandidateL2(cap, "dvsc_life", life_frac=0.2)
            for _ in range(20000):
                line = rng.randrange(3 * cap)
                a.access(line, False)
                b.access(line, False)
            self.assertEqual(a.reads, b.reads)

    def test_dirty_victim_written_back_but_readable(self):
        c = CandidateL2(4, "dvsc_life", life_frac=0.5, dirty_cap_lines=4)  # life = 2 fills
        c.access(1, True)
        for x in (2, 3, 4, 5):   # 5 forces an eviction: 1 is dirty LRU -> written back, parked
            c.access(x, False)
        self.assertEqual(c.writebacks, 1)
        before = c.reads
        c.access(1, False)       # still readable
        self.assertEqual(c.reads, before)


PRED = {"cells": []}
for r, w in b8.CELLS:
    over = r + w > 1
    PRED["cells"].append({"r": r, "w": w,
                          "v4": {"RD": 1.0 if over else 0.0, "WB": 1.0 if over else 0.0, "RB": 1.0 if over else 0.0},
                          "dvsc": {"RD": 1.0 if over else 0.0, "WB": 1.0 if over else 0.0,
                                   "RB": (0.0 if r + w < 1.2 else 1.0) if over else 0.0}})


def summary(rb_fn, rd_fn=lambda r: 0.0 if r <= 1 else 1.0):
    out = {}
    for r, w in b8.CELLS:
        over = r + w > 1
        rd = (rd_fn(r) if w == 0 else (1.0 if over else 0.0)) if r else None
        wb = (1.0 if over else 0.0) if w else None
        rb = (rb_fn(r, w) if over else 0.0) if w else None
        out[ws.key(r, w)] = {"r": r, "w": w, "WB": wb, "WBR": wb, "RB": rb, "RD": rd, "RDR": rd,
                             "WARM": 1.0 if r else None}
    out[ws.key(1.0, 0.125)]["RB"] = 0.085
    return out


class TestRules(unittest.TestCase):
    def q(self, *a, **k):
        return b8.score(summary(*a, **k), PRED, log=lambda *_: None)

    def test_dvsc_world(self):
        v = self.q(lambda r, w: 0.0 if r + w < 1.2 else 1.0)
        self.assertTrue(v["Z-0"])
        self.assertTrue(v["Z-0 replication"])
        self.assertEqual(v["Z-1"], "dvsc supported")
        self.assertEqual(v["Z-2"], "effective capacity >= 0.94 L2")
        self.assertEqual(v["Z-2b"], "LRU-sharp cliff")

    def test_lru_world_and_small_capacity(self):
        v = self.q(lambda r, w: 1.0, rd_fn=lambda r: 0.5 if r >= 0.93 else 0.0)
        self.assertEqual(v["Z-1"], "v4 (LRU) supported")
        self.assertEqual(v["Z-2"], "effective capacity < 0.94 L2")
        self.assertTrue(v["Z-2b"].startswith("reads retained"))


if __name__ == "__main__":
    unittest.main()
