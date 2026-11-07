"""Unit tests for cache_sim_v4 (CPU only; SYNTHETIC access sequences, no measured data)."""

import random
import unittest

from cache_sim import Cache
from cache_sim_v4 import DirtyCapLRU


def probe(cap: int, r: float, w: float, steady: int = 5):
    """wb_probe launch sequence (flush, warm, steady x S, readback) as line accesses.
    Returns per-launch (reads, writebacks) for warm, steady..., readback."""
    c = DirtyCapLRU(cap)
    nc, nd, nf = int(r * cap), int(w * cap), 6 * cap
    C, D, F = 0, 10 * cap, 20 * cap
    for j in range(nf):
        c.access(F + j, False)
    out = []
    for _ in range(1 + steady):
        c.reset_counters()
        for j in range(max(nc, nd)):
            if j < nc:
                c.access(C + j, False)
            if j < nd:
                c.access(D + j, True)
        out.append((c.reads, c.writebacks))
    c.reset_counters()
    for j in range(nd):
        c.access(D + j, False)
    out.append((c.reads, c.writebacks))
    return out


class TestEqualsLockedLRUWithoutCap(unittest.TestCase):
    def test_random_sequences(self):
        rng = random.Random(7)
        for cap in (1, 4, 64):
            a, b = Cache(cap, "lru"), DirtyCapLRU(cap, dirty_cap_lines=cap)
            for _ in range(20000):
                line, write = rng.randrange(3 * cap), rng.random() < 0.4
                a.access(line, write)
                b.access(line, write)
            self.assertEqual((a.reads, a.writebacks), (b.reads, b.writebacks))
            self.assertEqual(b.cleanings, 0)


class TestReproducesB6Shape(unittest.TestCase):
    CAP = 1024

    def steady_wb(self, r, w):
        res = probe(self.CAP, r, w)
        nd = int(w * self.CAP)
        return [wb / nd for _, wb in res[2:-1]], res

    def test_step_in_w_independent_of_r(self):
        for w in (0.125, 0.25, 0.375, 0.4375, 0.5):
            for r in (0.0, 0.25, 0.375):
                if r + w <= 0.875:
                    wbs, res = self.steady_wb(r, w)
                    self.assertEqual(max(wbs), 0.0, (r, w))
        for w in (0.5625, 0.625, 0.75):
            for r in (0.0, 0.125):
                if r + w <= 0.875:
                    wbs, res = self.steady_wb(r, w)
                    self.assertTrue(all(x == 1.0 for x in wbs), (r, w, wbs))

    def test_cleaned_lines_stay_resident_and_clean_reads_hit(self):
        _, res = self.steady_wb(0.125, 0.625)
        self.assertEqual(res[-1][0], 0)                       # readback: no read miss
        self.assertTrue(all(rd == 0 for rd, _ in res[1:-1]))  # steady: C stays resident


if __name__ == "__main__":
    unittest.main()
