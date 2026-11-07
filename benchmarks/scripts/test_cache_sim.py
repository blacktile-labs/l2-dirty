"""Unit tests for cache_sim (B4 model v2). CPU only; small caches keep them fast."""

import unittest

import cache_sim as cs

L2 = 64 * 1024  # 512 lines


def n_for(l2x):
    return int(l2x * L2 / 4)


class TestCache(unittest.TestCase):
    def test_lru_eviction_and_writeback(self):
        c = cs.Cache(2, "lru")
        c.access(1, True)    # write-allocate, no read
        c.access(2, False)   # read miss
        c.access(1, False)   # hit, 1 becomes MRU
        c.access(3, False)   # evicts 2 (clean)
        self.assertEqual((c.reads, c.writebacks), (2, 0))
        c.access(4, False)   # evicts 1 (dirty)
        self.assertEqual((c.reads, c.writebacks), (3, 1))

    def test_random_policy_deterministic_and_bounded(self):
        def run():
            c = cs.Cache(8, "random", seed=7)
            for i in range(100):
                c.access(i % 13, i % 3 == 0)
            return c.reads, c.writebacks, len(c.lines)
        self.assertEqual(run(), run())
        self.assertEqual(run()[2], 8)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            cs.Cache(0, "lru")
        with self.assertRaises(ValueError):
            cs.Cache(4, "fifo")


class TestFusionSim(unittest.TestCase):
    def test_everything_fits(self):
        n = n_for(0.25)  # B = L2/4: x, t, y all fit
        b = 4 * n
        for pol in ("lru", "random"):
            s = cs.simulate_fusion(n, L2, pol)
            self.assertEqual(s[("fused", "isolated")], b)          # x read once, y deferred
            self.assertEqual(s[("materialized", "isolated")], b)   # t never leaves L2
            self.assertEqual(s[("stage2_warm", "isolated")], 0)
            self.assertEqual(s[("stage2_cold", "isolated")], b)
            self.assertEqual(s[("fused", "steady")], 0)
            self.assertEqual(s[("materialized", "steady")], 0)

    def test_lru_exact_fit_boundary(self):
        # 2B == L2: x and t exactly fill the cache, so stage2 finds all of t (LRU)
        s = cs.simulate_fusion(n_for(0.5), L2, "lru")
        self.assertEqual(s[("stage2_warm", "isolated")], 0)

    def test_large_sizes_approach_two(self):
        for pol in ("lru", "random"):
            r = cs.ratios(cs.simulate_fusion(n_for(16), L2, pol))
            self.assertAlmostEqual(r["mat/fused_iso"], 2.0, delta=0.05)
            self.assertAlmostEqual(r["mat/fused_steady"], 2.0, delta=0.05)
            self.assertAlmostEqual(r["warm/cold"], 1.0, delta=0.05)

    def test_warm_pays_dirty_writebacks(self):
        # just above the cliff, stage2 after stage1 must evict stage1's dirty t lines
        s = cs.simulate_fusion(n_for(1.0), L2, "lru")
        self.assertGreater(s[("stage2_warm", "isolated")], s[("stage2_cold", "isolated")])


class TestTimeModel(unittest.TestCase):
    def test_units(self):
        sim = {("fused", "isolated"): 333_500, ("materialized", "steady"): 0}
        t = cs.predict_times_us(sim, tau_iso_us=3.0, tau_steady_us=2.0, bw_gbps=333.5)
        self.assertAlmostEqual(t[("fused", "isolated")], 4.0)        # 3 + 1 us
        self.assertAlmostEqual(t[("materialized", "steady")], 4.0)   # 2 kernels x 2 us

    def test_calibrate(self):
        pts = [("fused", "isolated", 4.0, 333_500), ("materialized", "isolated", 8.0, 667_000),
               ("fused", "steady", 3.0, 333_500)]
        tau = cs.calibrate_tau(pts, 333.5)
        self.assertAlmostEqual(tau["isolated"], 3.0)
        self.assertAlmostEqual(tau["steady"], 2.0)


if __name__ == "__main__":
    unittest.main()
