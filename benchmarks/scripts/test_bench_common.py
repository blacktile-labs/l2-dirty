"""Unit tests for bench_common (CPU only; no GPU needed).

Run:  python3 -m unittest discover -s benchmarks/scripts -p 'test_*.py' -v
"""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import bench_common as bc


class TestStats(unittest.TestCase):
    def test_percentile_matches_numpy_linear(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        # numpy.percentile([1,2,3,4], 95) == 3.85 ; 50 -> 2.5 ; 0 -> 1 ; 100 -> 4
        self.assertAlmostEqual(bc.percentile(xs, 95), 3.85)
        self.assertAlmostEqual(bc.percentile(xs, 50), 2.5)
        self.assertEqual(bc.percentile(xs, 0), 1.0)
        self.assertEqual(bc.percentile(xs, 100), 4.0)
        self.assertEqual(bc.percentile([7.0], 95), 7.0)

    def test_percentile_order_independent(self):
        self.assertEqual(bc.percentile([4, 1, 3, 2], 50), bc.percentile([1, 2, 3, 4], 50))

    def test_percentile_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bc.percentile([], 50)
        with self.assertRaises(ValueError):
            bc.percentile([1.0], 101)

    def test_summarize(self):
        s = bc.summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertEqual(s.n, 8)
        self.assertEqual(s.median, 4.5)
        self.assertEqual(s.min, 2.0)
        self.assertEqual(s.max, 9.0)
        self.assertAlmostEqual(s.mean, 5.0)
        self.assertAlmostEqual(s.stdev, math.sqrt(32 / 7))  # sample stdev
        self.assertAlmostEqual(s.cv, math.sqrt(32 / 7) / 5.0)
        self.assertEqual(bc.summarize([3.0]).stdev, 0.0)


class TestTrafficModel(unittest.TestCase):
    N = 16 * 1024 * 1024  # 64 MiB of uint32

    def test_stride1_reads_whole_buffer(self):
        for g in (32, 64, 128):
            self.assertEqual(bc.model_strided_read_bytes(self.N, 1, 4, g), 4 * self.N)

    def test_flat_region_until_span_reaches_g(self):
        # With g = 32 B, strides 1..8 touch every sector -> full buffer read.
        for s in (1, 2, 4, 8):
            self.assertEqual(bc.model_strided_read_bytes(self.N, s, 4, 32), 4 * self.N)
        # Beyond that, one 32 B sector per element.
        self.assertEqual(bc.model_strided_read_bytes(self.N, 16, 4, 32), 32 * (self.N // 16))
        self.assertEqual(bc.model_strided_read_bytes(self.N, 32, 4, 32), self.N)

    def test_granularity_changes_prediction_at_stride16(self):
        # This is the discriminating point for H1b: g=32 -> 2N, g=64 -> 4N bytes.
        self.assertEqual(bc.model_strided_read_bytes(self.N, 16, 4, 32), 2 * self.N)
        self.assertEqual(bc.model_strided_read_bytes(self.N, 16, 4, 64), 4 * self.N)

    def test_amplification(self):
        self.assertAlmostEqual(bc.amplification(self.N, 1, 4, 32), 1.0)
        # stride 8: read 4N, write 4N/8 -> (4N + N/2) / (2 * 4N/8) = 4.5
        self.assertAlmostEqual(bc.amplification(self.N, 8, 4, 32), 4.5)
        # stride 32: read N (32 B per element), write N/8 -> (N + N/8)/(N/4) = 4.5
        self.assertAlmostEqual(bc.amplification(self.N, 32, 4, 32), 4.5)

    def test_brute_force_agrees_with_closed_form(self):
        # Enumerate touched chunks explicitly on small sizes.
        for n_src in (1, 7, 64, 1000, 4099):
            for s in (1, 2, 3, 5, 8, 9, 16, 33):
                for g in (32, 64, 128):
                    n_out = n_src // s
                    chunks = {(i * s * 4) // g for i in range(n_out)}
                    self.assertEqual(
                        bc.model_strided_read_bytes(n_src, s, 4, g), len(chunks) * g,
                        msg=f"n_src={n_src} s={s} g={g}",
                    )

    def test_rejects_violated_assumptions(self):
        with self.assertRaises(ValueError):
            bc.model_strided_read_bytes(100, 1, 4, 2)  # g < elem_bytes
        with self.assertRaises(ValueError):
            bc.model_strided_read_bytes(100, 1, 4, 30)  # g not multiple of elem

    def test_theoretical_bw(self):
        # Arithmetic check only: 7001 MHz x 2 x 192 bit / 8 = 336.048 GB/s
        self.assertAlmostEqual(bc.theoretical_dram_bw_gbps(7_001_000, 192), 336.048, places=3)
        self.assertTrue(math.isnan(bc.theoretical_dram_bw_gbps(-1, 192)))


class TestCsv(unittest.TestCase):
    def test_load_groups_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(bc.REPS_COLUMNS)
                # Synthetic fixture rows for parser testing only -- not measurements.
                w.writerow(["b", "k", 4, 8, 2, 4, 16, 16, 256, 1, 1, 0, 0.5])
                w.writerow(["a", "k", 4, 8, 1, 8, 32, 32, 256, 1, 1, 0, 0.25])
                w.writerow(["b", "k", 4, 8, 2, 4, 16, 16, 256, 1, 1, 1, 0.75])
            groups = bc.load_reps_csv(p)
            self.assertEqual([g.variant for g in groups], ["b", "a"])
            self.assertEqual(groups[0].times_ms, [0.5, 0.75])
            self.assertEqual(groups[0].logical_bytes, 32)

    def test_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.csv"
            p.write_text("variant,time_ms\nx,1.0\n")
            with self.assertRaises(ValueError):
                bc.load_reps_csv(p)


if __name__ == "__main__":
    unittest.main()
