"""Unit tests for the B2-B4 traffic models, the generic CSV loader and run_bench summaries.

CPU only. All CSV/JSON fixtures below are SYNTHETIC test inputs, not measurements.
Run: python3 -m unittest discover -s benchmarks/scripts -p 'test_*.py' -v
"""

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import bench_common as bc
import run_bench

L2 = 2_359_296


class TestTransposeModel(unittest.TestCase):
    def test_dram_same_for_all_variants(self):
        for v in ("copy_2d", "naive", "tiled", "tiled_pad"):
            self.assertEqual(bc.model_transpose(4096, 4096, v)["dram_bytes"], 8.0 * 4096 * 4096)

    def test_naive_store_sectors_8x(self):
        c = bc.model_transpose(1024, 512, "copy_2d")["l2_store_sectors"]
        n = bc.model_transpose(1024, 512, "naive")["l2_store_sectors"]
        self.assertAlmostEqual(n / c, 8.0)

    def test_bank_conflicts(self):
        self.assertEqual(bc.model_transpose(64, 64, "tiled")["smem_conflict_ways"], 32.0)
        self.assertEqual(bc.model_transpose(64, 64, "tiled_pad")["smem_conflict_ways"], 1.0)
        with self.assertRaises(ValueError):
            bc.model_transpose(64, 64, "bogus")


class TestGatherModel(unittest.TestCase):
    def test_seq(self):
        tb = bc.model_gather_table_bytes(m=1000, t=4000, dist="seq", l2_bytes=L2)
        self.assertEqual(tb["compulsory"], 4000.0)
        self.assertEqual(tb["no_reuse"], 4000.0)
        # m > t with a table larger than L2: the second pass misses again
        tb = bc.model_gather_table_bytes(m=2 * 10**6, t=10**6, dist="seq", l2_bytes=L2)
        self.assertEqual(tb["hot_set"], 8e6)
        # ... but hits if the table fits in L2
        tb = bc.model_gather_table_bytes(m=2000, t=1000, dist="seq", l2_bytes=L2)
        self.assertEqual(tb["hot_set"], 4000.0)

    def test_random_bounds_are_ordered(self):
        # Synthetic statistics consistent with each other.
        for distinct, share in ((1_700_000, 0.04), (300_000, 0.70), (10, 1.0)):
            tb = bc.model_gather_table_bytes(4_194_304, 16_777_216, "zipf1.0", L2, distinct, share)
            self.assertLessEqual(tb["compulsory"], tb["hot_set"] + 1e-6)
            self.assertLessEqual(tb["hot_set"], tb["no_reuse"] + 1e-6)

    def test_random_requires_stats(self):
        with self.assertRaises(ValueError):
            bc.model_gather_table_bytes(10, 100, "uniform", L2)

    def test_totals(self):
        tb = {"compulsory": 100.0, "hot_set": 200.0, "no_reuse": 300.0}
        self.assertEqual(bc.model_gather_bytes(10, tb)["hot_set"], 280.0)
        self.assertEqual(bc.model_scatter_add_bytes(10, tb)["hot_set"], 480.0)


class TestPairProbeModel(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(bc.model_pair_bytes_per_lookup(8, 32), 64)    # two 32-B sectors
        self.assertEqual(bc.model_pair_bytes_per_lookup(8, 64), 64)    # same 64-B chunk
        self.assertEqual(bc.model_pair_bytes_per_lookup(16, 64), 128)  # different 64-B halves
        self.assertEqual(bc.model_pair_bytes_per_lookup(16, 128), 128)

    def test_ratios_discriminate_granularity(self):
        r = {g: (bc.predicted_pair_ratio("gather_pair64", g), bc.predicted_pair_ratio("gather_pair128", g))
             for g in (32, 64, 128)}
        self.assertAlmostEqual(r[32][0], 1.8)
        self.assertAlmostEqual(r[32][1], 1.8)
        self.assertAlmostEqual(r[64][0], 1.0)
        self.assertAlmostEqual(r[64][1], 136 / 72)
        self.assertAlmostEqual(r[128][0], 1.0)
        self.assertAlmostEqual(r[128][1], 1.0)


class TestFusionModel(unittest.TestCase):
    def test_residency_models(self):
        b_small, b_mid, b_big = L2 // 4, L2, 4 * L2
        self.assertEqual(bc.fusion_residency(b_small, L2, "lru_cliff"), 1.0)
        self.assertEqual(bc.fusion_residency(b_mid, L2, "lru_cliff"), 0.0)
        self.assertEqual(bc.fusion_residency(b_mid, L2, "proportional"), 0.5)
        self.assertEqual(bc.fusion_residency(b_big, L2, "proportional"), 0.125)
        self.assertEqual(bc.fusion_residency(b_small, L2, "none"), 0.0)
        with self.assertRaises(ValueError):
            bc.fusion_residency(1, L2, "magic")

    def test_bytes(self):
        n = L2 // 4  # B = L2
        fu = bc.model_fusion_bytes(n, L2, "fused")
        self.assertEqual(fu["dram"], 2.0 * L2)
        self.assertEqual(bc.model_fusion_bytes(n, L2, "materialized", "none")["dram"], 4.0 * L2)
        self.assertEqual(bc.model_fusion_bytes(n, L2, "materialized", "proportional")["dram"], 3.0 * L2)
        self.assertEqual(bc.model_fusion_bytes(n, L2, "materialized")["l2_tier"], 4.0 * L2)
        tiny = 1024
        self.assertEqual(bc.model_fusion_bytes(tiny, L2, "stage2_warm", "lru_cliff")["dram"], 4.0 * tiny)
        self.assertEqual(bc.model_fusion_bytes(tiny, L2, "stage2_cold", "lru_cliff")["dram"], 8.0 * tiny)

    def test_flops_and_peak(self):
        self.assertEqual(bc.fusion_flops(10, 64, "fused"), 2560.0)
        self.assertEqual(bc.fusion_flops(10, 64, "stage2_warm"), 1280.0)
        # Arithmetic check only: 28 SM x 128 lanes x 2 x 1.777 GHz
        self.assertAlmostEqual(bc.fp32_peak_gflops(28, 1_777_000), 12737.5, places=1)
        self.assertTrue(math.isnan(bc.fp32_peak_gflops(0, 1)))


def write_fixture(d: Path, bench: str, rows: list[list], meta_extra: dict) -> None:
    with open(d / "reps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(bc.GENERIC_COLUMNS)
        w.writerows(rows)
    meta = {"benchmark": bench, "device_name": "SYNTHETIC", "compute_capability": "8.6",
            "mem_clock_khz_attr": 7_501_000, "mem_bus_width_bits": 192, "sm_count": 28,
            "sm_clock_khz_attr": 1_777_000, "l2_cache_bytes": L2, "reps": 2, "warmup": 0,
            "steady_iters": 1, "flush_l2": True, "correctness": {"x": 0}, "correctness_failures": 0}
    meta.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(meta))


def run_summary(bench: str, d: Path) -> str:
    with contextlib.redirect_stdout(io.StringIO()):
        run_bench.summarize(bench, d)
    return (d / "summary.md").read_text()


class TestGenericCsvAndSummaries(unittest.TestCase):
    def test_parse_params(self):
        self.assertEqual(bc.parse_params("R=4;C=5"), {"R": "4", "C": "5"})
        with self.assertRaises(ValueError):
            bc.parse_params("R4")

    def test_transpose_summary(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            p = "R=64;C=32"
            rows = []
            for v, ms in (("copy_2d", 0.010), ("naive", 0.020), ("tiled_pad", 0.0125)):
                for mode in ("isolated", "steady"):
                    rows += [["transpose", v, p, 2048, 8192, 8192, mode, 1, 1, r, ms] for r in range(2)]
            write_fixture(d, "transpose", rows, {})
            s = bc.load_generic_csv(d / "reps.csv")
            self.assertEqual(len(s), 6)
            md = run_summary("transpose", d)
            self.assertIn("| R=64;C=32 | isolated | naive | 20.000 | 10.000 | 50.000 |", md)
            self.assertIn("| R=64;C=32 | isolated | tiled_pad | 12.500 | 10.000 | 80.000 |", md)

    def test_fusion_summary(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            p = f"n=589824;l2x=1.000000;K=1;l2_bytes={L2}"
            rows = []
            for v, mode, ms in (("fused", "isolated", 0.010), ("materialized", "isolated", 0.018),
                                ("fused", "steady", 0.009), ("materialized", "steady", 0.0135),
                                ("stage2_cold", "isolated", 0.008), ("stage2_warm", "isolated", 0.006)):
                rows += [["fusion", v, p, 589824, 2359296, 2359296, mode, 1, 1, r, ms] for r in range(2)]
            write_fixture(d, "fusion", rows, {})
            md = run_summary("fusion", d)
            # mat/fused isolated 1.8, steady 1.5, warm/cold 0.75, lru_cliff r=0, proportional r=0.5
            self.assertIn("| 1.000000 | 589824 | 1 | 10.000 | 18.000 | 1.800 | 1.500 | 0.750 | 0.000 | 0.500 |", md)

    def test_gather_summary(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            p = "T=16777216;M=4194304;dist=uniform"
            rows = [["gather", v, p, 4194304, 1, 1, "isolated", 1, 1, r, ms]
                    for v, ms in (("gather", 1.0), ("scatter_add", 2.0), ("gather_pair64", 1.1),
                                  ("gather_pair128", 1.8)) for r in range(2)]
            stats = {p: {"distinct_sectors": 1_700_000, "hot_sectors_k": 73728,
                         "hot_sector_share": 0.04, "max_element_count": 3}}
            write_fixture(d, "gather", rows, {"index_stats": stats})
            md = run_summary("gather", d)
            self.assertIn("| gather | uniform | 16777216 | 1000.000 |", md)
            self.assertIn("scatter_add", md)
            self.assertIn("| gather_pair64 | uniform | 1100.000 | 1000.000 | 1.100 | 1.800 | 1.000 | 1.000 |", md)
            self.assertIn("### Fetch-granularity probe", md)


if __name__ == "__main__":
    unittest.main()
