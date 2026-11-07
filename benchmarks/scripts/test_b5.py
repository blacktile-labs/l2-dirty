"""Unit tests for the B5 model (cache_sim_b5, b5_model) and the stencil summary. CPU only;
fixtures are SYNTHETIC."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import bench_common as bc
import b5_model
import cache_sim_b5 as b5
import run_bench

L2 = 64 * 1024  # small cache keeps the simulation fast


class TestSizes(unittest.TestCase):
    def test_n_for_matches_stencil_cu_rounding(self):
        # stencil.cu: h = max(8, floor(f*L2/4/W) // 8 * 8); n = h * W
        self.assertEqual(b5.n_for(4, 2_359_296), 2304 * 1024)
        self.assertEqual(b5.n_for(0.0625, 2_359_296), 32 * 1024)
        self.assertEqual(b5.n_for(1e-6, 2_359_296), 8 * 1024)  # floor of 8 rows


class TestSim(unittest.TestCase):
    def test_tiny_fits_everything(self):
        n = 1024 * 4  # 16 KiB per buffer, all fit
        s = b5.simulate_stencil(n, L2, "lru")
        self.assertEqual(s[("mat4", "isolated")], 4 * n)   # X read once; P, Q stay resident
        self.assertEqual(s[("fused2", "isolated")], 4 * n)
        self.assertEqual(s[("sweep_warm", "isolated")], 0)
        self.assertEqual(s[("mat4", "steady")], 0)

    def test_large_ratios(self):
        n = 16 * L2 // 4
        for pol in ("lru", "random"):
            s = b5.simulate_stencil(n, L2, pol)
            self.assertAlmostEqual(s[("mat4", "isolated")] / s[("fused2", "isolated")], 2.0, delta=0.05)
            self.assertAlmostEqual(s[("mat4", "steady")] / s[("fused2", "steady")], 2.0, delta=0.05)

    def test_warm_cold_equals_b4_pattern(self):
        # sweep_warm/cold has the same line pattern as B4 stage2_warm/cold
        import cache_sim
        n = L2 // 4  # B = L2
        s5 = b5.simulate_stencil(n, L2, "lru")
        s4 = cache_sim.simulate_fusion(n, L2, "lru")
        self.assertEqual(s5[("sweep_warm", "isolated")], s4[("stage2_warm", "isolated")])
        self.assertEqual(s5[("sweep_cold", "isolated")], s4[("stage2_cold", "isolated")])


class TestTimeModel(unittest.TestCase):
    def test_calibrate_and_predict(self):
        pts = [("mat4", "isolated", 4 * 3.0 + 1.0, 333_500), ("sweep_cold", "isolated", 3.0 + 1.0, 333_500),
               ("fused2", "isolated", 2 * 5.0 + 1.0, 333_500), ("mat4", "steady", 4 * 2.0, 0)]
        tau = b5.calibrate_tau(pts, 333.5)
        self.assertAlmostEqual(tau[("5pt", "isolated")], 3.0)
        self.assertAlmostEqual(tau[("13pt", "isolated")], 5.0)
        self.assertAlmostEqual(tau[("5pt", "steady")], 2.0)
        t = b5.predict_times_us({("fused2", "isolated"): 667_000}, tau, 333.5)
        self.assertAlmostEqual(t[("fused2", "isolated")], 2 * 5.0 + 2.0)


def write_fixture(d: Path):
    p = "W=1024;H=576;n=589824;l2x=1.000000;S=4;l2_bytes=2359296"
    rows = []
    for v, mode, ms in (("mat4", "isolated", 0.040), ("fused2", "isolated", 0.020), ("mat4", "steady", 0.030),
                        ("fused2", "steady", 0.015), ("sweep_cold", "isolated", 0.010),
                        ("sweep_warm", "isolated", 0.012)):
        rows += [["stencil", v, p, 589824, 2359296, 2359296, mode, 1, 1, r, ms] for r in range(2)]
    with open(d / "reps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(bc.GENERIC_COLUMNS)
        w.writerows(rows)
    (d / "meta.json").write_text(json.dumps({
        "benchmark": "stencil", "device_name": "SYNTHETIC", "l2_cache_bytes": 2359296, "mem_clock_khz_attr": 1,
        "mem_bus_width_bits": 1, "sm_count": 1, "sm_clock_khz_attr": 1, "correctness": {}, "correctness_failures": 0}))


class TestSummaries(unittest.TestCase):
    def test_stencil_summary_and_measured_ratios(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_fixture(d)
            with contextlib.redirect_stdout(io.StringIO()):
                run_bench.summarize("stencil", d)
            md = (d / "summary.md").read_text()
            self.assertIn("| 1.000000 | 576 | 2.250 | 40.000 | 20.000 | 2.000 | nan | 2.000 | 1.200 |", md)
            _, med = b5_model.measured_medians(d)
            r = b5_model.measured_ratios(med, 1.0)
            self.assertAlmostEqual(r["warm/cold"], 1.2)
            self.assertAlmostEqual(r["mat4/fused2_iso"], 2.0)


if __name__ == "__main__":
    unittest.main()


class TestCompareB4(unittest.TestCase):
    def test_g4_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as t:
            d5 = Path(t) / "b5"; d4 = Path(t) / "b4"; d5.mkdir(); d4.mkdir()
            write_fixture(d5)  # B5 warm/cold = 1.2 at l2x 1
            p4 = "n=589824;l2x=1.000000;K=1;l2_bytes=2359296"
            rows = [["fusion", v, p4, 589824, 1, 1, "isolated", 1, 1, r, ms]
                    for v, ms in (("stage2_cold", 0.010), ("stage2_warm", 0.0115)) for r in range(2)]
            with open(d4 / "reps.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(bc.GENERIC_COLUMNS); w.writerows(rows)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                b5_model.compare_b4([d5], d4)
            self.assertIn("PASS", buf.getvalue())   # 1.2 vs 1.15 -> 4.3 %
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                b5_model.compare_b4([d5], d4, tol_pct=3.0)
            self.assertIn("FAIL", buf.getvalue())
