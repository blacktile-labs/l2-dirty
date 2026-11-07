"""Unit tests for model_v3 (CPU only; SYNTHETIC fixtures)."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import bench_common as bc
import model_v3 as m3


class TestFit(unittest.TestCase):
    def test_exact_recovery(self):
        tau, beta = 3.0, 1 / 300e3  # 300 GB/s in bytes per us
        pts = [(k, b, k * tau + b * beta) for k, b in ((2, 10_000_000), (2, 20_000_000), (4, 20_000_000))]
        t, bt = m3.fit_tau_beta(pts)
        self.assertAlmostEqual(t, tau, places=6)
        self.assertAlmostEqual(1 / bt / 1e3, 300.0, places=3)

    def test_tau_nonnegative(self):
        # data whose unconstrained fit would give tau < 0 (time grows faster than linear in bytes)
        pts = [(1, 10_000_000, 30.0), (1, 20_000_000, 70.0)]
        t, bt = m3.fit_tau_beta(pts)
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(bt, (10e6 * 30 + 20e6 * 70) / (10e6 ** 2 + 20e6 ** 2))

    def test_singular(self):
        with self.assertRaises(ValueError):
            m3.fit_tau_beta([(1, 100, 5.0), (2, 200, 10.0)])  # collinear rows


class TestMapping(unittest.TestCase):
    def test_fused2c_shares_schedule(self):
        sim = m3.simulate("stencil", 1024 * 64, 64 * 1024, "lru")
        self.assertEqual(sim[("fused2c", "isolated")], sim[("fused2", "isolated")])
        self.assertEqual(sim[("fused2c", "steady")], sim[("fused2", "steady")])

    def test_n_for(self):
        self.assertEqual(m3.n_for("fusion", 1.0, 2_359_296), 589824)
        self.assertEqual(m3.n_for("stencil", 4.0, 2_359_296), 2304 * 1024)

    def test_predict_and_ratios(self):
        params = {("pw_fused", "isolated"): (2.0, 1e-3), ("pw_stage", "isolated"): (1.0, 1e-3)}
        sim = {("fused", "isolated"): 1000, ("materialized", "isolated"): 3000,
               ("stage2_cold", "isolated"): 1000, ("stage2_warm", "isolated"): 1500}
        t = m3.predict_times("fusion", sim, params)
        self.assertAlmostEqual(t[("fused", "isolated")], 3.0)
        self.assertAlmostEqual(t[("materialized", "isolated")], 5.0)
        r = m3.ratios("fusion", t)
        self.assertAlmostEqual(r["materialized/fused_iso"], 5 / 3)
        self.assertAlmostEqual(r["stage2_warm/stage2_cold_iso"], 2.5 / 2.0)


class TestCalibrateFromRun(unittest.TestCase):
    def test_calibrate_identifies_bw(self):
        # Build a synthetic stencil run whose times follow t = k*tau + bytes/BW exactly.
        L2 = 2_359_296
        tau = {"5pt": 2.0, "13pt": 5.0}
        bw = {"5pt": 320e3, "13pt": 200e3}  # bytes per us
        rows = []
        for f in (2.0, 4.0):
            n = m3.n_for("stencil", f, L2)
            sim = m3.simulate("stencil", n, L2, "lru")
            p = f"W=1024;H={n // 1024};n={n};l2x={f:.6f};S=4;l2_bytes={L2}"
            for (v, mode), b in sim.items():
                if v == "fused2c":
                    continue
                kt, k = m3.KMAP["stencil"][v]
                t_us = k * tau[kt] + b / bw[kt]
                rows += [["stencil", v, p, n, 1, 1, mode, 1, 1, r, t_us / 1e3] for r in range(2)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "reps.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(bc.GENERIC_COLUMNS); w.writerows(rows)
            (d / "meta.json").write_text(json.dumps({"l2_cache_bytes": L2}))
            meta, med = m3.measured_medians(d)
            params = m3.calibrate("stencil", med, L2)
            for kt in ("5pt", "13pt"):
                t, beta = params[(kt, "isolated")]
                self.assertAlmostEqual(t, tau[kt], places=3)
                self.assertAlmostEqual(1 / beta, bw[kt], delta=bw[kt] * 1e-6)


if __name__ == "__main__":
    unittest.main()
