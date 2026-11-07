"""Unit tests for kernel_time_score.py (B23) on synthetic data."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import kernel_time_score as S

SHAPES = ("qkv", "o", "gate_up", "down")
KT = {"qkv": 34.0, "o": 15.0, "gate_up": 55.0, "down": 31.0}       # in-model kernel times, fp16
SP = {"fp16": 1.0, "int4": 1.8, "nf4": 1.2}


def build(rot=1.0, cold=1.002, event_off=3.0, idle=1.0, traced=1.0, ok=True, row_ok=True, fails=0, spread=0.01,
          variants=("fp16", "int4", "nf4"), skew=None, runs=3, run_rot=None, smoke=False, drop_row=None,
          traced_by_variant=None, spread_row=None, idle_row=None, empty_truth=None):
    """skew / idle_row: {(variant, shape): factor} on cupti_rot / cupti_rot_idle only; run_rot: per-run rot factors;
    spread_row: (variant, shape) whose truth spread is 0.05; drop_row: (variant, shape) missing in run 0."""
    metas = []
    for i in range(runs):
        ev = []
        for v in variants:
            step = 4 * sum(KT.values()) / SP[v] * 1.05
            tf = (traced_by_variant or {}).get(v, traced)
            ev.append({"kind": "step", "variant": v, "untraced_us": [step * 0.999, step, step * 1.001],
                       "traced_us": [step * tf] * 3, "count_ok": ok, "warm": [{"warm_n": 100, "warm_s": 1.0}]})
            for s in SHAPES:
                if drop_row == (v, s) and i == 0:
                    continue
                t = KT[s] / SP[v]
                sp = 0.05 if spread_row == (v, s) else spread
                f = (skew or {}).get((v, s), 1.0) * (run_rot[i] if run_rot else rot)
                samples = [] if empty_truth == (v, s) else [t * (1 - sp), t, t * (1 + sp)] * 4
                ev.append({"kind": "row", "variant": v, "shape": s, "names": ["k"], "copies": 4,
                           "truth": {"samples": samples, "round_medians": [t, t * 1.001]},
                           "cupti_rot": {"median": t * f},
                           "cupti_rot_idle": {"median": t * f * idle * (idle_row or {}).get((v, s), 1.0)},
                           "cupti_cold": {"median": t * cold}, "event_cold": t + event_off, "event_graph": t + 0.8,
                           "rot_tracing_effect": 0.001, "warm": [{"warm_n": 100, "warm_s": 1.0}],
                           "count_ok": {"cupti_cold": ok, "cupti_rot": ok and row_ok, "cupti_rot_idle": ok}})
        metas.append({"device": "NVIDIA A100-SXM4-40GB", "l2_bytes": 40 << 20, "layers": 4, "events": ev,
                      "correctness_failures": fails, "notes": {}, "smoke": smoke, "ncopies": {}})
    return metas


class T(unittest.TestCase):
    def q(self, **kw):
        return S.score(build(**kw), log=lambda *_: None)

    def test_predicted(self):
        v = self.q()
        self.assertTrue(v["W-0"])
        self.assertEqual(v["W-1"], "every kernel within 1 %")
        self.assertEqual(v["within 1 %"]["cupti_rot"], "12/12")
        self.assertEqual(v["W-2"], "CUPTI closer than events")
        self.assertEqual(v["W-3"], "warm-up effect < 1 %")
        self.assertAlmostEqual(v["kernel share of step"]["fp16"], round(1 / 1.05, 4), places=3)
        self.assertEqual(v["shortest warm-up s"], 1.0)

    def test_runs_combined_by_median(self):
        # one outlier run: the median of per-run values ignores it, a mean (+1.7 %) would not
        self.assertEqual(self.q(run_rot=[1.0, 1.05, 1.0])["W-1"], "every kernel within 1 %")
        self.assertEqual(self.q(run_rot=[1.0, 1.05, 1.05])["W-1"], "not")

    def test_bands(self):
        self.assertEqual(self.q(rot=1.0099)["W-1"], "every kernel within 1 %")
        self.assertEqual(self.q(rot=1.0101)["W-1"], "every kernel within 2 %")
        self.assertEqual(self.q(rot=0.985)["W-1"], "every kernel within 2 %")
        self.assertEqual(self.q(rot=0.97)["W-1"], "not")
        v = self.q(skew={("nf4", "o"): 1.05})                         # one nf4 row drives the max
        self.assertEqual(v["W-1"], "not")
        self.assertEqual(v["within 1 %"]["cupti_rot"], "11/12")

    def test_w2_w3(self):
        self.assertEqual(self.q(cold=1.5, event_off=0.1)["W-2"], "not")
        self.assertEqual(self.q(idle=0.97)["W-3"], "warm-up changes kernel time by >= 1 %")
        self.assertEqual(self.q(idle=1.03)["W-3"], "warm-up changes kernel time by >= 1 %")
        self.assertEqual(self.q(idle_row={("int4", "o"): 1.5})["W-3"], "warm-up effect < 1 %")   # median, one outlier

    def test_gate(self):
        bad = {"fails": 1, "ok": False, "row_ok": False, "traced": 1.006, "spread": 0.04, "runs": 2, "smoke": True,
               "drop_row": ("fp16", "o"), "traced_by_variant": {"int4": 0.994}, "spread_row": ("nf4", "down"),
               "variants": ("fp16", "nf4"), "empty_truth": ("int4", "qkv")}
        for k, x in bad.items():
            with self.subTest(k=k):
                self.assertFalse(self.q(**{k: x})["W-0"])
        self.assertFalse(self.q(traced=0.994)["W-0"])                  # negative tracing effect counts too
        self.assertTrue(self.q(traced=1.004)["W-0"])
        self.assertTrue(self.q(variants=("fp16", "int4"))["W-0"])      # nf4 optional
        self.assertEqual(self.q(fails=1)["W-1"], "not scored (W-0 failed)")

    def test_all(self):
        vs = [self.q(), self.q(rot=1.015), self.q(fails=1), self.q(idle=0.97)]
        out = S.score_all(vs, log=lambda *_: None)
        self.assertEqual(out["W-0 pass"], 3)
        self.assertEqual(out["W-1 every kernel within 1 %"], "2/3")
        self.assertEqual(out["W-2"], "3/3")
        self.assertEqual(sum(x.endswith("warm-up effect < 1 %") for x in out["W-3"]), 2)
        self.assertEqual(len(out["W-1"]), 3)                           # same GPU model on 3 hosts: no collapse

    def test_check_only_prints_no_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r1.json"
            p.write_text(json.dumps(build(runs=1)[0]))
            r = subprocess.run([sys.executable, str(Path(S.__file__)), "--meta", str(p), "--check-only"],
                               capture_output=True, text=True, check=True)
        self.assertIn("record checks ok 39/39", r.stdout)
        self.assertNotIn("15.0", r.stdout)                              # no kernel time printed


if __name__ == "__main__":
    unittest.main()
