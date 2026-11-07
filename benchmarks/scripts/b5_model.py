#!/usr/bin/env python3
"""B5 (multi-sweep stencil) model: predict, pre-register, calibrate, score -- ESTIMATES only.

Same model as B4 v2 (cache_sim.py, unchanged) applied to the stencil schedules in
cache_sim_b5.py. tau is calibrated per (kernel type, timing mode) at l2x = 4 of run 1.

  # indicative (before measurement), manual parameters:
  python3 benchmarks/scripts/b5_model.py --gpu-label rtx3060 --l2-bytes 2359296 --bw-gbps 333.4 \
      --tau 3.53,2.63 --prereg research/preregistration/B5_rtx3060_indicative.json
  # after measurement, binding protocol:
  python3 benchmarks/scripts/b5_model.py --calib-run <stencil r1> --bw-run <copy_stride run> \
      --prereg research/preregistration/B5_<gpu>_calibrated.json
  python3 benchmarks/scripts/b5_model.py --score <r1> <r2> <r3> --prereg <calibrated json>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import bench_common as bc
import cache_sim_b5 as b5
from harness_env import sha256_of

POLICIES = ("lru", "random")
SIZES = "0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2,4"
SHOW = (("mat4", "isolated"), ("fused2", "isolated"), ("sweep_cold", "isolated"), ("sweep_warm", "isolated"),
        ("mat4", "steady"), ("fused2", "steady"))
PRIMARY = ("mat4/fused2_iso", "warm/cold")
MIN_L2X = 0.625


def measured_medians(run: Path) -> tuple[dict, dict]:
    meta = json.loads((run / "meta.json").read_text())
    out = {}
    for s in bc.load_generic_csv(run / "reps.csv"):
        out[(round(float(s.params["l2x"]), 6), s.variant, s.timing_mode)] = statistics.median(s.times_ms) * 1e3
    return meta, out


def bw_from_copy_stride(path: Path) -> float:
    g = {v.variant: v for v in bc.load_reps_csv(path / "reps.csv")}["memcpy_d2d"]
    return g.logical_bytes / (statistics.median(g.times_ms) * 1e-3) / 1e9


def measured_ratios(med: dict, f: float) -> dict[str, float] | None:
    k = lambda v, m: (round(f, 6), v, m)
    try:
        return {"mat4/fused2_iso": med[k("mat4", "isolated")] / med[k("fused2", "isolated")],
                "mat4/fused2_steady": med[k("mat4", "steady")] / med[k("fused2", "steady")],
                "warm/cold": med[k("sweep_warm", "isolated")] / med[k("sweep_cold", "isolated")]}
    except KeyError:
        return None


def predict(args) -> int:
    l2, calib_points = args.l2_bytes, []
    if args.calib_run:
        meta, med = measured_medians(args.calib_run)
        l2 = meta["l2_cache_bytes"]
        bw = bw_from_copy_stride(args.bw_run)
        sim = b5.simulate_stencil(b5.n_for(args.calib_l2x, l2), l2, "lru")  # policies agree at l2x >= 4
        for v, m in SHOW:
            key = (round(args.calib_l2x, 6), v, m)
            if key in med:
                calib_points.append((v, m, med[key], sim[(v, m)]))
        tau = b5.calibrate_tau(calib_points, bw)
    else:
        bw = args.bw_gbps
        ti, ts = (float(x) for x in args.tau.split(","))
        tau = {(kt, m): (ti if m == "isolated" else ts) for kt in ("5pt", "13pt") for m in ("isolated", "steady")}
    sizes = [float(x) for x in args.sizes_l2x.split(",")]
    print(f"B5 MODEL ESTIMATES (not measurements). L2={l2} B, BW={bw:.1f} GB/s, tau(us)=" +
          ", ".join(f"{kt}/{m[:3]}={v:.2f}" for (kt, m), v in sorted(tau.items())))
    print(f"{'l2x':>7} {'policy':>7} | " + " ".join(f"{v[:6]}/{m[:3]:>3}" for v, m in SHOW) +
          " | m4/f2_iso m4/f2_st warm/cold")
    rows = []
    for f in sizes:
        n = b5.n_for(f, l2)
        for pol in POLICIES:
            sim = b5.simulate_stencil(n, l2, pol)
            t = b5.predict_times_us(sim, tau, bw)
            r = b5.ratios(t)
            rows.append({"l2x": f, "n": n, "policy": pol, "dram_bytes": {f"{v}|{m}": sim[(v, m)] for v, m in SHOW},
                         "pred_us": {f"{v}|{m}": t[(v, m)] for v, m in SHOW}, "pred_ratios": r})
            print(f"{f:7.4f} {pol:>7} | " + " ".join(f"{t[(v, m)]:10.2f}" for v, m in SHOW) +
                  f" | {r['mat4/fused2_iso']:9.3f} {r['mat4/fused2_steady']:8.3f} {r['warm/cold']:8.3f}")
    if args.prereg:
        rec = {
            "kind": "B5 stencil pre-registration (model v2 + cache_sim_b5)",
            "gpu_label": args.gpu_label,
            "indicative": not bool(args.calib_run),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "note": args.note,
            "sha256": {"cache_sim.py": sha256_of(Path(b5.__file__).with_name("cache_sim.py")),
                       "cache_sim_b5.py": sha256_of(Path(b5.__file__)),
                       "b5_model.py": sha256_of(Path(__file__))},
            "l2_bytes": l2, "bw_gbps": bw, "tau_us": {f"{kt}|{m}": v for (kt, m), v in tau.items()},
            "calib_run": str(args.calib_run) if args.calib_run else None, "calib_l2x": args.calib_l2x,
            "in_sample_l2x": [args.calib_l2x],
            "decision_rule": f"PRIMARY: per policy, MAPE of {PRIMARY} over out-of-sample sizes with l2x >= "
                             f"{MIN_L2X} (calibration size excluded), using the median over the measured runs. "
                             "Lower MAPE wins; both > 10 % falsifies the model for this GPU and workload.",
            "rows": rows,
        }
        args.prereg.parent.mkdir(parents=True, exist_ok=True)
        args.prereg.write_text(json.dumps(rec, indent=1))
        print(f"\npre-registration written: {args.prereg}")
    return 0


def score(runs: list[Path], prereg: Path) -> int:
    rec = json.loads(prereg.read_text())
    meds = [measured_medians(r)[1] for r in runs]
    ins = set(rec["in_sample_l2x"])

    def agg(f):
        per = [measured_ratios(m, f) for m in meds]
        if any(p is None for p in per):
            return None
        return {k: statistics.median(p[k] for p in per) for k in per[0]}

    def mape(ratio_at):
        errs = {p: [] for p in POLICIES}
        for row in rec["rows"]:
            f = row["l2x"]
            if f in ins or f < MIN_L2X:
                continue
            meas = ratio_at(f)
            if meas is None:
                continue
            for name in PRIMARY:
                pv = row["pred_ratios"][name]
                errs[row["policy"]].append(abs(meas[name] - pv) / pv * 100)
        return errs

    print(f"{'l2x':>7} {'policy':>7} {'sample':>6} | measured (median of {len(runs)}) / predicted")
    for row in rec["rows"]:
        meas = agg(row["l2x"])
        if meas is None:
            continue
        cells = [f"{n} {meas[n]:.3f}/{row['pred_ratios'][n]:.3f} "
                 f"({abs(meas[n] - row['pred_ratios'][n]) / row['pred_ratios'][n] * 100:.1f}%)" for n in meas]
        print(f"{row['l2x']:7.4f} {row['policy']:>7} {'in' if row['l2x'] in ins else 'OUT':>6} | " + "  ".join(cells))
    print()
    for i, m in enumerate(meds):
        e = mape(lambda f, m=m: measured_ratios(m, f))
        print(f"run {i + 1} ({runs[i].name}): " + ", ".join(
            f"{p} MAPE {statistics.fmean(v):.1f}%" if v else f"{p} n/a" for p, v in e.items()))
    for p, v in mape(agg).items():
        print(f"{p}: PRIMARY out-of-sample MAPE = {statistics.fmean(v):.1f}% over {len(v)} ratios; max {max(v):.1f}%"
              if v else f"{p}: no out-of-sample points")
    return 0


def compare_b4(runs: list[Path], b4_run: Path, tol_pct: float = 10.0) -> int:
    """G4 (model-free cross-family check): B5 sweep warm/cold (median over runs) vs B4 stage2
    warm/cold measured on the same instance, at every common size with l2x >= MIN_L2X."""
    meds = [measured_medians(r)[1] for r in runs]
    b4 = {}
    for s in bc.load_generic_csv(b4_run / "reps.csv"):
        if s.params.get("K") == "1":
            b4[(round(float(s.params["l2x"]), 6), s.variant, s.timing_mode)] = statistics.median(s.times_ms) * 1e3
    worst = 0.0
    print(f"{'l2x':>7} | {'B5 warm/cold':>12} {'B4 warm/cold':>12} {'diff %':>7}")
    for f in sorted({k[0] for k in meds[0]}):
        try:
            w5 = statistics.median(m[(f, "sweep_warm", "isolated")] / m[(f, "sweep_cold", "isolated")] for m in meds)
            w4 = b4[(f, "stage2_warm", "isolated")] / b4[(f, "stage2_cold", "isolated")]
        except KeyError:
            continue
        d = (w5 - w4) / w4 * 100
        if f >= MIN_L2X:
            worst = max(worst, abs(d))
        print(f"{f:7.4f} | {w5:12.3f} {w4:12.3f} {d:7.1f}{'' if f >= MIN_L2X else '  (below rule threshold)'}")
    print(f"\nG4: max |diff| at l2x >= {MIN_L2X} = {worst:.1f}% -> {'PASS' if worst <= tol_pct else 'FAIL'} "
          f"(tolerance {tol_pct:.0f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-run", type=Path)
    ap.add_argument("--calib-l2x", type=float, default=4.0)
    ap.add_argument("--bw-run", type=Path)
    ap.add_argument("--l2-bytes", type=int, default=2_359_296)
    ap.add_argument("--bw-gbps", type=float, default=333.4)
    ap.add_argument("--tau", default="3.53,2.63", help="manual tau_iso,tau_steady (us) for both kernel types")
    ap.add_argument("--sizes-l2x", default=SIZES)
    ap.add_argument("--gpu-label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--prereg", type=Path)
    ap.add_argument("--score", type=Path, nargs="+")
    ap.add_argument("--compare-b4", type=Path, help="B4 fusion run from the same instance (G4); use with --score runs")
    args = ap.parse_args()
    if args.score and args.compare_b4:
        return compare_b4(list(args.score), args.compare_b4)
    if args.score:
        return score(list(args.score), args.prereg)
    if args.calib_run and not args.bw_run:
        ap.error("--calib-run needs --bw-run (memcpy BW from the same instance)")
    return predict(args)


if __name__ == "__main__":
    sys.exit(main())
