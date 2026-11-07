#!/usr/bin/env python3
"""B4 model v2: simulate, calibrate, and print / pre-register fusion predictions (ESTIMATES).

  # calibrate tau on the l2x=4 points of an existing run, predict a list of sizes, and write a
  # pre-registration record (JSON) BEFORE those sizes are measured:
  python3 benchmarks/scripts/fusion_model_v2.py \
      --calib-run benchmarks/results/fusion/20261104-060348-smoke-remote --calib-l2x 4 \
      --sizes-l2x 0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2,4,16 \
      --prereg research/preregistration/B4_v2_predictions.json

  # compare a measured run against a pre-registration record:
  python3 benchmarks/scripts/fusion_model_v2.py --score benchmarks/results/fusion/<run> \
      --prereg research/preregistration/B4_v2_predictions.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import bench_common as bc
import cache_sim as cs
from harness_env import ROOT, sha256_of

POLICIES = ("lru", "random")
SHOW = (("fused", "isolated"), ("materialized", "isolated"), ("stage2_cold", "isolated"),
        ("stage2_warm", "isolated"), ("fused", "steady"), ("materialized", "steady"))


def n_for(l2x: float, l2: int) -> int:  # identical rounding to fusion.cu
    return max(256, int(l2x * l2 / 4) // 256 * 256)


def measured_medians(run: Path, k: int = 1) -> tuple[dict, dict]:
    meta = json.loads((run / "meta.json").read_text())
    out: dict[tuple[float, str, str], float] = {}
    for s in bc.load_generic_csv(run / "reps.csv"):
        if int(s.params["K"]) != k:
            continue
        out[(round(float(s.params["l2x"]), 6), s.variant, s.timing_mode)] = statistics.median(s.times_ms) * 1e3
    return meta, out


def bw_from_copy_stride(path: Path | None) -> float:
    if path is None:
        return 333.5  # MEASURED memcpy D2D on the RTX 3060 (BASELINE_REPORT §10/§12)
    g = {v.variant: v for v in bc.load_reps_csv(path / "reps.csv")}["memcpy_d2d"]
    return g.logical_bytes / (statistics.median(g.times_ms) * 1e-3) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-run", type=Path)
    ap.add_argument("--calib-l2x", type=float, default=4.0)
    ap.add_argument("--bw-run", type=Path, help="copy_stride run dir for memcpy BW (default 333.5 GB/s)")
    ap.add_argument("--l2-bytes", type=int, default=2_359_296)
    ap.add_argument("--sizes-l2x", default="0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2,4,16")
    ap.add_argument("--prereg", type=Path, help="write (with --calib-run) or read (with --score)")
    ap.add_argument("--score", type=Path, nargs="+",
                    help="measured fusion run(s) to score against --prereg; with several runs each ratio "
                         "is the median over runs (per-run MAPE is reported too)")
    ap.add_argument("--tau-iso", type=float, help="manual tau (us) when there is no --calib-run (indicative)")
    ap.add_argument("--tau-steady", type=float)
    ap.add_argument("--bw-gbps", type=float, help="manual BW instead of --bw-run (indicative)")
    ap.add_argument("--in-sample-l2x", default="0.25,0.5,1,4,16",
                    help="sizes excluded from scoring (seen while designing, or used for calibration)")
    ap.add_argument("--label", default="", help="free text stored in the pre-registration record")
    ap.add_argument("--note", default="", help="overrides the default (RTX 3060) note in the record")
    args = ap.parse_args()

    if args.score:
        return score(list(args.score), args.prereg)

    bw = args.bw_gbps if args.bw_gbps else bw_from_copy_stride(args.bw_run)
    l2 = args.l2_bytes
    tau = {"isolated": float("nan"), "steady": float("nan")}
    if args.tau_iso is not None and args.tau_steady is not None:
        tau = {"isolated": args.tau_iso, "steady": args.tau_steady}
    calib_points = []
    if args.calib_run:
        meta, med = measured_medians(args.calib_run)
        l2 = meta["l2_cache_bytes"]
        sim = cs.simulate_fusion(n_for(args.calib_l2x, l2), l2, "lru")  # both policies agree at l2x >= 4
        for (v, m) in SHOW:
            key = (round(args.calib_l2x, 6), v, m)
            if key in med:
                calib_points.append((v, m, med[key], sim[(v, m)]))
        tau = cs.calibrate_tau(calib_points, bw)
    sizes = [float(x) for x in args.sizes_l2x.split(",")]
    rows = []
    print(f"MODEL v2 ESTIMATES (not measurements). L2={l2} B, BW={bw:.1f} GB/s, "
          f"tau_iso={tau['isolated']:.2f} us, tau_steady={tau['steady']:.2f} us (calibrated at l2x={args.calib_l2x})")
    print(f"{'l2x':>6} {'policy':>7} | " + " ".join(f"{v[:5]}/{m[:3]:>3}" for v, m in SHOW) +
          " | mat/fus_iso mat/fus_st warm/cold")
    for f in sizes:
        n = n_for(f, l2)
        for pol in POLICIES:
            sim = cs.simulate_fusion(n, l2, pol)
            t = cs.predict_times_us(sim, tau["isolated"], tau["steady"], bw)
            r = {"mat/fused_iso": t[("materialized", "isolated")] / t[("fused", "isolated")],
                 "mat/fused_steady": t[("materialized", "steady")] / t[("fused", "steady")],
                 "warm/cold": t[("stage2_warm", "isolated")] / t[("stage2_cold", "isolated")]}
            rows.append({"l2x": f, "n": n, "policy": pol,
                         "dram_bytes": {f"{v}|{m}": sim[(v, m)] for v, m in SHOW},
                         "pred_us": {f"{v}|{m}": t[(v, m)] for v, m in SHOW}, "pred_ratios": r})
            print(f"{f:6.3f} {pol:>7} | " + " ".join(f"{t[(v, m)]:9.2f}" for v, m in SHOW) +
                  f" | {r['mat/fused_iso']:11.3f} {r['mat/fused_steady']:10.3f} {r['warm/cold']:9.3f}")
    in_sample = [float(x) for x in args.in_sample_l2x.split(",") if x]
    if args.prereg and (args.calib_run or (args.tau_iso is not None and args.bw_gbps)):
        rec = {
            "kind": "B4 fusion model v2 pre-registration",
            "label": args.label,
            "indicative": not bool(args.calib_run),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "note": args.note or ("ESTIMATES written before the listed sizes are measured. Sizes 0.25, 0.5, 1, 4, 16 were "
                    "measured in the 2026-11-04 smoke run and influenced the model design (in-sample); all "
                    "others are out-of-sample. tau was calibrated only on the l2x=4 points."),
            "cache_sim_sha256": sha256_of(Path(cs.__file__)),
            "l2_bytes": l2, "bw_gbps": bw, "tau_us": tau, "calib_l2x": args.calib_l2x,
            "calib_run": str(args.calib_run) if args.calib_run else None, "in_sample_l2x": in_sample,
            "decision_rule": "PRIMARY: per policy, mean absolute percentage error (MAPE) of the two isolated-"
                             "mode ratios (mat/fused_iso, warm/cold) over out-of-sample sizes with l2x >= 0.625 "
                             "(kernel times >= ~8 us). The policy with lower MAPE is preferred; if both exceed "
                             "10 %, model v2 is falsified for this GPU. SECONDARY (reported, not used in the "
                             "decision): mat/fused_steady, because the time model has no L2-bandwidth term and "
                             "under-predicts windows whose DRAM traffic is small (see known_limitations).",
            "known_limitations": ["no L2-bandwidth term: predictions for windows with little DRAM traffic "
                                  "(small sizes, steady mode) are lower bounds",
                                  "fully-associative cache, strict index-order streaming, no set conflicts "
                                  "or L2 slicing", "tau calibrated from 4 isolated + 2 steady points at l2x=4"],
            "rows": rows,
        }
        args.prereg.parent.mkdir(parents=True, exist_ok=True)
        args.prereg.write_text(json.dumps(rec, indent=1))
        print(f"\npre-registration written: {args.prereg}")
    return 0


def _ratios(med: dict, f: float) -> dict[str, float] | None:
    k = lambda v, m: (round(f, 6), v, m)
    try:
        return {"mat/fused_iso": med[k("materialized", "isolated")] / med[k("fused", "isolated")],
                "mat/fused_steady": med[k("materialized", "steady")] / med[k("fused", "steady")],
                "warm/cold": med[k("stage2_warm", "isolated")] / med[k("stage2_cold", "isolated")]}
    except KeyError:
        return None


PRIMARY = ("mat/fused_iso", "warm/cold")


def _mape(rec: dict, ratio_at) -> dict[str, list[float]]:
    ins = set(rec["in_sample_l2x"])
    errs: dict[str, list[float]] = {p: [] for p in POLICIES}
    for row in rec["rows"]:
        f = row["l2x"]
        if f in ins or f < 0.625:
            continue
        meas = ratio_at(f)
        if meas is None:
            continue
        for name in PRIMARY:
            pv = row["pred_ratios"][name]
            errs[row["policy"]].append(abs(meas[name] - pv) / pv * 100)
    return errs


def score(runs: list[Path], prereg: Path) -> int:
    """Primary rule: MAPE of PRIMARY ratios at out-of-sample sizes with l2x >= 0.625."""
    rec = json.loads(prereg.read_text())
    meds = [measured_medians(r)[1] for r in runs]
    ins = set(rec["in_sample_l2x"])

    def agg(f):
        per = [_ratios(m, f) for m in meds]
        if any(p is None for p in per):
            return None
        return {k: statistics.median(p[k] for p in per) for k in per[0]}

    print(f"{'l2x':>7} {'policy':>7} {'sample':>6} | measured (median of {len(runs)} run(s)) / predicted")
    for row in rec["rows"]:
        f = row["l2x"]
        meas = agg(f)
        if meas is None:
            continue
        cells = [f"{n} {meas[n]:.3f}/{row['pred_ratios'][n]:.3f} "
                 f"({abs(meas[n] - row['pred_ratios'][n]) / row['pred_ratios'][n] * 100:.1f}%)" for n in meas]
        print(f"{f:7.4f} {row['policy']:>7} {'in' if f in ins else 'OUT':>6} | " + "  ".join(cells))
    print()
    for i, m in enumerate(meds):
        e = _mape(rec, lambda f, m=m: _ratios(m, f))
        print(f"run {i + 1} ({runs[i].name}): " + ", ".join(
            f"{p} MAPE {statistics.fmean(v):.1f}%" if v else f"{p} n/a" for p, v in e.items()))
    e = _mape(rec, agg)
    for p, v in e.items():
        print(f"{p}: PRIMARY out-of-sample MAPE = {statistics.fmean(v):.1f}% over {len(v)} ratios; "
              f"max {max(v):.1f}%" if v else f"{p}: no out-of-sample points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
