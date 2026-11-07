#!/usr/bin/env python3
"""Model v3: same L2 simulator, time model with a per-kernel-type bandwidth -- ESTIMATES only.

    t(window) = sum over kernel types kt of  k_kt * tau[kt, mode]  +  DRAM_bytes * beta[kt, mode]

(every timed window in B4/B5 uses a single kernel type, so this is k*tau + bytes*beta, beta = 1/BW).
v2 used one tau per mode and the memcpy BW for every kernel; that could not represent a kernel that
is slower *per byte* (the 13-point stencil on RTX 3060, BASELINE_REPORT §17). v3 fits (tau, beta)
per (kernel type, mode) by least squares on two DRAM-bound calibration sizes (l2x = 2 and 4).

Byte counts come from the *unchanged* simulators (cache_sim.py for B4, cache_sim_b5.py for B5).

  python3 benchmarks/scripts/model_v3.py predict --bench stencil --calib-run <r1> --sizes-l2x ... \
      [--assume KT=FROM:SCALE ...] --prereg <json>
  python3 benchmarks/scripts/model_v3.py score --bench stencil --runs <r1> <r2> <r3> --prereg <json>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import bench_common as bc
import cache_sim
import cache_sim_b5
from harness_env import sha256_of

POLICIES = ("lru", "random")
CALIB_L2X = (2.0, 4.0)
MIN_L2X = 0.5625

# variant -> (kernel type, kernels in the timed window)
KMAP = {
    "fusion": {"fused": ("pw_fused", 1), "materialized": ("pw_stage", 2),
               "stage2_cold": ("pw_stage", 1), "stage2_warm": ("pw_stage", 1)},
    "stencil": {"mat4": ("5pt", 4), "fused2": ("13pt", 2), "fused2c": ("13pt_c", 2),
                "sweep_cold": ("5pt", 1), "sweep_warm": ("5pt", 1)},
}
PRIMARY = {
    "fusion": (("materialized", "fused", "isolated"), ("stage2_warm", "stage2_cold", "isolated")),
    "stencil": (("mat4", "fused2", "isolated"), ("mat4", "fused2c", "isolated"),
                ("sweep_warm", "sweep_cold", "isolated")),
}
SECONDARY = {
    "fusion": (("materialized", "fused", "steady"),),
    "stencil": (("mat4", "fused2", "steady"), ("mat4", "fused2c", "steady")),
}


def ratio_name(r) -> str:
    return f"{r[0]}/{r[1]}_{r[2][:3]}"


def n_for(bench: str, l2x: float, l2: int) -> int:
    if bench == "fusion":
        return max(256, int(l2x * l2 / 4) // 256 * 256)  # == fusion.cu
    return cache_sim_b5.n_for(l2x, l2)                  # == stencil.cu


def simulate(bench: str, n: int, l2: int, policy: str) -> dict[tuple[str, str], int]:
    if bench == "fusion":
        return cache_sim.simulate_fusion(n, l2, policy)
    sim = cache_sim_b5.simulate_stencil(n, l2, policy)
    for mode in ("isolated", "steady"):  # fused2c runs the same schedule as fused2
        sim[("fused2c", mode)] = sim[("fused2", mode)]
    return sim


def measured_medians(run: Path) -> tuple[dict, dict]:
    meta = json.loads((run / "meta.json").read_text())
    out = {}
    for s in bc.load_generic_csv(run / "reps.csv"):
        if "K" in s.params and s.params["K"] != "1":
            continue
        out[(round(float(s.params["l2x"]), 6), s.variant, s.timing_mode)] = statistics.median(s.times_ms) * 1e3
    return meta, out


def fit_tau_beta(points: list[tuple[int, int, float]]) -> tuple[float, float]:
    """Least squares for t = k*tau + b*beta. points: (k, bytes, measured_us). Needs two
    linearly independent (k, bytes) rows."""
    skk = sum(k * k for k, b, t in points)
    skb = sum(k * b for k, b, t in points)
    sbb = sum(b * b for k, b, t in points)
    skt = sum(k * t for k, b, t in points)
    sbt = sum(b * t for k, b, t in points)
    det = skk * sbb - skb * skb
    if abs(det) <= 1e-12 * max(1.0, skk * sbb):
        raise ValueError("calibration points do not identify (tau, beta)")
    tau = (skt * sbb - sbt * skb) / det
    beta = (sbt * skk - skt * skb) / det
    if tau < 0:  # a per-kernel overhead cannot be negative: refit with tau = 0
        tau, beta = 0.0, sbt / sbb
    return tau, beta


def calibrate(bench: str, med: dict, l2: int) -> dict[tuple[str, str], tuple[float, float]]:
    pts: dict[tuple[str, str], list] = {}
    for f in CALIB_L2X:
        sim = simulate(bench, n_for(bench, f, l2), l2, "lru")  # policies agree at l2x >= 2 (<= 1 %)
        for (v, mode), b in sim.items():
            key = (round(f, 6), v, mode)
            if key in med:
                kt, k = KMAP[bench][v]
                pts.setdefault((kt, mode), []).append((k, b, med[key]))
    return {key: fit_tau_beta(p) for key, p in pts.items()}


def predict_times(bench: str, sim: dict, params: dict) -> dict[tuple[str, str], float]:
    out = {}
    for (v, mode), b in sim.items():
        kt, k = KMAP[bench][v]
        if (kt, mode) in params:
            tau, beta = params[(kt, mode)]
            out[(v, mode)] = k * tau + b * beta
    return out


def ratios(bench: str, t: dict) -> dict[str, float]:
    out = {}
    for r in PRIMARY[bench] + SECONDARY[bench]:
        a, b = (r[0], r[2]), (r[1], r[2])
        if a in t and b in t and t[b] > 0:  # t[b] == 0 only when tau == 0 and no DRAM bytes (L2-resident)
            out[ratio_name(r)] = t[a] / t[b]
    return out


def cmd_predict(args) -> int:
    meta, med = measured_medians(args.calib_run)
    l2 = meta["l2_cache_bytes"]
    params = calibrate(args.bench, med, l2)
    for spec in args.assume or []:  # e.g. 13pt_c=13pt:0.7037 -> beta scaled, floored at 5pt beta
        kt, rhs = spec.split("=")
        src, scale = rhs.split(":")
        for mode in ("isolated", "steady"):
            tau, beta = params[(src, mode)]
            floor = params.get(("5pt", mode), (0, 0))[1]
            params[(kt, mode)] = (tau, max(floor, beta * float(scale)))
    sizes = [float(x) for x in args.sizes_l2x.split(",")]
    print(f"MODEL v3 ESTIMATES ({args.bench}). L2={l2} B. (tau us, BW GB/s) per (kernel type, mode):")
    for (kt, mode), (tau, beta) in sorted(params.items()):
        print(f"   {kt:9} {mode:8} tau={tau:8.2f}  BW={1 / beta / 1e3 if beta > 0 else float('nan'):7.1f}")
    rows = []
    for f in sizes:
        n = n_for(args.bench, f, l2)
        for pol in POLICIES:
            sim = simulate(args.bench, n, l2, pol)
            t = predict_times(args.bench, sim, params)
            r = ratios(args.bench, t)
            rows.append({"l2x": f, "n": n, "policy": pol, "pred_ratios": r,
                         "pred_us": {f"{v}|{m}": x for (v, m), x in t.items()},
                         "dram_bytes": {f"{v}|{m}": x for (v, m), x in sim.items()}})
            print(f"{f:7.4f} {pol:>7} | " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
    if args.prereg:
        rec = {"kind": f"model v3 pre-registration ({args.bench})", "gpu_label": args.gpu_label,
               "indicative": args.indicative, "note": args.note,
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "sha256": {f: sha256_of(Path(__file__).with_name(f)) for f in
                          ("cache_sim.py", "cache_sim_b5.py", "model_v3.py")},
               "l2_bytes": l2, "calib_run": str(args.calib_run), "calib_l2x": list(CALIB_L2X),
               "assumptions": args.assume or [],
               "params": {f"{kt}|{m}": {"tau_us": p[0], "bw_gbps": 1 / p[1] / 1e3 if p[1] > 0 else None}
                          for (kt, m), p in params.items()},
               "primary": [ratio_name(r) for r in PRIMARY[args.bench]],
               "decision_rule": f"PRIMARY ratios {[ratio_name(r) for r in PRIMARY[args.bench]]}: per policy MAPE "
                                f"over sizes l2x >= {MIN_L2X} excluding calibration sizes {list(CALIB_L2X)}, "
                                "median over runs. Lower MAPE wins; both > 10 % falsifies v3 for this GPU/bench.",
               "rows": rows}
        args.prereg.parent.mkdir(parents=True, exist_ok=True)
        args.prereg.write_text(json.dumps(rec, indent=1))
        print(f"\npre-registration written: {args.prereg}")
    return 0


def measured_ratio_set(bench: str, med: dict, f: float) -> dict[str, float]:
    out = {}
    for r in PRIMARY[bench] + SECONDARY[bench]:
        a, b = (round(f, 6), r[0], r[2]), (round(f, 6), r[1], r[2])
        if a in med and b in med:
            out[ratio_name(r)] = med[a] / med[b]
    return out


def cmd_score(args) -> int:
    rec = json.loads(args.prereg.read_text())
    meds = [measured_medians(r)[1] for r in args.runs]
    prim = rec["primary"]
    excluded = set(rec["calib_l2x"])
    errs = {p: [] for p in POLICIES}
    for row in rec["rows"]:
        f = row["l2x"]
        per = [measured_ratio_set(args.bench, m, f) for m in meds]
        keys = set.intersection(*(set(p) for p in per)) if per else set()
        if not keys:
            continue
        agg = {k: statistics.median(p[k] for p in per) for k in keys}
        scored = f >= MIN_L2X and f not in excluded
        cells = []
        for k in sorted(keys):
            pv = row["pred_ratios"].get(k)
            if pv is None or pv <= 0:  # degenerate (tau == 0 and no DRAM bytes): not scorable
                continue
            e = abs(agg[k] - pv) / pv * 100
            if scored and k in prim:
                errs[row["policy"]].append(e)
            cells.append(f"{k} {agg[k]:.3f}/{pv:.3f} ({e:.1f}%)")
        tag = "cal" if f in excluded else ("OUT" if scored else "low")
        print(f"{f:7.4f} {row['policy']:>7} {tag:>4} | " + "  ".join(cells))
    print()
    for p, e in errs.items():
        print(f"{p}: PRIMARY MAPE = {statistics.fmean(e):.1f}% over {len(e)} ratios; max {max(e):.1f}%"
              if e else f"{p}: no scored points")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--bench", choices=("fusion", "stencil"), required=True)
    p.add_argument("--calib-run", type=Path, required=True)
    p.add_argument("--sizes-l2x", required=True)
    p.add_argument("--assume", action="append", help="KT=FROM:SCALE (indicative only)")
    p.add_argument("--prereg", type=Path)
    p.add_argument("--gpu-label", default="")
    p.add_argument("--note", default="")
    p.add_argument("--indicative", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--bench", choices=("fusion", "stencil"), required=True)
    s.add_argument("--runs", type=Path, nargs="+", required=True)
    s.add_argument("--prereg", type=Path, required=True)
    args = ap.parse_args()
    return cmd_predict(args) if args.cmd == "predict" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
