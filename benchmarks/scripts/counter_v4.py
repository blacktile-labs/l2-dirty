#!/usr/bin/env python3
"""Model v4 counter test: per-launch DRAM bytes predicted by v3's locked LRU simulator
(cache_sim.Cache "lru") and by model v4 (cache_sim_v4.DirtyCapLRU = the same LRU + a dirty cap
of half the L2), compared with Nsight Compute counters on B4 (fusion) and B5 (stencil) at sizes
whose counters have never been measured. ESTIMATES only. Rules: research/HYPOTHESES.md,
"Model v4 counter pre-registration (K-0..K-3)".

Launch plans, sizes, flush accounting and CSV alignment are reused unchanged from counter_prereg.py
(the D-026 protocol): binaries run with --reps 1 --warmup 0 --steady-iters 3 (fusion: --ks 1).
A cell is one launch's read bytes (flush launches excluded) or write bytes (the first flush of
each size excluded: it follows the unmodelled correctness phase). e = |measured - predicted| / B,
B = bytes of one array. The measured value of a cell is the median over runs.

  python3 benchmarks/scripts/counter_v4.py predict --l2-bytes 3145728 --sizes-l2x <...> --prereg <json>
  python3 benchmarks/scripts/counter_v4.py score --bench fusion --ncu-csv r1.csv r2.csv r3.csv --prereg <json>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import counter_prereg as cp
from cache_sim import LINE, Cache
from cache_sim_v4 import DirtyCapLRU
from harness_env import sha256_of

MODELS = ("v3", "v4")
DISC = 0.25            # a cell discriminates when |v4 - v3| >= DISC arrays
K1_WIN, K1_LOSE = 0.75, 0.25   # share of discriminating cells where v4 is strictly closer
K2_MEAN_E = 0.20       # v4 mean e over discriminating cells
K3_SLACK = 0.01        # v4 mean e over all cells <= v3 mean e + slack
FAR_L2X = (2.0, 3.0)
FAR_E, FAR_FRAC = 0.10, 0.90
S0_TOL = cp.S0_TOL


def make_cache(model: str, l2: int):
    cap = max(1, l2 // LINE)
    return Cache(cap, "lru") if model == "v3" else DirtyCapLRU(cap)


def simulate(bench: str, n: int, l2: int, model: str) -> list[dict]:
    """== counter_prereg.simulate_launches with the cache chosen by `model`."""
    n_lines = max(1, (4 * n + LINE - 1) // LINE)
    f_lines = (cp.flush_bytes(l2) + LINE - 1) // LINE
    names = ("X", "T", "Y") if bench == "fusion" else ("X", "P", "Q")
    base = {b: i * n_lines for i, b in enumerate(names)}
    fbase = len(names) * n_lines
    c = make_cache(model, l2)
    out = []
    for i, (label, kern, src, dst) in enumerate(cp.launch_plan(bench)):
        c.reset_counters()
        if src is None:
            for j in range(f_lines):
                c.access(fbase + j, False)
        else:
            s, d = base[src], base[dst]
            for j in range(n_lines):
                c.access(s + j, False)
                c.access(d + j, True)
        out.append({"i": i, "label": label, "kernel": kern, "read": c.reads * LINE,
                    "write": c.writebacks * LINE, "scored_write": not (i == 0 and src is None)})
    return out


def cells_of(launches_v3: list[dict], launches_v4: list[dict], B: int) -> list[dict]:
    out = []
    for a, b in zip(launches_v3, launches_v4):
        if a["kernel"] != cp.FLUSH:
            out.append({"i": a["i"], "label": a["label"], "dir": "R", "v3": a["read"] / B, "v4": b["read"] / B})
        if a["scored_write"]:
            out.append({"i": a["i"], "label": a["label"], "dir": "W", "v3": a["write"] / B, "v4": b["write"] / B})
    for c in out:
        c["disc"] = abs(c["v4"] - c["v3"]) >= DISC
    return out


def cmd_predict(args) -> int:
    l2 = args.l2_bytes
    sizes = [float(x) for x in args.sizes_l2x.split(",")]
    rec = {"kind": "model v4 counter pre-registration (per-launch DRAM bytes, v3 vs v4)",
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "sha256": {f: sha256_of(Path(__file__).with_name(f)) for f in
                      ("cache_sim.py", "cache_sim_v4.py", "cache_sim_b5.py", "counter_prereg.py", "counter_v4.py")},
           "l2_bytes": l2, "flush_bytes": cp.flush_bytes(l2), "steady_iters": cp.STEADY_ITERS,
           "rules": {"DISC": DISC, "K1_WIN": K1_WIN, "K1_LOSE": K1_LOSE, "K2_MEAN_E": K2_MEAN_E,
                     "K3_SLACK": K3_SLACK, "FAR_L2X": FAR_L2X, "FAR_E": FAR_E, "FAR_FRAC": FAR_FRAC,
                     "S0_TOL": S0_TOL},
           "benches": {}}
    for bench in ("fusion", "stencil"):
        rows = []
        for f in sizes:
            n = cp.n_for(bench, f, l2)
            B = 4 * n
            cells = cells_of(simulate(bench, n, l2, "v3"), simulate(bench, n, l2, "v4"), B)
            rows.append({"l2x": f, "n": n, "array_bytes": B, "cells": cells})
            nd = sum(c["disc"] for c in cells)
            print(f"{bench:8} l2x {f:<7g} cells {len(cells):3d}  discriminating {nd:3d}")
        rec["benches"][bench] = {"sizes_l2x": sizes, "plan": [p[:2] for p in cp.launch_plan(bench)], "rows": rows}
    if args.prereg:
        args.prereg.parent.mkdir(parents=True, exist_ok=True)
        args.prereg.write_text(json.dumps(rec, indent=1))
        print(f"pre-registration written: {args.prereg}")
    return 0


def score_bench(bench_rec: dict, runs: list[list[list[dict]]], flush_bytes: int, log=print) -> dict:
    """runs[k][size_index] = aligned measured launches of run k."""
    v: dict = {}
    fl = [m["read"] for run in runs for seg in run for m in seg if m["kernel"] == cp.FLUSH]
    s0 = max(abs(x - flush_bytes) / flush_bytes for x in fl)
    v["S-0"] = s0 <= S0_TOL
    log(f"S-0 flush read bytes: max rel err {s0 * 100:.1f}% -> {'PASS' if v['S-0'] else 'FAIL (stop)'}")
    allc = []
    for si, row in enumerate(bench_rec["rows"]):
        B = row["array_bytes"]
        for c in row["cells"]:
            key = "read" if c["dir"] == "R" else "write"
            meas = statistics.median(run[si][c["i"]][key] for run in runs) / B
            allc.append({**c, "l2x": row["l2x"], "meas": meas,
                         "e3": abs(meas - c["v3"]), "e4": abs(meas - c["v4"])})
    disc = [c for c in allc if c["disc"]]
    if disc:
        win = sum(c["e4"] < c["e3"] for c in disc) / len(disc)
        k1 = "v4 better" if win >= K1_WIN else "v3 better" if win <= K1_LOSE else "mixed"
        m4, m3 = statistics.fmean(c["e4"] for c in disc), statistics.fmean(c["e3"] for c in disc)
    else:
        win, k1, m4, m3 = float("nan"), "n/a", float("nan"), float("nan")
    v["K-1"], v["K-1 share"] = k1, round(win, 3)
    v["K-2"] = bool(disc) and m4 <= K2_MEAN_E
    log(f"K-1 discriminating cells: {len(disc)}; v4 strictly closer in {win * 100:.1f}% -> {k1}")
    log(f"K-2 mean e over discriminating cells: v4 {m4:.3f}, v3 {m3:.3f} (v4 must be <= {K2_MEAN_E}) "
        f"-> {'PASS' if v['K-2'] else 'FAIL'}")
    a4, a3 = statistics.fmean(c["e4"] for c in allc), statistics.fmean(c["e3"] for c in allc)
    far = [c for c in allc if c["l2x"] in FAR_L2X]
    ffrac = sum(c["e4"] <= FAR_E for c in far) / len(far) if far else float("nan")
    v["K-3"] = a4 <= a3 + K3_SLACK and ffrac >= FAR_FRAC
    log(f"K-3 all {len(allc)} cells: mean e v4 {a4:.3f} vs v3 {a3:.3f}; far cells within {FAR_E} B: "
        f"{ffrac * 100:.1f}% of {len(far)} -> {'PASS' if v['K-3'] else 'FAIL'}")
    v["_cells"] = allc
    return v


def cmd_score(args) -> int:
    rec = json.loads(args.prereg.read_text())
    br = rec["benches"][args.bench]
    runs = [cp.align(cp.load_ncu_raw(p), args.bench, len(br["sizes_l2x"])) for p in args.ncu_csv]
    v = score_bench(br, runs, rec["flush_bytes"])
    cells = v.pop("_cells")
    if args.verbose:
        print(f"\n{'l2x':>7} {'label':32} d   meas    v3    v4   {'disc':4}")
        for c in cells:
            print(f"{c['l2x']:7g} {c['label']:32} {c['dir']} {c['meas']:6.3f} {c['v3']:5.2f} {c['v4']:5.2f}"
                  f"   {'*' if c['disc'] else ''}")
    print("verdict:", json.dumps(v))
    if args.json_out:
        args.json_out.write_text(json.dumps({"verdict": v, "cells": cells}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--l2-bytes", type=int, required=True)
    p.add_argument("--sizes-l2x", required=True)
    p.add_argument("--prereg", type=Path)
    s = sub.add_parser("score")
    s.add_argument("--bench", choices=("fusion", "stencil"), required=True)
    s.add_argument("--ncu-csv", type=Path, nargs="+", required=True)
    s.add_argument("--prereg", type=Path, required=True)
    s.add_argument("--json-out", type=Path)
    s.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return cmd_predict(args) if args.cmd == "predict" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
