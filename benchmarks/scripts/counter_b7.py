#!/usr/bin/env python3
"""B7 scorer: per-launch DRAM bytes of real PyTorch kernels (eager elementwise kernels and the
TorchInductor/Triton fused kernel, torch_opchain.py) vs. models v3 (cache_sim.Cache "lru") and v4
(cache_sim_v4.DirtyCapLRU). Rules R-0..R-3: research/HYPOTHESES.md "B7 pre-registration".

Neither model has a free parameter here (L2 size from the device, dirty cap 1/2 fixed by B6/B6b).
Each launch of the plan (real tensor addresses recorded by the harness) is replayed through one
cache in issue order, flush kernels included:
  small ranges (the [4096] bias) are read first, then the full-size streams line by line in index
  order: for line j, read every input's line j, then write every output's line j (S4 of cache_sim).
A cell is one launch's read bytes (flush launches excluded) or write bytes (the first flush of the
measured phase excluded: it follows the unmodelled correctness phase); e = |meas - pred| / B,
B = bytes of one [M, 4096] tensor. Measured value = median over runs.

  python3 counter_b7.py predict --plan <plan.json from a run without ncu> --out <prereg.json>
  python3 counter_b7.py score --prereg <prereg.json> --plan <plan of run 1> --ncu-csv r1.csv r2.csv r3.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

from cache_sim import LINE, Cache
from cache_sim_v4 import DirtyCapLRU
from counter_prereg import load_ncu_raw
from harness_env import sha256_of

MODELS = ("v3", "v4")
S0_TOL = 0.05
DISC = 0.25
R1_E, R1_FRAC = 0.10, 0.80
R2_WIN, R2_LOSE = 0.75, 0.25
R3_L2X, R3_E, R3_FRAC = 2.0, 0.10, 0.90
T1_L2X, T1_LO, T1_HI = (2.0, 3.0), 1.9, 2.8   # graph_steady eager/compiled speedup where DRAM-bound


def kernel_class(name: str) -> str:
    if name.startswith("flush_l2_read"):
        return "flush"
    if name.startswith("triton_"):
        return "inductor_fused"
    if "elementwise_kernel" in name:
        return "eager"
    return "other"


def plan_class(kernel: str) -> str:
    return "flush" if kernel == "flush" else "inductor_fused" if kernel == "inductor_fused" else "eager"


def simulate(launches: list[dict], l2: int, model: str) -> list[dict]:
    cap = max(1, l2 // LINE)
    c = Cache(cap, "lru") if model == "v3" else DirtyCapLRU(cap)
    out = []
    for L in launches:
        c.reset_counters()
        big = max((n for _, n in L["reads"] + L["writes"]), default=0)
        small = [(p, n) for p, n in L["reads"] if n < big]
        rd = [(p // LINE, (n + LINE - 1) // LINE) for p, n in L["reads"] if n == big]
        wr = [(p // LINE, (n + LINE - 1) // LINE) for p, n in L["writes"] if n == big]
        for p, n in small:
            for j in range(p // LINE, (p + n + LINE - 1) // LINE):
                c.access(j, False)
        n_lines = max([n for _, n in rd + wr], default=0)
        for j in range(n_lines):
            for b, n in rd:
                if j < n:
                    c.access(b + j, False)
            for b, n in wr:
                if j < n:
                    c.access(b + j, True)
        out.append({"read": c.reads * LINE, "write": c.writebacks * LINE})
    return out


def predictions(plan: dict) -> list[dict]:
    l2 = plan["meta"]["l2_bytes"]
    L = plan["launches"]
    sims = {m: simulate(L, l2, m) for m in MODELS}
    out = []
    for i, x in enumerate(L):
        B = x["M"] * plan["meta"]["H"] * 4
        out.append({"i": i, "l2x": x["l2x"], "variant": x["variant"], "mode": x["mode"], "kernel": x["kernel"],
                    "B": B, **{f"{m}_{d}": sims[m][i][d] for m in MODELS for d in ("read", "write")}})
    return out


def cmd_predict(a) -> int:
    plan = json.loads(a.plan.read_text())
    pred = predictions(plan)
    rec = {"kind": "B7 per-launch DRAM-byte predictions (v3 vs v4), real PyTorch kernels",
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "sha256": {f: sha256_of(Path(__file__).with_name(f)) for f in
                      ("cache_sim.py", "cache_sim_v4.py", "counter_b7.py", "torch_opchain.py")},
           "plan_meta": plan["meta"],
           "rules": {"S0_TOL": S0_TOL, "DISC": DISC, "R1_E": R1_E, "R1_FRAC": R1_FRAC, "R2_WIN": R2_WIN,
                     "R2_LOSE": R2_LOSE, "R3_L2X": R3_L2X, "R3_E": R3_E, "R3_FRAC": R3_FRAC},
           "launches": pred}
    a.out.write_text(json.dumps(rec, indent=1))
    by = {}
    for p in pred:
        for d in ("read", "write"):
            if abs(p[f"v4_{d}"] - p[f"v3_{d}"]) / p["B"] >= DISC:
                by[p["l2x"]] = by.get(p["l2x"], 0) + 1
    print(f"{len(pred)} launches; discriminating cells by l2x: {dict(sorted(by.items()))}; written {a.out}")
    return 0


def align(rows: list[dict], plan_launches: list[dict]) -> list[dict]:
    first = next((i for i, r in enumerate(rows) if kernel_class(r["kernel"]) == "flush"), None)
    if first is None:
        raise ValueError("no flush kernel in the export")
    seg = rows[first:]
    if len(seg) != len(plan_launches):
        raise ValueError(f"{len(seg)} measured launches after the first flush, plan has {len(plan_launches)}")
    for k, (r, p) in enumerate(zip(seg, plan_launches)):
        if kernel_class(r["kernel"]) != plan_class(p["kernel"]):
            raise ValueError(f"launch {k}: profiler kernel {r['kernel'][:60]!r} vs plan {p['kernel']!r}")
    return seg


def score(pred: list[dict], runs: list[list[dict]], flush_bytes: int, log=print) -> dict:
    v: dict = {}
    fl = [run[i]["read"] for run in runs for i, p in enumerate(pred) if p["kernel"] == "flush"]
    s0 = max(abs(x - flush_bytes) / flush_bytes for x in fl)
    v["R-0"] = s0 <= S0_TOL
    log(f"R-0 flush reads: max rel err {s0 * 100:.2f}% -> {'PASS' if v['R-0'] else 'FAIL (stop)'}")
    cells = []
    first_flush = next(i for i, p in enumerate(pred) if p["kernel"] == "flush")
    for i, p in enumerate(pred):
        dirs = ([] if p["kernel"] == "flush" else ["read"]) + ([] if i == first_flush else ["write"])
        for d in dirs:
            meas = statistics.median(run[i][d] for run in runs) / p["B"]
            v3, v4 = p[f"v3_{d}"] / p["B"], p[f"v4_{d}"] / p["B"]
            cells.append({"i": i, "l2x": p["l2x"], "variant": p["variant"], "mode": p["mode"], "kernel": p["kernel"],
                          "dir": d, "meas": meas, "v3": v3, "v4": v4, "e3": abs(meas - v3), "e4": abs(meas - v4),
                          "disc": abs(v4 - v3) >= DISC})
    f4 = sum(c["e4"] <= R1_E for c in cells) / len(cells)
    f3 = sum(c["e3"] <= R1_E for c in cells) / len(cells)
    v["R-1"] = f4 >= R1_FRAC
    v["R-1 share v4"], v["R-1 share v3"] = round(f4, 3), round(f3, 3)
    log(f"R-1 cells within {R1_E} B: v4 {f4 * 100:.1f}% (v3 {f3 * 100:.1f}%) of {len(cells)}; need >= {R1_FRAC * 100:.0f}% "
        f"-> {'PASS' if v['R-1'] else 'FAIL'}")
    disc = [c for c in cells if c["disc"]]
    if disc:
        w = sum(c["e4"] < c["e3"] for c in disc) / len(disc)
        v["R-2"] = "v4 better" if w >= R2_WIN else "v3 better" if w <= R2_LOSE else "mixed"
        v["R-2 share"] = round(w, 3)
        log(f"R-2 discriminating cells {len(disc)}: v4 strictly closer in {w * 100:.1f}% -> {v['R-2']} "
            f"(mean e v4 {statistics.fmean(c['e4'] for c in disc):.3f}, v3 {statistics.fmean(c['e3'] for c in disc):.3f})")
    else:
        v["R-2"] = "n/a"
    far = [c for c in cells if c["l2x"] >= R3_L2X]
    ff = sum(c["e4"] <= R3_E for c in far) / len(far) if far else float("nan")
    v["R-3"] = bool(far) and ff >= R3_FRAC
    log(f"R-3 far cells (l2x >= {R3_L2X}) within {R3_E} B: {ff * 100:.1f}% of {len(far)} -> {'PASS' if v['R-3'] else 'FAIL'}")
    for var in ("eager", "compiled"):
        cs = [c for c in cells if c["variant"] == var]
        log(f"   {var:8} mean e: v4 {statistics.fmean(c['e4'] for c in cs):.3f}, v3 {statistics.fmean(c['e3'] for c in cs):.3f} ({len(cs)} cells)")
    v["_cells"] = cells
    return v


def cmd_score(a) -> int:
    rec = json.loads(a.prereg.read_text())
    pred = rec["launches"]
    plan = json.loads(a.plan.read_text())
    same = [{k: p[k] for k in p if k != "i"} for p in predictions(plan)] == [{k: p[k] for k in p if k != "i"} for p in pred]
    print(f"plan of the measured run gives identical predictions to the locked ones: {same}")
    runs = [align(load_ncu_raw(p), plan["launches"]) for p in a.ncu_csv]
    v = score(pred, runs, rec["plan_meta"]["flush_bytes"])
    v["plan_consistent"] = same
    cells = v.pop("_cells")
    if a.verbose:
        print(f"\n{'l2x':>7} {'variant':8} {'mode':8} {'kernel':15} d     meas    v3    v4")
        for c in cells:
            print(f"{c['l2x']:7g} {c['variant']:8} {c['mode']:8} {c['kernel']:15} {c['dir'][0].upper()} "
                  f"{c['meas']:8.3f} {c['v3']:5.2f} {c['v4']:5.2f}{'  *' if c['disc'] else ''}")
    print("verdict:", json.dumps(v))
    if a.json_out:
        a.json_out.write_text(json.dumps({"verdict": v, "cells": cells}, indent=1))
    return 0


def cmd_time(a) -> int:
    """T-1 (pre-registered) + descriptive speedup table; median over runs of per-run medians."""
    import csv
    per = {}
    for k, path in enumerate(a.csv):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                key = (float(row["l2x"]), row["variant"], row["timing"])
                per.setdefault(key, {}).setdefault(k, []).append(float(row["us_per_call"]))
    med = {key: statistics.median(statistics.median(v) for v in runs.values()) for key, runs in per.items()}
    sizes = sorted({key[0] for key in med})
    timings = ("graph_isolated", "graph_steady", "e2e_steady")
    print(f"{'l2x':>7} " + " ".join(f"{t:>28}" for t in timings) + "   (eager / compiled us, speedup)")
    for f in sizes:
        cells = []
        for t in timings:
            e, c = med.get((f, "eager", t)), med.get((f, "compiled", t))
            cells.append(f"{e:9.1f} /{c:8.1f} ({e / c:4.2f}x)" if e and c else f"{'-':>28}")
        print(f"{f:7g} " + " ".join(f"{x:>28}" for x in cells))
    sp = {f: med[(f, "eager", "graph_steady")] / med[(f, "compiled", "graph_steady")] for f in T1_L2X if (f, "eager", "graph_steady") in med}
    ok = len(sp) == len(T1_L2X) and all(T1_LO <= x <= T1_HI for x in sp.values())
    print(f"T-1 graph_steady speedup at l2x {T1_L2X}: " + ", ".join(f"{x:.2f}" for x in sp.values())
          + f" (must be in [{T1_LO}, {T1_HI}]) -> {'PASS' if ok else 'FAIL'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--prereg", type=Path, required=True)
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--ncu-csv", type=Path, nargs="+", required=True)
    s.add_argument("--json-out", type=Path)
    s.add_argument("--verbose", action="store_true")
    t = sub.add_parser("time")
    t.add_argument("--csv", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    return {"predict": cmd_predict, "score": cmd_score, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
