#!/usr/bin/env python3
"""Counter pre-registration: per-launch DRAM read / write bytes from the *unchanged* L2
simulator, to be compared with Nsight Compute counters -- ESTIMATES only.

Why a new module: v2/v3 simulate one timed window at a time from an empty cache (S5) and report
read misses + dirty evictions as one number. A profiler reports bytes per kernel launch and
separates reads from writes. So this module replays each size's whole *timed* launch sequence
exactly as fusion.cu / stencil.cu issue it with `--reps 1 --warmup 0 --steady-iters 3` through
one Cache (cache_sim.py, unchanged), the L2-flush kernel included, and predicts for every launch:
    read  = read misses x 128 B            (compare: dram__bytes_read.sum)
    write = dirty evictions x 128 B        (compare: dram__bytes_write.sum)
Deferred write-back becomes directly testable: dirty lines a window leaves behind are written
back during the *next* kernel that evicts them (usually the flush kernel).

Differences from v2/v3 accounting (declared before any counter is read):
  * the cache is never empty: after a flush it is full of clean flush lines. For LRU this equals
    S5 exactly (unit-tested); for random replacement new lines now compete with clean flush lines,
    so random retains less than the v2/v3 "random" did.
  * the first flush of every size follows the untimed correctness phase, which is not modelled:
    its write bytes are UNSCORED (its read bytes are the flush-size sanity check S-0).

  python3 benchmarks/scripts/counter_prereg.py predict --l2-bytes 3145728 \
      --sizes-l2x 0.25,0.46875,0.5625,0.75,1,2,4 --gpu-label rtx3060-laptop \
      --prereg research/preregistration/counters_rtx3060laptop.json
  python3 benchmarks/scripts/counter_prereg.py score --bench fusion \
      --ncu-csv <raw.csv> --prereg <json>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

import cache_sim_b5
from cache_sim import LINE, Cache
from harness_env import sha256_of

POLICIES = ("lru", "random")
FLUSH = "flush_l2_read_kernel"
STEADY_ITERS = 3

# Decision rules (see research/HYPOTHESES.md, "Counter pre-registration"). e = |meas - pred| / B,
# B = bytes of one array (4 n). Thresholds are fixed here so the scorer cannot drift from the text.
E_CELL = 0.10        # C-1 / C-4 per-cell tolerance, in arrays
FRAC_CELLS = 0.90    # C-1 / C-4: share of cells that must be within E_CELL
FAR_L2X = (2.0, 4.0)  # DRAM-bound: LRU and random agree within 0.02 B per cell (C-1 / C-4)
# L2-resident and cliff zone: the policies disagree by up to ~0.4 B per cell (C-3, directional)
EDGE_L2X = (0.25, 0.46875, 0.5625, 0.75, 1.0)
S0_TOL = 0.05        # S-0: flush-kernel read bytes within 5 % of the flush buffer size


def flush_bytes(l2: int) -> int:
    """== Timer::init in common.cuh: max(4 x L2, 16 MiB), whole uint32."""
    return max(4 * l2, 16 << 20) // 4 * 4


def n_for(bench: str, l2x: float, l2: int) -> int:
    if bench == "fusion":
        return max(256, int(l2x * l2 / 4) // 256 * 256)  # == fusion.cu
    return cache_sim_b5.n_for(l2x, l2)                  # == stencil.cu


def launch_plan(bench: str) -> list[tuple[str, str, str | None, str | None]]:
    """(label, kernel name, src buffer, dst buffer) for one size's timed phase, in issue order.
    label = window|mode|role. Buffers: fusion X,T,Y; stencil X,P,Q; None = flush."""
    plan: list[tuple[str, str, str | None, str | None]] = []

    def flush(window):
        plan.append((f"{window}|flush", FLUSH, None, None))

    if bench == "fusion":
        F, S1, S2 = "fused_kernel", "stage1_kernel", "stage2_kernel"
        flush("fused|isolated")
        plan.append(("fused|isolated|k1", F, "X", "Y"))
        for it in range(STEADY_ITERS):
            plan.append((f"fused|steady{it + 1}|k1", F, "X", "Y"))
        flush("materialized|isolated")
        plan += [("materialized|isolated|k1", S1, "X", "T"), ("materialized|isolated|k2", S2, "T", "Y")]
        for it in range(STEADY_ITERS):
            plan += [(f"materialized|steady{it + 1}|k1", S1, "X", "T"),
                     (f"materialized|steady{it + 1}|k2", S2, "T", "Y")]
        flush("stage2_cold|isolated")
        plan.append(("stage2_cold|isolated|k1", S2, "T", "Y"))
        flush("stage2_warm|isolated")
        plan += [("stage2_warm|pre|k1", S1, "X", "T"), ("stage2_warm|isolated|k1", S2, "T", "Y")]
        return plan
    if bench == "stencil":
        K5, K13, K13C = "sweep5_kernel", "sweep13_kernel", "sweep13c_kernel"
        mat4 = [("X", "P"), ("P", "Q"), ("Q", "P"), ("P", "Q")]
        f2 = [("X", "P"), ("P", "Q")]
        for name, kern, calls in (("mat4", K5, mat4), ("fused2", K13, f2), ("fused2c", K13C, f2)):
            flush(f"{name}|isolated")
            plan += [(f"{name}|isolated|k{i + 1}", kern, s, d) for i, (s, d) in enumerate(calls)]
            for it in range(STEADY_ITERS):
                plan += [(f"{name}|steady{it + 1}|k{i + 1}", kern, s, d) for i, (s, d) in enumerate(calls)]
        flush("sweep_cold|isolated")
        plan.append(("sweep_cold|isolated|k1", K5, "P", "Q"))
        flush("sweep_warm|isolated")
        plan += [("sweep_warm|pre|k1", K5, "X", "P"), ("sweep_warm|isolated|k1", K5, "P", "Q")]
        return plan
    raise ValueError(bench)


def simulate_launches(bench: str, n: int, l2: int, policy: str, seed: int = 1) -> list[dict]:
    """Per-launch predicted DRAM read / write bytes for one size, in issue order."""
    n_lines = max(1, (4 * n + LINE - 1) // LINE)
    f_lines = (flush_bytes(l2) + LINE - 1) // LINE
    names = ("X", "T", "Y") if bench == "fusion" else ("X", "P", "Q")
    base = {b: i * n_lines for i, b in enumerate(names)}
    fbase = len(names) * n_lines
    c = Cache(max(1, l2 // LINE), policy, seed)
    out = []
    for i, (label, kern, src, dst) in enumerate(launch_plan(bench)):
        c.reset_counters()
        if src is None:
            for j in range(f_lines):
                c.access(fbase + j, False)
        else:
            s, d = base[src], base[dst]
            for j in range(n_lines):  # == cache_sim._kernel
                c.access(s + j, False)
                c.access(d + j, True)
        out.append({"i": i, "label": label, "kernel": kern,
                    "read": c.reads * LINE, "write": c.writebacks * LINE,
                    "scored_write": not (i == 0 and src is None)})
    return out


def window_sums(launches: list[dict]) -> dict[tuple[str, str], int]:
    """read + write summed over the kernels of each timed window (flush and 'pre' excluded),
    keyed like cache_sim.simulate_fusion; steady = the STEADY_ITERS-th repetition."""
    out: dict[tuple[str, str], int] = {}
    for L in launches:
        w, mode, role = L["label"].split("|")
        if role == "flush" or mode == "pre":
            continue
        if mode.startswith("steady"):
            if mode != f"steady{STEADY_ITERS}":
                continue
            mode = "steady"
        out[(w, mode)] = out.get((w, mode), 0) + L["read"] + L["write"]
    return out


def cmd_predict(args) -> int:
    l2 = args.l2_bytes
    sizes = [float(x) for x in args.sizes_l2x.split(",")]
    rec = {"kind": "counter pre-registration (per-launch DRAM bytes)", "gpu_label": args.gpu_label,
           "note": args.note, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "sha256": {f: sha256_of(Path(__file__).with_name(f)) for f in
                      ("cache_sim.py", "cache_sim_b5.py", "counter_prereg.py")},
           "l2_bytes": l2, "flush_bytes": flush_bytes(l2), "steady_iters": STEADY_ITERS,
           "binary_args": "--ks 1 --reps 1 --warmup 0 --steady-iters 3 (fusion); "
                          "--reps 1 --warmup 0 --steady-iters 3 (stencil)",
           "rules": {"E_CELL": E_CELL, "FRAC_CELLS": FRAC_CELLS, "FAR_L2X": FAR_L2X, "EDGE_L2X": EDGE_L2X,
                     "S0_TOL": S0_TOL},
           "benches": {}}
    for bench in ("fusion", "stencil"):
        rows = []
        for f in sizes:
            n = n_for(bench, f, l2)
            for pol in POLICIES:
                rows.append({"l2x": f, "n": n, "array_bytes": 4 * n, "policy": pol,
                             "launches": simulate_launches(bench, n, l2, pol)})
        rec["benches"][bench] = {"sizes_l2x": sizes, "plan": [p[:2] for p in launch_plan(bench)], "rows": rows}
        print(f"\n== {bench}: predicted DRAM bytes per launch, in arrays (B = 4n); LRU / random")
        for f in sizes:
            lr = next(r for r in rows if r["l2x"] == f and r["policy"] == "lru")
            rn = next(r for r in rows if r["l2x"] == f and r["policy"] == "random")
            B = lr["array_bytes"]
            cells = []
            for a, b in zip(lr["launches"], rn["launches"]):
                w, mode, role = a["label"].split("|")
                if mode.startswith("steady") and mode != "steady3":
                    continue
                cells.append(f"{w[:6]}.{mode[:4]}.{role}: R {a['read'] / B:.2f}/{b['read'] / B:.2f} "
                             f"W {a['write'] / B:.2f}/{b['write'] / B:.2f}")
            print(f"l2x {f:g} (B = {B / 2**20:.3f} MiB)\n   " + "\n   ".join(cells))
    if args.prereg:
        args.prereg.parent.mkdir(parents=True, exist_ok=True)
        args.prereg.write_text(json.dumps(rec, indent=1))
        print(f"\npre-registration written: {args.prereg}")
    return 0


# ----------------------------------------------------------------------------------------------
# Scoring (reads an `ncu --import <rep> --csv --page raw --print-units base` export)
# ----------------------------------------------------------------------------------------------
UNIT_SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
              "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
M_READ, M_WRITE = "dram__bytes_read.sum", "dram__bytes_write.sum"


def load_ncu_raw(path: Path) -> list[dict]:
    """Rows of an ncu raw CSV export: {'kernel', 'read', 'write'} in launch-ID order."""
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.startswith("==")]  # drop ==PROF== / ==WARNING== lines
    rows = list(csv.reader(lines))
    hdr = rows[0]
    units = rows[1] if len(rows) > 1 and not rows[1][0].strip().isdigit() else None
    body = rows[2:] if units else rows[1:]
    col = {h: i for i, h in enumerate(hdr)}
    for m in (M_READ, M_WRITE, "Kernel Name", "ID"):
        if m not in col:
            raise ValueError(f"{path}: column {m!r} missing")

    def val(r, m):
        x = float(r[col[m]].replace(",", ""))
        u = units[col[m]].strip() if units else "byte"
        if u not in UNIT_SCALE:
            raise ValueError(f"unknown unit {u!r} for {m}")
        return x * UNIT_SCALE[u]

    def kname(r):
        # ncu exports the same kernel as "dmk::flush_l2_read_kernel(...)" in one report and
        # "flush_l2_read_kernel(...)" in another; launch_plan() uses unqualified names.
        return re.sub(r"^(?:\w+::)+", "", r[col["Kernel Name"]].split("(")[0].strip())

    out = [{"id": int(r[col["ID"]]), "kernel": kname(r),
            "read": val(r, M_READ), "write": val(r, M_WRITE)} for r in body if r and r[0].strip()]
    return sorted(out, key=lambda d: d["id"])


def align(measured: list[dict], bench: str, n_sizes: int) -> list[list[dict]]:
    """Locate each size's timed phase: it starts at the first flush after the previous phase and
    must match launch_plan() name for name. Raises if the sequence differs (no silent mapping)."""
    plan = [p[1] for p in launch_plan(bench)]
    out, pos = [], 0
    for s in range(n_sizes):
        while pos < len(measured) and measured[pos]["kernel"] != FLUSH:
            pos += 1
        seg = measured[pos:pos + len(plan)]
        got = [m["kernel"] for m in seg]
        if got != plan:
            raise ValueError(f"{bench} size #{s}: launch sequence differs from launch_plan at ID "
                             f"{seg[0]['id'] if seg else '?'}: {got[:8]}... vs {plan[:8]}...")
        out.append(seg)
        pos += len(plan)
    return out


def cmd_score(args) -> int:
    rec = json.loads(args.prereg.read_text())
    bench_rec = rec["benches"][args.bench]
    sizes = bench_rec["sizes_l2x"]
    segs = align(load_ncu_raw(args.ncu_csv), args.bench, len(sizes))
    fb = rec["flush_bytes"]
    verdict = {}
    # S-0: flush read bytes == flush buffer size (are we reading the counters we think we are?)
    fl = [m["read"] for seg in segs for m in seg if m["kernel"] == FLUSH]
    s0 = [abs(x - fb) / fb for x in fl]
    verdict["S-0"] = max(s0) <= S0_TOL
    print(f"S-0 flush read bytes: median {statistics.median(fl) / 2**20:.3f} MiB vs {fb / 2**20:.3f} MiB; "
          f"max rel err {max(s0) * 100:.1f}% -> {'PASS' if verdict['S-0'] else 'FAIL (stop: counters not as assumed)'}")
    errs = {p: {"far": [], "edge": [], "all": []} for p in POLICIES}
    for f, seg in zip(sizes, segs):
        for pol in POLICIES:
            row = next(r for r in bench_rec["rows"] if r["l2x"] == f and r["policy"] == pol)
            B = row["array_bytes"]
            for meas, pred in zip(seg, row["launches"]):
                cells = [("R", meas["read"], pred["read"])] if pred["kernel"] != FLUSH else []
                if pred["scored_write"]:
                    cells.append(("W", meas["write"], pred["write"]))
                for d, mv, pv in cells:
                    e = abs(mv - pv) / B
                    errs[pol]["all"].append((f, pred["label"], d, mv / B, pv / B, e))
                    if f in FAR_L2X:
                        errs[pol]["far"].append(e)
                    if f in EDGE_L2X:
                        errs[pol]["edge"].append(e)
    if args.verbose:
        for pol in POLICIES:
            print(f"\n-- {pol}: l2x label dir measured/predicted (arrays) e")
            for f, lab, d, mv, pv, e in errs[pol]["all"]:
                print(f"{f:8g} {lab:32} {d} {mv:6.3f}/{pv:6.3f}  {e:.3f}{'  <-- >E_CELL' if e > E_CELL else ''}")
    print()
    for pol in POLICIES:
        far, edge = errs[pol]["far"], errs[pol]["edge"]
        frac = sum(e <= E_CELL for e in far) / len(far) if far else float("nan")
        print(f"{pol:>6}: far cells within {E_CELL} B: {frac * 100:.1f}% of {len(far)}; "
              f"edge mean e {statistics.fmean(edge) if edge else float('nan'):.3f} over {len(edge)}")
    best = max(POLICIES, key=lambda p: sum(e <= E_CELL for e in errs[p]["far"]))
    frac_best = sum(e <= E_CELL for e in errs[best]["far"]) / max(1, len(errs[best]["far"]))
    rule = "C-1" if args.bench == "fusion" else "C-4"
    verdict[rule] = frac_best >= FRAC_CELLS
    edge_win = min(POLICIES, key=lambda p: statistics.fmean(errs[p]["edge"]) if errs[p]["edge"] else 9e9)
    print(f"{rule}: {'PASS' if verdict[rule] else 'FAIL'} (best policy {best}: {frac_best * 100:.1f}% >= "
          f"{FRAC_CELLS * 100:.0f}% required)")
    print(f"C-3 (directional, cliff zone): lower mean e -> {edge_win}")
    if args.bench == "fusion" and 0.25 in sizes:
        seg = segs[sizes.index(0.25)]
        B = next(r for r in bench_rec["rows"] if r["l2x"] == 0.25)["array_bytes"]
        lab = [p[0] for p in launch_plan("fusion")]
        fused_w = seg[lab.index("fused|isolated|k1")]["write"] / B
        defer_w = seg[lab.index("stage2_warm|isolated|flush")]["write"] / B
        verdict["C-2"] = fused_w < 0.5 and defer_w > 0.5
        print(f"C-2 deferred write-back @ l2x 0.25: fused-window DRAM write {fused_w:.3f} B (must be < 0.5), "
              f"flush after stage2_cold writes {defer_w:.3f} B (must be > 0.5) -> "
              f"{'PASS' if verdict['C-2'] else 'FAIL'}")
    print("verdict:", json.dumps(verdict))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--l2-bytes", type=int, required=True)
    p.add_argument("--sizes-l2x", required=True)
    p.add_argument("--prereg", type=Path)
    p.add_argument("--gpu-label", default="")
    p.add_argument("--note", default="")
    s = sub.add_parser("score")
    s.add_argument("--bench", choices=("fusion", "stencil"), required=True)
    s.add_argument("--ncu-csv", type=Path, required=True)
    s.add_argument("--prereg", type=Path, required=True)
    s.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return cmd_predict(args) if args.cmd == "predict" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
