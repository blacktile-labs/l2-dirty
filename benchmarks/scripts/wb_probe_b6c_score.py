"""B6c: does the L2 prefer to evict clean lines (keep dirty ones)? Rules X-0..X-4 pre-registered in
research/HYPOTHESES.md, section "B6c pre-registration".

Cells have a working set above L2 (r + w > 1) with a dirty footprint below the dirty cap (w <= 3/8),
so D1 (dirty cap) never fires and DRAM writes come from evictions only.
Competing models, simulated on the exact wb_probe launch sequence (flush, warm, steady x 5, readback):
  v4  cache_sim_v4.DirtyCapLRU          (LRU victim)            -> WB = 1, RB 0.33..1
  cf  cache_sim_cf.CleanFirstDirtyCap   (clean-first victim, D2) -> WB = 0, RB = 0
  python3 wb_probe_b6c_score.py predict --out <json>
  python3 wb_probe_b6c_score.py score --prereg <json> --meta m.json --kernel-csv k1 k2 k3 --range-csv r1 r2 r3
Constants must not change after data is seen.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import wb_probe_score as ws
from cache_sim_cf import CleanFirstDirtyCap
from cache_sim_v4 import DirtyCapLRU
from counter_prereg import load_ncu_raw
from harness_env import sha256_of

L2_BYTES = 3145728
LINE = 128
STEADY = 5
MIXED = [(0.875, 0.25), (1.0, 0.25), (1.25, 0.25), (1.75, 0.25),
         (0.75, 0.375), (0.875, 0.375), (1.125, 0.375), (1.625, 0.375),
         (1.0, 0.125), (1.125, 0.125), (1.375, 0.125), (1.875, 0.125)]
CONTROLS = [(0.5, 0.25), (2.0, 0.0), (0.0, 2.0)]
EXPLORATORY = [(0.625, 0.5), (0.5, 0.75)]
CELLS = MIXED + CONTROLS + EXPLORATORY
BAND = (0.90, 1.10)
CTRL_MAX = 0.05
X1_D2, X1_V4 = 0.75, 0.25
X2_FULL, X2_NONE = 0.10, 0.90
X3_RES, X3_EVI = 0.25, 0.75
X4_RD, X4_FRAC = 0.90, 0.75
MODELS = {"v4": DirtyCapLRU, "cf": CleanFirstDirtyCap}


def cells_arg() -> str:
    return ",".join(f"{r:g}:{w:g}" for r, w in CELLS)


def elems(f: float) -> int:  # == wb_probe.cu elems_for
    return 0 if f == 0 else max(256, int(f * L2_BYTES / 4) // 256 * 256)


def simulate(model: str, r: float, w: float) -> dict:
    """Per-launch steady RD, WB (launches 2..S, as fractions) and readback RB for one cell."""
    cap = L2_BYTES // LINE
    nc, nd = elems(r) * 4 // LINE, elems(w) * 4 // LINE
    c = MODELS[model](cap)
    C, D, F = 0, 1 << 24, 1 << 25
    for j in range(max(4 * L2_BYTES, 16 << 20) // LINE):  # flush
        c.access(F + j, False)
    st = []
    for _ in range(1 + STEADY):
        c.reset_counters()
        for j in range(max(nc, nd)):
            if j < nc:
                c.access(C + j, False)
            if j < nd:
                c.access(D + j, True)
        st.append((c.reads, c.writebacks))
    c.reset_counters()
    for j in range(nd):
        c.access(D + j, False)
    st2 = st[2:]
    return {"RD": statistics.fmean(x for x, _ in st2) / nc if nc else None,
            "WB": statistics.fmean(y for _, y in st2) / nd if nd else None,
            "RB": c.reads / nd if nd else None}


def cmd_predict(a) -> int:
    rec = {"kind": "B6c predictions (v4 LRU vs clean-first)",
           "sha256": {f: sha256_of(Path(__file__).with_name(f)) for f in
                      ("cache_sim_v4.py", "cache_sim_cf.py", "wb_probe_b6c_score.py")},
           "cells": []}
    for r, w in CELLS:
        rec["cells"].append({"r": r, "w": w, **{m: simulate(m, r, w) for m in MODELS}})
        p = rec["cells"][-1]
        print(f"r={r:<6g} w={w:<6g}  v4 {p['v4']}  cf {p['cf']}")
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"written: {a.out}")
    return 0


def inb(x) -> bool:
    return x is not None and BAND[0] <= x <= BAND[1]


def score(cells: dict, prereg: dict, log=print) -> dict:
    v: dict = {}
    k = ws.key
    pred = {k(c["r"], c["w"]): c for c in prereg["cells"]}
    a_r, a_w, ctl = cells[k(2, 0)], cells[k(0, 2)], cells[k(0.5, 0.25)]
    warm = [c["WARM"] for c in cells.values() if c["WARM"] is not None]
    ok = (inb(a_r["RD"]) and inb(a_r["RDR"]) and inb(a_w["WB"]) and inb(a_w["WBR"])
          and ctl["WBR"] < CTRL_MAX and ctl["RDR"] < CTRL_MAX and all(inb(x) for x in warm))
    v["X-0"] = ok
    log(f"X-0 sanity: 2:0 RD/RDR {a_r['RD']:.3f}/{a_r['RDR']:.3f}; 0:2 WB/WBR {a_w['WB']:.3f}/{a_w['WBR']:.3f}; "
        f"0.5:0.25 WBR/RDR {ctl['WBR']:.3f}/{ctl['RDR']:.3f}; WARM [{min(warm):.3f}, {max(warm):.3f}] "
        f"-> {'PASS' if ok else 'FAIL (stop: do not interpret)'}")
    mixed = [cells[k(r, w)] for r, w in MIXED]
    closer = [abs(c["WBR"] - pred[k(c["r"], c["w"])]["cf"]["WB"]) < abs(c["WBR"] - pred[k(c["r"], c["w"])]["v4"]["WB"])
              for c in mixed]
    share = sum(closer) / len(closer)
    v["X-1"] = ("clean-first (D2) supported" if share >= X1_D2 else
                "LRU victim (v4) supported" if share <= X1_V4 else "mixed")
    v["X-1 share"] = round(share, 3)
    log(f"X-1 WBR closer to clean-first in {share * 100:.0f}% of {len(mixed)} mixed cells -> {v['X-1']}")
    med = statistics.median(c["WBR"] for c in mixed)
    v["X-2"] = ("full protection" if med <= X2_FULL else "no protection" if med >= X2_NONE else "partial protection")
    v["X-2 median WBR"] = round(med, 3)
    log(f"X-2 median WBR over mixed cells = {med:.3f} -> {v['X-2']}")
    rb = statistics.median(c["RB"] for c in mixed)
    v["X-3"] = "resident" if rb <= X3_RES else "evicted" if rb >= X3_EVI else "partial"
    v["X-3 median RB"] = round(rb, 3)
    log(f"X-3 median RB (readback misses of D) over mixed cells = {rb:.3f} -> {v['X-3']}")
    frac = sum(c["RDR"] >= X4_RD for c in mixed) / len(mixed)
    v["X-4"] = frac >= X4_FRAC
    log(f"X-4 mixed cells with RDR >= {X4_RD}: {frac * 100:.0f}% (both models predict RD = 1) "
        f"-> {'as predicted' if v['X-4'] else 'BOTH models wrong about clean misses'}")
    return v


def cmd_score(a) -> int:
    prereg = json.loads(a.prereg.read_text())
    meta = json.loads(a.meta.read_text())
    n, steady = len(meta["cells"]), int(meta["steady"])
    kr = [ws.align_kernel(load_ncu_raw(p), n, steady, *ws.kernel_names(meta)) for p in a.kernel_csv]
    rr = [ws.align_range(load_ncu_raw(p), n) for p in a.range_csv]
    cells = ws.summarize(meta, kr, rr)
    ws.table(cells)
    print(f"\n{'r':>6} {'w':>6}   WBR   v4.WB  cf.WB |   RB   v4.RB  cf.RB |  RDR")
    for c in prereg["cells"]:
        m = cells[ws.key(c["r"], c["w"])]
        if c["w"] and c["r"]:
            print(f"{c['r']:6g} {c['w']:6g} {m['WBR']:6.3f} {c['v4']['WB']:6.3f} {c['cf']['WB']:6.3f} | "
                  f"{m['RB']:6.3f} {c['v4']['RB']:6.3f} {c['cf']['RB']:6.3f} | {m['RDR']:6.3f}")
    v = score(cells, prereg)
    print("verdict:", json.dumps(v))
    if a.json_out:
        a.json_out.write_text(json.dumps({"cells": [cells[x] for x in sorted(cells)], "verdict": v}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--prereg", type=Path, required=True)
    s.add_argument("--meta", type=Path, required=True)
    s.add_argument("--kernel-csv", type=Path, nargs="+", required=True)
    s.add_argument("--range-csv", type=Path, nargs="+", required=True)
    s.add_argument("--json-out", type=Path)
    sub.add_parser("cells")
    a = ap.parse_args()
    if a.cmd == "cells":
        print(cells_arg())
        return 0
    return cmd_predict(a) if a.cmd == "predict" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
