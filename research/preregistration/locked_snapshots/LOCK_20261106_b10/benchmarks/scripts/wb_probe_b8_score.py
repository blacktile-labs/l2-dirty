"""B8: replacement near L2 capacity. Rules Z-0..Z-3 pre-registered in research/HYPOTHESES.md,
section "B8 pre-registration". Same binary (wb_probe, full lines) and launch sequence as B6b/B6c.

Models (cache_sim_v5cand.CandidateL2, both with the dirty cap D1):
  v4    "lru"                      LRU victim; a dirty victim is written back and evicted
  dvsc  "dvsc_life", life 0.125    a dirty victim is written back and stays readable for
                                   0.125 x capacity further fills (life fitted on B6c data, D-046)
Groups (none measured before):
  M  mixed, r + w in {1.0625, 1.1875, 1.3125, 1.4375}, w in {3/16, 5/16}
  W  write-only above capacity, w in {1.0625, 1.125, 1.25}
  R  read-only around capacity, r in {0.90625 ... 1.0625}  (both models: RD = 0 up to 1.0, 1 above)
  controls 0.5:0.25, 2:0, 0:2 and the B6c replication cell 1:0.125 (B6c RB = 0.085)
  python3 wb_probe_b8_score.py cells | predict --out <json> | score --prereg ... --meta ... --kernel-csv ... --range-csv ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import wb_probe_score as ws
from cache_sim_v5cand import CandidateL2
from counter_prereg import load_ncu_raw
from harness_env import sha256_of

L2_BYTES, LINE, STEADY = 3145728, 128, 5
LIFE = 0.125
M = [(0.875, 0.1875), (1.0, 0.1875), (1.125, 0.1875), (1.25, 0.1875),
     (0.75, 0.3125), (0.875, 0.3125), (1.0, 0.3125), (1.125, 0.3125)]
W = [(0.0, 1.0625), (0.0, 1.125), (0.0, 1.25)]
R = [(0.90625, 0.0), (0.9375, 0.0), (0.96875, 0.0), (1.0, 0.0), (1.03125, 0.0), (1.0625, 0.0)]
CONTROLS = [(0.5, 0.25), (2.0, 0.0), (0.0, 2.0), (1.0, 0.125)]
CELLS = M + W + R + CONTROLS
MODELS = {"v4": lambda cap: CandidateL2(cap, "lru"),
          "dvsc": lambda cap: CandidateL2(cap, "dvsc_life", life_frac=LIFE)}
BAND = (0.90, 1.10)
DISC = 0.15
Z1_WIN, Z1_LOSE, Z1_MAE = 0.75, 0.25, 0.15
Z2_R, Z2_RD = 0.9375, 0.10
Z2B_R, Z2B_RD = 1.03125, 0.90
REPL_CELL, REPL_RB, REPL_TOL = (1.0, 0.125), 0.085, 0.10


def elems(f: float) -> int:  # == wb_probe.cu elems_for
    return 0 if f == 0 else max(256, int(f * L2_BYTES / 4) // 256 * 256)


def simulate(model: str, r: float, w: float) -> dict:
    cap = L2_BYTES // LINE
    nc, nd = elems(r) * 4 // LINE, elems(w) * 4 // LINE
    c = MODELS[model](cap)
    C, D, F = 0, 1 << 24, 1 << 25
    for j in range(max(4 * L2_BYTES, 16 << 20) // LINE):
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
    s2 = st[2:]
    return {"RD": statistics.fmean(x for x, _ in s2) / nc if nc else None,
            "WB": statistics.fmean(y for _, y in s2) / nd if nd else None,
            "RB": c.reads / nd if nd else None}


def inb(x) -> bool:
    return x is not None and BAND[0] <= x <= BAND[1]


def score(cells: dict, prereg: dict, log=print) -> dict:
    k = ws.key
    pred = {k(c["r"], c["w"]): c for c in prereg["cells"]}
    v: dict = {}
    a_r, a_w, ctl = cells[k(2, 0)], cells[k(0, 2)], cells[k(0.5, 0.25)]
    warm = [c["WARM"] for c in cells.values() if c["WARM"] is not None]
    ok = (inb(a_r["RD"]) and inb(a_r["RDR"]) and inb(a_w["WB"]) and inb(a_w["WBR"])
          and ctl["WBR"] < 0.05 and ctl["RDR"] < 0.05 and all(inb(x) for x in warm))
    v["Z-0"] = ok
    rep = cells[k(*REPL_CELL)]["RB"]
    v["Z-0 replication"] = abs(rep - REPL_RB) <= REPL_TOL
    log(f"Z-0 sanity -> {'PASS' if ok else 'FAIL (stop)'}; replication RB{REPL_CELL} = {rep:.3f} (B6c {REPL_RB}) "
        f"-> {'PASS' if v['Z-0 replication'] else 'FAIL'}")
    mw = [k(r, w) for r, w in M + W]
    disc = [x for x in mw if abs(pred[x]["v4"]["RB"] - pred[x]["dvsc"]["RB"]) >= DISC]
    closer = [abs(cells[x]["RB"] - pred[x]["dvsc"]["RB"]) < abs(cells[x]["RB"] - pred[x]["v4"]["RB"]) for x in disc]
    share = sum(closer) / len(closer)
    mae_d = statistics.fmean(abs(cells[x]["RB"] - pred[x]["dvsc"]["RB"]) for x in mw)
    mae_v = statistics.fmean(abs(cells[x]["RB"] - pred[x]["v4"]["RB"]) for x in mw)
    v["Z-1"] = ("dvsc supported" if share >= Z1_WIN and mae_d <= Z1_MAE else
                "v4 (LRU) supported" if share <= Z1_LOSE else "mixed")
    v["Z-1 share"], v["Z-1 MAE dvsc"], v["Z-1 MAE v4"] = round(share, 3), round(mae_d, 3), round(mae_v, 3)
    log(f"Z-1 RB on {len(disc)} discriminating cells: dvsc closer in {share * 100:.0f}%; MAE over M+W "
        f"dvsc {mae_d:.3f} vs v4 {mae_v:.3f} (dvsc needs <= {Z1_MAE}) -> {v['Z-1']}")
    rd = cells[k(Z2_R, 0)]["RDR"]
    v["Z-2"] = "effective capacity < 0.94 L2" if rd >= Z2_RD else "effective capacity >= 0.94 L2"
    rdb = cells[k(Z2B_R, 0)]["RDR"]
    v["Z-2b"] = "reads retained just above capacity (not LRU-sharp)" if rdb <= Z2B_RD else "LRU-sharp cliff"
    curve = ", ".join(f"{r:g}:{cells[k(r, 0)]['RDR']:.3f}" for r, _ in R)
    log(f"Z-2 RDR(0.9375) = {rd:.3f} -> {v['Z-2']}; Z-2b RDR(1.03125) = {rdb:.3f} -> {v['Z-2b']}; curve {curve}")
    wb = [cells[x]["WBR"] for x in mw]
    v["Z-3 WBR median M+W"] = round(statistics.median(wb), 3)
    log(f"Z-3 (both models predict WB = 1 in M and W) median WBR = {v['Z-3 WBR median M+W']}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cells")
    p = sub.add_parser("predict")
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--prereg", type=Path, required=True)
    s.add_argument("--meta", type=Path, required=True)
    s.add_argument("--kernel-csv", type=Path, nargs="+", required=True)
    s.add_argument("--range-csv", type=Path, nargs="+", required=True)
    s.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    if a.cmd == "cells":
        print(",".join(f"{r:g}:{w:g}" for r, w in CELLS))
        return 0
    if a.cmd == "predict":
        rec = {"kind": "B8 predictions (v4 LRU vs dirty-victim second chance, life 0.125)",
               "sha256": {f: sha256_of(Path(__file__).with_name(f)) for f in
                          ("cache_sim_v4.py", "cache_sim_v5cand.py", "wb_probe_b8_score.py")},
               "cells": [{"r": r, "w": w, **{m: simulate(m, r, w) for m in MODELS}} for r, w in CELLS]}
        a.out.write_text(json.dumps(rec, indent=1))
        print(f"written {a.out}")
        return 0
    prereg = json.loads(a.prereg.read_text())
    meta = json.loads(a.meta.read_text())
    n, steady = len(meta["cells"]), int(meta["steady"])
    kr = [ws.align_kernel(load_ncu_raw(p), n, steady, *ws.kernel_names(meta)) for p in a.kernel_csv]
    rr = [ws.align_range(load_ncu_raw(p), n) for p in a.range_csv]
    cells = ws.summarize(meta, kr, rr)
    ws.table(cells)
    print(f"\n{'r':>8} {'w':>7}    RB  v4.RB dvsc.RB |   WBR |   RDR")
    for c in prereg["cells"]:
        m = cells[ws.key(c["r"], c["w"])]
        rb = f"{m['RB']:6.3f} {c['v4']['RB']:6.3f} {c['dvsc']['RB']:6.3f}" if c["w"] else "     -      -       -"
        print(f"{c['r']:8g} {c['w']:7g} {rb} | {m['WBR'] if m['WBR'] is None else round(m['WBR'], 3)!s:>5} | "
              f"{m['RDR'] if m['RDR'] is None else round(m['RDR'], 3)!s:>5}")
    v = score(cells, prereg)
    print("verdict:", json.dumps(v))
    if a.json_out:
        a.json_out.write_text(json.dumps({"cells": [cells[x] for x in sorted(cells)], "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
