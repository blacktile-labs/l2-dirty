"""B10b first-cell test. Rules F-0..F-2 and V-4' pre-registered in research/HYPOTHESES.md,
section "B11 pre-registration", before any cell of this run was measured on any GPU.

Post-hoc observation behind it (B10, D-057): on GA102 the two inactive cells with an anomalous
range-mode WBR (full 0:0.1875 = 0.031, sector 0:0.375 = 0.143) were each the FIRST cell of their
process, while their per-kernel WB was ~0. Hypothesis: range-mode writes in the first cell of a
process are inflated (e.g. write-backs of setup data), which made the V-4 control fail.
Test: the B6b sector set with a dummy cell 0:0.25 prepended (one change; same binary and flags).

  F-0  sanity: correctness 0 in every run; sector 0:2 WB and WBR in [0.9, 1.1]
  F-1  (primary) control WBR(0:0.375) when it is NOT the first cell:
         < 0.05 -> "first-cell artifact supported"; >= 0.10 -> "control active without first-cell
         position"; otherwise "inconclusive"
  F-2  dummy WBR(0:0.25) (the new first cell): >= 0.05 -> "inflation present in the first cell";
         < 0.05 -> "no first-cell inflation"   (supporting, not decisive)
  V-4' the B6b V-4 rule applied to this run (control WBR(0:0.375) < 0.05 required; median WBR over
         w = 0.625, 0.75, 0.875: >= 0.80 lines, <= 0.05 bytes/sectors, otherwise mixed)
  python3 wb_probe_firstcell_score.py cells | score --meta ... --kernel-csv ... --range-csv ...
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import wb_probe_b6b_score as b6b
import wb_probe_score as ws

DUMMY = (0.0, 0.25)
CELLS = "0:0.25,0:0.375,0:0.625,0:0.75,0:0.875,0:2"   # == B6b SECTOR_CELLS with the dummy prepended
CONTROL_W, TEST_W = 0.375, (0.625, 0.75, 0.875)
BAND = (0.90, 1.10)
F1_ART, F1_ACTIVE = 0.05, 0.10
F2_INFL = 0.05
V4_LINES, V4_BYTES = 0.80, 0.05
k = ws.key


def inb(x) -> bool:
    return x is not None and BAND[0] <= x <= BAND[1]


def score(sector: dict, correctness_failures: int, log=print) -> dict:
    v: dict = {}
    anchor = sector[k(0, 2)]
    v["F-0"] = correctness_failures == 0 and inb(anchor["WB"]) and inb(anchor["WBR"])
    log(f"F-0 correctness failures {correctness_failures}; sector 0:2 WB/WBR "
        f"{anchor['WB']:.3f}/{anchor['WBR']:.3f} -> {'PASS' if v['F-0'] else 'FAIL'}")
    ctrl = sector[k(0, CONTROL_W)]["WBR"]
    ctrl_wb = sector[k(0, CONTROL_W)]["WB"]
    if ctrl < F1_ART:
        f1 = "first-cell artifact supported"
    elif ctrl >= F1_ACTIVE:
        f1 = "control active without first-cell position"
    else:
        f1 = "inconclusive"
    v["F-1"], v["F-1 control WBR"], v["F-1 control WB"] = f1, round(ctrl, 4), round(ctrl_wb, 4)
    log(f"F-1 control WBR(0:{CONTROL_W}) as second cell = {ctrl:.3f} (per-kernel WB {ctrl_wb:.3f}) -> {f1}")
    dummy = sector[k(*DUMMY)]["WBR"]
    f2 = "inflation present in the first cell" if dummy >= F2_INFL else "no first-cell inflation"
    v["F-2"], v["F-2 dummy WBR"] = f2, round(dummy, 4)
    log(f"F-2 dummy WBR(0:{DUMMY[1]}) as first cell = {dummy:.3f} "
        f"(per-kernel WB {sector[k(*DUMMY)]['WB']:.3f}) -> {f2}")
    test = statistics.median(sector[k(0, w)]["WBR"] for w in TEST_W)
    if ctrl >= V4_BYTES:
        v4 = f"control failed (WBR(0:{CONTROL_W}) = {ctrl:.3f}); not interpreted"
    elif test >= V4_LINES:
        v4 = "counts lines"
    elif test <= V4_BYTES:
        v4 = "counts bytes/sectors"
    else:
        v4 = "mixed"
    v["V-4'"], v["V-4' median WBR"] = v4, round(test, 4)
    log(f"V-4' median WBR at w={TEST_W} = {test:.3f} (each: "
        + ", ".join(f"{sector[k(0, w)]['WBR']:.3f}" for w in TEST_W) + f") -> {v4}")
    log("verdict: " + json.dumps(v))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cells")
    s = sub.add_parser("score")
    s.add_argument("--meta", type=Path, required=True)
    s.add_argument("--kernel-csv", type=Path, nargs="+", required=True)
    s.add_argument("--range-csv", type=Path, nargs="+", required=True)
    s.add_argument("--meta-glob", required=True, help="all run metas, for correctness failures")
    s.add_argument("--json-out", type=Path)
    a = p.parse_args()
    if a.cmd == "cells":
        print(CELLS)
        return 0
    sector = b6b.load_set(a.meta, a.kernel_csv, a.range_csv)
    fails = sum(json.loads(Path(m).read_text())["correctness_failures"] for m in sorted(glob.glob(a.meta_glob)))
    ws.table(sector)
    v = score(sector, fails)
    if a.json_out:
        a.json_out.write_text(json.dumps({"cells": [sector[x] for x in sorted(sector)], "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
