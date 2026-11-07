"""Score the B6 write-back probe (wb_probe.cu) against the rules pre-registered in
research/HYPOTHESES.md, section "B6 write-back probe pre-registration (W-0..W-5)".

Inputs: ncu raw CSV exports (`ncu --import <rep> --csv --page raw --print-units base`)
  --kernel-csv  one per run of `ncu --replay-mode application`  (per-launch counters)
  --range-csv   one per run of `ncu --replay-mode app-range`    (one row per cell: steady 2..S)
  --meta        the binary's JSON meta of any one run (cells and S are identical in every run)

Quantities, all as fractions of the bytes the launch touches (B_r = r-buffer bytes, B_w = w-buffer):
  WB    median over runs x steady launches 2..S of dram__bytes_write / B_w
  RD    median over runs x steady launches 2..S of dram__bytes_read  / B_r
  RB    median over runs of readback dram__bytes_read / B_w      (is D still resident?)
  WARM  median over runs of warm-launch dram__bytes_read / B_r   (sanity: C comes from DRAM)
  WBR   median over runs of (range dram__bytes_write / (S-1)) / B_w   (no per-kernel gaps)
The rules (thresholds, cells) are constants below and must not change after data is seen.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from counter_prereg import load_ncu_raw

FLUSH, RW, READBACK, RANGE = "flush_l2_read_kernel", "rw_kernel", "readback_kernel", "range"

# ---- pre-registered constants ----------------------------------------------------------------
ACTIVE, INACTIVE = 0.10, 0.05          # WB >= ACTIVE: write-back active; WB < INACTIVE: inactive
ANCHOR_LO, ANCHOR_HI = 0.90, 1.10      # W-0 sanity band
IN_CACHE_MAX = 0.875                   # cells with r + w <= this are "in cache" for a write-back LRU
W2_ROW_W = 0.125                       # W-2: dirty footprint held fixed ...
W2_CELLS = (0.625, 0.75)               # ... while the clean footprint is raised to these r
W2_DELTA_ACTIVE, W2_DELTA_FLAT = 0.10, 0.05
W3_ROWS = (0.125, 0.25, 0.375, 0.5)
W3_FLAT, W3_RISE = 0.15, 0.20
W4_RESIDENT, W4_EVICTED = 0.25, 0.50
W5_NO_GAP, W5_GAP = 0.75, 0.50


def key(r: float, w: float) -> tuple[float, float]:
    return (round(r, 6), round(w, 6))


def kernel_names(meta: dict) -> tuple[str, str]:
    """(rw, readback) kernel names; B6b's --one-sector runs use the 1s variants."""
    return ("rw1s_kernel", "readback1s_kernel") if meta.get("one_sector") else (RW, READBACK)


def written_bytes(meta: dict, nd: int) -> int:
    """Bytes of D a launch writes: all nd words, or one 32-B sector (8 words) per 32-word line."""
    return nd if meta.get("one_sector") else 4 * nd


def align_kernel(rows: list[dict], n_cells: int, steady: int, rw: str = RW,
                 readback: str = READBACK) -> list[dict]:
    """Per cell: {'warm', 'steady': [...], 'readback'} launch dicts. A cell's sequence starts at
    the first flush after the previous cell and must be exactly rw x (1+S), readback."""
    plan = [rw] * (1 + steady) + [readback]
    out, pos = [], 0
    for c in range(n_cells):
        while pos < len(rows) and rows[pos]["kernel"] != FLUSH:
            pos += 1
        seg = rows[pos + 1:pos + 1 + len(plan)]
        got = [m["kernel"] for m in seg]
        if pos >= len(rows) or got != plan:
            raise ValueError(f"cell #{c}: launch sequence after flush is {got} (expected {plan})")
        out.append({"warm": seg[0], "steady": seg[1:1 + steady], "readback": seg[-1]})
        pos += 1 + len(plan)
    return out


def align_range(rows: list[dict], n_cells: int) -> list[dict]:
    rng = [m for m in rows if m["kernel"] == RANGE]
    if len(rng) != n_cells:
        raise ValueError(f"{len(rng)} ranges in the export, {n_cells} cells in the meta")
    return rng


def med(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def summarize(meta: dict, kernel_runs: list[list[dict]], range_runs: list[list[dict]]) -> dict:
    steady = int(meta["steady"])
    cells = meta["cells"]
    out = {}
    for i, c in enumerate(cells):
        br, bw = 4 * c["nc"], written_bytes(meta, c["nd"])
        st = [s for run in kernel_runs for s in run[i]["steady"][1:]]  # steady launches 2..S
        d = {"r": c["r"], "w": c["w"], "n_steady_samples": len(st)}
        d["WB"] = med([s["write"] / bw for s in st]) if bw else None
        d["RD"] = med([s["read"] / br for s in st]) if br else None
        d["RB"] = med([run[i]["readback"]["read"] / bw for run in kernel_runs]) if bw else None
        d["WARM"] = med([run[i]["warm"]["read"] / br for run in kernel_runs]) if br else None
        d["WBR"] = (med([run[i]["write"] / (steady - 1) / bw for run in range_runs])
                    if bw and range_runs else None)
        d["RDR"] = (med([run[i]["read"] / (steady - 1) / br for run in range_runs])
                    if br and range_runs else None)
        out[key(c["r"], c["w"])] = d
    return out


def state(wb: float | None) -> str:
    if wb is None:
        return "n/a"
    return "active" if wb >= ACTIVE else ("inactive" if wb < INACTIVE else "ambiguous")


def score(cells: dict, log=print) -> dict:
    verdict: dict = {}
    in_cache = {k: v for k, v in cells.items() if v["r"] + v["w"] <= IN_CACHE_MAX + 1e-9}

    # W-0 sanity
    a_r, a_w = cells.get(key(2, 0)), cells.get(key(0, 2))
    warm = [v["WARM"] for v in cells.values() if v["WARM"] is not None]
    w0 = (a_r is not None and a_w is not None
          and ANCHOR_LO <= a_r["RD"] <= ANCHOR_HI and ANCHOR_LO <= a_w["WB"] <= ANCHOR_HI
          and all(ANCHOR_LO <= x <= ANCHOR_HI for x in warm))
    verdict["W-0"] = w0
    log(f"W-0 sanity: anchor 2:0 RD {a_r and a_r['RD']:.3f}, anchor 0:2 WB {a_w and a_w['WB']:.3f}, "
        f"WARM range [{min(warm):.3f}, {max(warm):.3f}] -> {'PASS' if w0 else 'FAIL (stop: do not interpret)'}")

    # W-1 write-back LRU
    act = sorted(k for k, v in in_cache.items() if v["WB"] is not None and v["WB"] >= ACTIVE)
    rd_hi = sorted(k for k, v in in_cache.items() if v["RD"] is not None and v["RD"] >= ACTIVE)
    verdict["W-1 LRU rejected"] = bool(act or rd_hi)
    log(f"W-1 write-back LRU: {len(act)} in-cache cells with WB >= {ACTIVE}, {len(rd_hi)} with RD >= {ACTIVE} "
        f"-> {'REJECTED' if act or rd_hi else 'not rejected'}")

    # W-2 dirty-only (H-D) vs total-footprint (H-E): raise r at fixed w
    base = cells[key(0, W2_ROW_W)]["WB"]
    deltas = [cells[key(r, W2_ROW_W)]["WB"] - base for r in W2_CELLS]
    if all(dl >= W2_DELTA_ACTIVE for dl in deltas):
        w2 = "H-E (total footprint) supported over H-D"
    elif all(abs(dl) < W2_DELTA_FLAT for dl in deltas):
        w2 = "H-D (dirty footprint only) supported over H-E"
    else:
        w2 = "inconclusive"
    verdict["W-2"] = w2
    log(f"W-2 WB(0:{W2_ROW_W}) = {base:.3f}; deltas at r = {W2_CELLS}: "
        + ", ".join(f"{dl:+.3f}" for dl in deltas) + f" -> {w2}")

    # W-3 row structure (secondary)
    spreads, rises = [], []
    for w in W3_ROWS:
        row = sorted((v["r"], v["WB"]) for v in in_cache.values() if abs(v["w"] - w) < 1e-9)
        wbs = [x for _, x in row]
        spreads.append(max(wbs) - min(wbs))
        rises.append(wbs[-1] - wbs[0])
    if all(s <= W3_FLAT for s in spreads):
        w3 = "flat rows (H-D)"
    elif sum(rz >= W3_RISE for rz in rises) >= 2:
        w3 = "rising rows (H-E)"
    else:
        w3 = "neither"
    verdict["W-3"] = w3
    log("W-3 rows w=" + ", ".join(f"{w:g}: spread {s:.3f} rise {rz:+.3f}" for w, s, rz in zip(W3_ROWS, spreads, rises))
        + f" -> {w3}")

    # W-4 residency of cleaned lines, W-5 per-kernel gaps (over active in-cache cells)
    if act:
        ratio_rb = med([in_cache[k]["RB"] / in_cache[k]["WB"] for k in act])
        w4 = ("cleaned, stays resident" if ratio_rb < W4_RESIDENT
              else "evicted" if ratio_rb >= W4_EVICTED else "inconclusive")
        log(f"W-4 median RB/WB over {len(act)} active cells = {ratio_rb:.3f} -> {w4}")
        rr = [in_cache[k]["WBR"] / in_cache[k]["WB"] for k in act if in_cache[k]["WBR"] is not None]
        if rr:
            ratio_r = med(rr)
            w5 = ("not gap-driven" if ratio_r >= W5_NO_GAP
                  else "gap-driven" if ratio_r <= W5_GAP else "inconclusive")
            log(f"W-5 median WBR/WB over {len(rr)} active cells = {ratio_r:.3f} -> {w5}")
        else:
            w5 = "n/a (no range data)"
    else:
        w4 = w5 = "n/a (no active cell)"
    verdict["W-4"], verdict["W-5"] = w4, w5
    return verdict


def table(cells: dict, log=print) -> None:
    log(f"{'r':>6} {'w':>6} {'r+w':>6} {'WB':>7} {'WBR':>7} {'RB':>7} {'RD':>7} {'RDR':>7} {'WARM':>6}  state")
    f = lambda x: f"{x:7.3f}" if x is not None else "      -"
    for k in sorted(cells):
        v = cells[k]
        log(f"{v['r']:6g} {v['w']:6g} {v['r'] + v['w']:6g} {f(v['WB'])} {f(v['WBR'])} {f(v['RB'])} "
            f"{f(v['RD'])} {f(v['RDR'])} {v['WARM'] if v['WARM'] is None else round(v['WARM'], 3)!s:>6}  {state(v['WB'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--range-csv", type=Path, nargs="*", default=[])
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--table-only", action="store_true", help="smoke runs: print the table, score nothing")
    args = ap.parse_args()
    meta = json.loads(args.meta.read_text())
    n, steady = len(meta["cells"]), int(meta["steady"])
    kernel_runs = [align_kernel(load_ncu_raw(p), n, steady, *kernel_names(meta)) for p in args.kernel_csv]
    range_runs = [align_range(load_ncu_raw(p), n) for p in args.range_csv]
    cells = summarize(meta, kernel_runs, range_runs)
    table(cells)
    if args.table_only:
        return 0
    verdict = score(cells)
    print("verdict:", json.dumps(verdict))
    if args.json_out:
        args.json_out.write_text(json.dumps({"cells": [cells[k] for k in sorted(cells)], "verdict": verdict},
                                            indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
