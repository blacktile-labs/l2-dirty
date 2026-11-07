"""Score the B6b confirmatory write-back probe against the rules pre-registered in
research/HYPOTHESES.md, section "B6b pre-registration (V-0..V-5)".

Two cell sets, each measured 3x per-kernel (application replay) and 3x back-to-back (app-range):
  full    wb_probe --cells <FULL_CELLS>                 (whole 128-B lines written, as in B6)
  sector  wb_probe --one-sector --cells <SECTOR_CELLS>  (one 32-B sector written per 128-B line)
Per-cell quantities (WB, WBR, RB, RD, RDR, WARM) come from wb_probe_score.summarize; for the sector
set WB/WBR/RB are normalized by the bytes actually written (one sector per line).
The primary statistic is WBR (back-to-back launches, the way a real pipeline runs); B6 showed that
per-kernel serialization adds a noisy extra write-back near full occupancy (D-034).
Constants below must not change after data is seen.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import wb_probe_score as ws
from counter_prereg import load_ncu_raw

# ---- pre-registered constants ----------------------------------------------------------------
FULL_CELLS = "0:0.1875,0.5:0.1875,0.625:0.1875,0.6875:0.1875,0:0.3125,0.375:0.3125,0.5:0.3125,0.5625:0.3125," \
             "0:0.4375,0:0.5,0:0.5625,0:0.625,0:0.6875,2:0,0:2"
SECTOR_CELLS = "0:0.375,0:0.625,0:0.75,0:0.875,0:2"
BAND = (0.90, 1.10)
B6_WBR = {(0.0, 0.5): 0.190, (0.0, 0.625): 0.992}   # B6 run 20261105-042441, score.json
REPLICATION_TOL = 0.10
V2_ROWS = {0.1875: (0.5, 0.625, 0.6875), 0.3125: (0.375, 0.5, 0.5625)}
V2_FLAT, V2_RISE = 0.05, 0.10
V3_LOW_W, V3_HIGH_W = 0.4375, 0.5625
V3_LOW_MAX, V3_HIGH_MIN = 0.05, 0.80          # sharp step
V3_LOW_FAIL, V3_HIGH_FAIL = 0.10, 0.50        # clearly not sharp
V4_CONTROL_W, V4_TEST_W = 0.375, (0.625, 0.75, 0.875)
V4_LINES, V4_BYTES = 0.80, 0.05
ACTIVE = 0.10
V5_RESIDENT, V5_EVICTED = 0.25, 0.50

k = ws.key


def load_set(meta_path: Path, kernel_csvs: list[Path], range_csvs: list[Path]) -> dict:
    meta = json.loads(meta_path.read_text())
    n, steady = len(meta["cells"]), int(meta["steady"])
    kr = [ws.align_kernel(load_ncu_raw(p), n, steady, *ws.kernel_names(meta)) for p in kernel_csvs]
    rr = [ws.align_range(load_ncu_raw(p), n) for p in range_csvs]
    return ws.summarize(meta, kr, rr)


def inb(x: float | None) -> bool:
    return x is not None and BAND[0] <= x <= BAND[1]


def score(full: dict, sector: dict, log=print) -> dict:
    v: dict = {}
    # V-0 sanity
    a_r, a_w, s_w = full[k(2, 0)], full[k(0, 2)], sector[k(0, 2)]
    warm = [c["WARM"] for c in full.values() if c["WARM"] is not None]
    ok = (inb(a_r["RD"]) and inb(a_r["RDR"]) and inb(a_w["WB"]) and inb(a_w["WBR"])
          and inb(s_w["WB"]) and inb(s_w["WBR"]) and all(inb(x) for x in warm))
    v["V-0"] = ok
    log(f"V-0 sanity: full 2:0 RD/RDR {a_r['RD']:.3f}/{a_r['RDR']:.3f}, full 0:2 WB/WBR {a_w['WB']:.3f}/{a_w['WBR']:.3f}, "
        f"sector 0:2 WB/WBR {s_w['WB']:.3f}/{s_w['WBR']:.3f}, WARM [{min(warm):.3f}, {max(warm):.3f}] "
        f"-> {'PASS' if ok else 'FAIL (stop: do not interpret)'}")

    # V-1 replication of B6 at r = 0
    reps = {c: full[k(*c)]["WBR"] for c in B6_WBR}
    ok1 = all(abs(reps[c] - B6_WBR[c]) <= REPLICATION_TOL for c in B6_WBR)
    v["V-1 replicates B6"] = ok1
    log("V-1 " + ", ".join(f"WBR{c} {reps[c]:.3f} (B6 {B6_WBR[c]:.3f})" for c in B6_WBR)
        + f" -> {'PASS' if ok1 else 'FAIL'}")

    # V-2 primary: does raising the clean footprint at fixed dirty footprint change WBR?
    deltas, top = [], []
    for w, rs in V2_ROWS.items():
        base = full[k(0, w)]["WBR"]
        for r in rs:
            dl = full[k(r, w)]["WBR"] - base
            deltas.append(((r, w), dl))
            if abs(r + w - 0.875) < 1e-9:
                top.append(dl)
        kd = [full[k(r, w)]["WB"] - full[k(0, w)]["WB"] for r in rs]
        log(f"V-2 row w={w}: WBR(0)={base:.3f}; delta WBR at r={rs}: "
            + ", ".join(f"{d:+.3f}" for (rw, d) in deltas if rw[1] == w)
            + "  [per-kernel WB deltas, not decisive: " + ", ".join(f"{d:+.3f}" for d in kd) + "]")
    if all(abs(d) < V2_FLAT for _, d in deltas):
        v2 = "H-D confirmed (dirty footprint only)"
    elif all(d >= V2_RISE for d in top):
        v2 = "H-E supported (total footprint)"
    else:
        v2 = "inconclusive"
    v["V-2"] = v2
    log(f"V-2 -> {v2}")

    # V-3 sharpness of the step at r = 0
    lo, hi = full[k(0, V3_LOW_W)]["WBR"], full[k(0, V3_HIGH_W)]["WBR"]
    if lo < V3_LOW_MAX and hi >= V3_HIGH_MIN:
        v3 = "sharp step"
    elif lo >= V3_LOW_FAIL or hi < V3_HIGH_FAIL:
        v3 = "not sharp"
    else:
        v3 = "partial"
    v["V-3"] = v3
    log(f"V-3 WBR(0:{V3_LOW_W}) = {lo:.3f}, WBR(0:{V3_HIGH_W}) = {hi:.3f} -> {v3}")

    # V-4 lines or bytes?
    ctrl = sector[k(0, V4_CONTROL_W)]["WBR"]
    test = statistics.median(sector[k(0, w)]["WBR"] for w in V4_TEST_W)
    if ctrl >= V3_LOW_MAX:
        v4 = f"control failed (WBR(0:{V4_CONTROL_W}) = {ctrl:.3f}); not interpreted"
    elif test >= V4_LINES:
        v4 = "counts lines"
    elif test <= V4_BYTES:
        v4 = "counts bytes/sectors"
    else:
        v4 = "mixed"
    v["V-4"] = v4
    log(f"V-4 one-sector-per-line: control WBR {ctrl:.3f}; median WBR at w={V4_TEST_W} = {test:.3f} "
        f"(each: " + ", ".join(f"{sector[k(0, w)]['WBR']:.3f}" for w in V4_TEST_W) + f") -> {v4}")

    # V-5 residency of cleaned lines (per-kernel runs; readback is outside the ranges)
    act = [c for s in (full, sector) for c in s.values()
           if c["WB"] is not None and c["r"] + c["w"] <= 0.875 + 1e-9 and c["WB"] >= ACTIVE]
    if act:
        ratio = statistics.median(c["RB"] / c["WB"] for c in act)
        v5 = "resident" if ratio < V5_RESIDENT else "evicted" if ratio >= V5_EVICTED else "inconclusive"
        log(f"V-5 median RB/WB over {len(act)} active cells = {ratio:.3f} -> {v5}")
    else:
        v5 = "n/a (no active cell)"
    v["V-5"] = v5
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-meta", type=Path, required=True)
    ap.add_argument("--full-kernel-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--full-range-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--sector-meta", type=Path, required=True)
    ap.add_argument("--sector-kernel-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--sector-range-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    full = load_set(a.full_meta, a.full_kernel_csv, a.full_range_csv)
    sector = load_set(a.sector_meta, a.sector_kernel_csv, a.sector_range_csv)
    print("== full lines"); ws.table(full)
    print("== one sector per line (WB, WBR, RB normalized by written bytes)"); ws.table(sector)
    verdict = score(full, sector)
    print("verdict:", json.dumps(verdict))
    if a.json_out:
        a.json_out.write_text(json.dumps({"full": [full[x] for x in sorted(full)],
                                          "sector": [sector[x] for x in sorted(sector)],
                                          "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
