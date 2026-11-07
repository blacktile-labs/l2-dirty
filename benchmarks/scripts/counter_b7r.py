#!/usr/bin/env python3
"""B7r scorer: does ncu's one-kernel-at-a-time measurement cause B7's error near L2 capacity?
Rules Y-0..Y-3: research/HYPOTHESES.md "B7r pre-registration". Model and predictions are B7's,
unchanged: research/preregistration/b7_predictions.json (model v4, counter_b7.py).

Same harness (torch_opchain.py --mode counters), same sizes and launches, one variable changed:
  kernel mode  ncu --replay-mode application: per-launch bytes, profiler gaps between launches
  range mode   ncu --replay-mode app-range + --profiler-ranges: one range per SEGMENT (the chain
               calls after one flush), launches back to back behind a spin kernel
A segment s is (l2x, variant, mode): 4 per size, n(s) launches (1, 1, 9 or 3 calls' kernels).
  P(s) = sum of the locked v4 predictions over the segment's launches
  K(s) = median over kernel-mode runs of the sum of the segment's per-launch bytes
  G(s) = median over range-mode runs of (range bytes - median of that run's control ranges)
  normalised per launch: k = (K - P) / (B n), g = (G - P) / (B n), d = (G - K) / (B n),
  B = bytes of one [M, 4096] fp32 tensor.
Control ranges (spin kernel only, after a flush) measure the device's background DRAM traffic.

  python3 counter_b7r.py score --prereg research/preregistration/b7_predictions.json \
      --kernel-plan k1.json ... --kernel-csv k1.csv ... --range-plan g1.json ... --range-csv g1.csv ...
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import counter_b7 as b7
from counter_prereg import load_ncu_raw

EDGE = (0.25, 0.3125, 0.375, 0.5, 1.0)
FAR = (2.0, 3.0)
BG_READ_MAX = 16_000      # Y-0b: median control-range read bytes (~0.02 B at l2x 0.25)
BG_WRITE_MAX = 4_000      # Y-0b: max control-range write bytes
ENQ_FRAC = 0.8            # Y-0c (spin gate, B7r): max segment enqueue time < ENQ_FRAC x median control duration
                          # Y-0c (flag gate, B7r2): no range's gate timed out (gate_iters < gate_max_iter)
FAR_D = 0.05              # Y-0d: |d| per launch at far sizes, both directions
Y1_D, Y1_FRAC = 0.05, 0.80
RHO_ART, RHO_MODEL = 0.5, 0.8
WITHIN = 0.10             # descriptive: |g| or |k| <= 0.10 B per launch
RANGE = "range"


def segments(pred: list[dict]) -> list[dict]:
    """Consecutive non-flush launches after each flush, from the locked predictions."""
    out, cur = [], None
    for i, p in enumerate(pred):
        if p["kernel"] == "flush":
            cur = None
            continue
        if cur is None:
            cur = {"l2x": p["l2x"], "variant": p["variant"], "mode": p["mode"], "first": i, "n": 0, "B": p["B"],
                   "P": {"read": 0.0, "write": 0.0}}
            out.append(cur)
        cur["n"] += 1
        for d in ("read", "write"):
            cur["P"][d] += p[f"v4_{d}"]
    return out


def range_durations(path: Path) -> list[float]:
    rows = list(csv.reader(open(path)))
    h = rows[0]
    k, t = h.index("Kernel Name"), h.index("gpu__time_duration.sum")
    return [float(r[t]) for r in rows[2:] if r and r[0].strip() and r[k] == RANGE]


def check_range_meta(segs: list[dict], ranges: list[dict]) -> None:
    sm = [r for r in ranges if r["kind"] == "segment"]
    if len(sm) != len(segs):
        raise ValueError(f"{len(sm)} segment ranges, {len(segs)} segments in the predictions")
    for s, r in zip(segs, sm):
        if (r["l2x"], r["variant"], r["mode"], r["first_launch"], r["n_launches"]) != \
                (s["l2x"], s["variant"], s["mode"], s["first"], s["n"]):
            raise ValueError(f"range {r} does not match segment {s}")


def score(segs: list[dict], kruns: list[list[dict]], rruns: list[list[dict]], ranges: list[dict],
          ctl_dur: list[float], valid: dict, log=print, gate_max: int | None = None) -> dict:
    """kruns: per kernel-mode run, aligned launches (dicts with read/write).
    rruns: per range-mode run, range rows in issue order (control and segment ranges, as `ranges`)."""
    v: dict = {}
    ci = [i for i, r in enumerate(ranges) if r["kind"] == "control"]
    si = [i for i, r in enumerate(ranges) if r["kind"] == "segment"]
    bg_r = [run[i]["read"] for run in rruns for i in ci]
    bg_w = [run[i]["write"] for run in rruns for i in ci]
    cells = []
    for j, s in enumerate(segs):
        a, n, B = s["first"], s["n"], s["B"]
        c = {"l2x": s["l2x"], "variant": s["variant"], "mode": s["mode"], "n": n}
        for d in ("read", "write"):
            K = statistics.median(sum(run[x][d] for x in range(a, a + n)) for run in kruns)
            G = statistics.median(run[si[j]][d] - statistics.median(run[i][d] for i in ci) for run in rruns)
            P = s["P"][d]
            c[d] = {"K": K / B, "G": G / B, "P": P / B, "k": (K - P) / (B * n), "g": (G - P) / (B * n),
                    "d": (G - K) / (B * n)}
        cells.append(c)
    enq = max(ranges[i]["enqueue_us"] for i in si)
    cd = statistics.median(ctl_dur) / 1e3
    if gate_max is None:
        back, back_txt = enq < ENQ_FRAC * cd, f"max enqueue {enq:.0f} us vs spin {cd:.0f} us (< {ENQ_FRAC}x)"
    else:
        gi = max(r["gate_iters"] for r in ranges)
        back, back_txt = gi < gate_max, f"flag gate max iterations {gi} (timeout at {gate_max}), max enqueue {enq:.0f} us"

    far = [c[d]["d"] for c in cells if c["l2x"] in FAR for d in ("read", "write")]
    y0 = {"a plans+correctness": valid["plans"] and valid["correctness"],
          "b background": statistics.median(bg_r) <= BG_READ_MAX and max(bg_w) <= BG_WRITE_MAX,
          "c back-to-back": back,
          "d far sizes agree": all(abs(x) <= FAR_D for x in far)}
    v["Y-0"] = all(y0.values())
    v["Y-0 parts"] = y0
    log(f"Y-0 a plans identical + correctness: {y0['a plans+correctness']}; b background: median control read "
        f"{statistics.median(bg_r):.0f} B (<= {BG_READ_MAX}), max control write {max(bg_w):.0f} B (<= {BG_WRITE_MAX}); "
        f"c {back_txt}; d far |d| max "
        f"{max(abs(x) for x in far):.3f} (<= {FAR_D}) -> {'PASS' if v['Y-0'] else 'FAIL (stop)'}")
    edge = [c for c in cells if c["l2x"] in EDGE]
    rd = [abs(c["read"]["d"]) <= Y1_D for c in edge]
    share = sum(rd) / len(rd)
    v["Y-1"] = "reads unchanged by replay mode" if share >= Y1_FRAC else "reads change with replay mode"
    v["Y-1 share"] = round(share, 3)
    log(f"Y-1 edge read segments with |G-K| <= {Y1_D} B/launch: {share * 100:.0f}% (need >= {Y1_FRAC * 100:.0f}%) -> {v['Y-1']}")

    def rho(dirs):
        g = statistics.fmean(abs(c[d]["g"]) for c in edge for d in dirs)
        k = statistics.fmean(abs(c[d]["k"]) for c in edge for d in dirs)
        return g / k, g, k

    def verdict(r, art, model, mid):
        return art if r <= RHO_ART else model if r >= RHO_MODEL else mid

    rw, gw, kw = rho(("write",))
    v["Y-2"] = verdict(rw, "write error mostly a measurement artifact", "write error not a measurement artifact",
                       "write error partly a measurement artifact")
    v["Y-2 rho"] = round(rw, 3)
    log(f"Y-2 edge writes: mean |g| {gw:.3f} vs mean |k| {kw:.3f} B/launch, rho {rw:.2f} -> {v['Y-2']} "
        f"(mean d {statistics.fmean(c['write']['d'] for c in edge):+.3f})")
    ra, ga, ka = rho(("read", "write"))
    v["Y-3"] = verdict(ra, "MEASUREMENT", "MODEL", "PARTIAL")
    v["Y-3 rho"] = round(ra, 3)
    log(f"Y-3 edge reads+writes: mean |g| {ga:.3f} vs mean |k| {ka:.3f} B/launch, rho {ra:.2f} -> {v['Y-3']}")
    allc = [c[d] for c in cells for d in ("read", "write")]
    v["within range"] = round(sum(abs(x["g"]) <= WITHIN for x in allc) / len(allc), 3)
    v["within kernel"] = round(sum(abs(x["k"]) <= WITHIN for x in allc) / len(allc), 3)
    log(f"descriptive: segment-dirs within {WITHIN} B/launch of v4: range {v['within range'] * 100:.0f}%, "
        f"kernel {v['within kernel'] * 100:.0f}% (of {len(allc)})")
    v["_cells"] = cells
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--prereg", type=Path, required=True)
    for k in ("kernel-plan", "kernel-csv", "range-plan", "range-csv"):
        s.add_argument(f"--{k}", type=Path, nargs="+", required=True)
    s.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    rec = json.loads(a.prereg.read_text())
    pred = rec["launches"]
    segs = segments(pred)
    strip = lambda ps: [{k: p[k] for k in p if k != "i"} for p in ps]
    plans = [json.loads(p.read_text()) for p in a.kernel_plan + a.range_plan]
    same = [strip(b7.predictions(pl)) == strip(pred) for pl in plans]
    corr = [pl["meta"].get("correctness_failures") == 0 for pl in plans]
    print(f"plans reproduce the locked predictions: {same}; correctness ok: {corr}")
    kruns = [b7.align(load_ncu_raw(c), pl["launches"]) for c, pl in zip(a.kernel_csv, plans[:len(a.kernel_plan)])]
    rplans = plans[len(a.kernel_plan):]
    ranges = rplans[0]["meta"]["ranges"]
    layout = lambda rs: [{k: r[k] for k in r if k not in ("enqueue_us", "gate_iters")} for r in rs]  # per-run values
    if any(layout(pl["meta"]["ranges"]) != layout(ranges) for pl in rplans[1:]):
        raise ValueError("range layout differs between range-mode runs")
    check_range_meta(segs, ranges)
    all_ranges = [dict(r, **{k: max(pl["meta"]["ranges"][i].get(k, 0) for pl in rplans)
                             for k in ("enqueue_us", "gate_iters") if k in r})
                  for i, r in enumerate(ranges)]
    gate_max = rplans[0]["meta"].get("gate_max_iter") if rplans[0]["meta"].get("range_gate") == "flag" else None
    rruns = []
    for c in a.range_csv:
        rows = [r for r in load_ncu_raw(c) if r["kernel"] == RANGE]
        if len(rows) != len(ranges):
            raise ValueError(f"{c}: {len(rows)} ranges, meta has {len(ranges)}")
        rruns.append(rows)
    ci = [i for i, r in enumerate(ranges) if r["kind"] == "control"]
    ctl_dur = [range_durations(c)[i] for c in a.range_csv for i in ci]
    v = score(segs, kruns, rruns, all_ranges, ctl_dur, {"plans": all(same), "correctness": all(corr)},
              gate_max=gate_max)
    cells = v.pop("_cells")
    print(f"\n{'l2x':>7} {'variant':8} {'mode':8} n  dir     P      K      G   k/launch g/launch d/launch")
    for c in cells:
        for d in ("read", "write"):
            x = c[d]
            print(f"{c['l2x']:7g} {c['variant']:8} {c['mode']:8} {c['n']} {d:5} {x['P']:6.3f} {x['K']:6.3f} {x['G']:6.3f} "
                  f"{x['k']:+8.3f} {x['g']:+8.3f} {x['d']:+8.3f}{'  edge' if c['l2x'] in EDGE else ''}")
    print("verdict:", json.dumps(v))
    if a.json_out:
        a.json_out.write_text(json.dumps({"verdict": v, "cells": cells}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
