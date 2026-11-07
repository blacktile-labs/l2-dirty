#!/usr/bin/env python3
"""B9 scorer (idle_wb_probe.py under ncu --replay-mode app-range). Rules I-0..I-3:
research/HYPOTHESES.md "B9 pre-registration".

Per range (bytes normalised by the bytes the write kernel stored, w x L2):
  A = DRAM writes in range A (write kernel + idle T),  B = DRAM writes in range B (the flush after it)
Primary quantity, robust to a constant extra in the flush range and to whether a range counts
traffic while no kernel runs:
  IWB(w, idle, T) = (B(w, host, 0) - B(w, idle, T)) / bytes   = fraction cleaned before the flush
Medians over all repetitions of all runs.

  python3 idle_wb_score.py score --meta m1.json m2.json m3.json --range-csv r1.csv r2.csv r3.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from counter_prereg import load_ncu_raw

CTL_W_MAX = 4_000          # I-0b: max control-range writes, bytes
BASE_LO, BASE_HI = 0.95, 1.15   # I-0c: A + B at T = 0 for every w
I1_A = 0.05                # I-1: A at T = 0 for every w
I2_YES, I2_NO, I2_T = 0.5, 0.1, 10.0
RANGE = "range"


def cells_from(metas: list[dict], runs: list[list[dict]]) -> dict:
    """{(w, idle, T): {"A": [...], "B": [...], "Bbytes": [...], "bytes": n}} over all reps and runs."""
    out: dict = {}
    for meta, rows in zip(metas, runs):
        for m, r in zip(meta["ranges"], rows):
            if m["kind"] not in ("A", "B"):
                continue
            c = out.setdefault((m["w"], m["idle"], m["T_ms"]), {"A": [], "B": [], "Bbytes": [], "bytes": m["bytes"]})
            c[m["kind"]].append(r["write"] / m["bytes"])
            if m["kind"] == "B":
                c["Bbytes"].append(r["write"])
    return out


def score(metas: list[dict], runs: list[list[dict]], log=print) -> dict:
    v: dict = {}
    cells = cells_from(metas, runs)
    ctl = [r["write"] for meta, rows in zip(metas, runs) for m, r in zip(meta["ranges"], rows) if m["kind"] == "control"]
    ws = sorted({k[0] for k in cells})
    med = statistics.median
    base = {w: med(cells[(w, "host", 0.0)]["Bbytes"]) for w in ws}
    summary = {}
    for k, c in sorted(cells.items()):
        summary[k] = {"A": med(c["A"]), "B": med(c["B"]), "IWB": (base[k[0]] - med(c["Bbytes"])) / c["bytes"],
                      "n": len(c["A"])}
    i0 = {"a correctness": all(m["correctness_failures"] == 0 for m in metas),
          "b control writes": max(ctl) <= CTL_W_MAX,
          "c baseline A+B": all(BASE_LO <= summary[(w, "host", 0.0)]["A"] + summary[(w, "host", 0.0)]["B"] <= BASE_HI
                                for w in ws)}
    v["I-0"] = all(i0.values())
    v["I-0 parts"] = i0
    log(f"I-0 correctness {i0['a correctness']}; control writes max {max(ctl):.0f} B (<= {CTL_W_MAX}); baseline A+B "
        + ", ".join(f"w {w:g}: {summary[(w, 'host', 0.0)]['A'] + summary[(w, 'host', 0.0)]['B']:.3f}" for w in ws)
        + f" (in [{BASE_LO}, {BASE_HI}]) -> {'PASS' if v['I-0'] else 'FAIL (stop)'}")
    a0 = {w: summary[(w, "host", 0.0)]["A"] for w in ws}
    v["I-1"] = "no write-back while writing below the cap" if all(x <= I1_A for x in a0.values()) else \
        "write-back while writing below the cap"
    log(f"I-1 A at T = 0: " + ", ".join(f"w {w:g}: {x:.3f}" for w, x in a0.items()) + f" (<= {I1_A}) -> {v['I-1']}")
    iwb = {w: summary[(w, "host", I2_T)]["IWB"] for w in ws}
    v["I-2"] = ("idle write-back exists" if all(x >= I2_YES for x in iwb.values()) else
                "no idle write-back" if all(x <= I2_NO for x in iwb.values()) else "partial or size-dependent")
    v["I-2 IWB"] = {f"{w:g}": round(x, 3) for w, x in iwb.items()}
    log(f"I-2 IWB after {I2_T:g} ms host idle: " + ", ".join(f"w {w:g}: {x:.3f}" for w, x in iwb.items())
        + f" (all >= {I2_YES}: exists; all <= {I2_NO}: none) -> {v['I-2']}")
    log("I-3 (descriptive)   w    idle   T ms      A      B    IWB   n")
    for k, s in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        log(f"              {k[0]:7g} {k[1]:5} {k[2]:5g} {s['A']:6.3f} {s['B']:6.3f} {s['IWB']:6.3f} {s['n']:3}")
    v["_summary"] = {f"{k[0]:g}|{k[1]}|{k[2]:g}": s for k, s in summary.items()}
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--meta", type=Path, nargs="+", required=True)
    s.add_argument("--range-csv", type=Path, nargs="+", required=True)
    s.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    metas = [json.loads(p.read_text()) for p in a.meta]
    runs = []
    for meta, p in zip(metas, a.range_csv):
        rows = [r for r in load_ncu_raw(p) if r["kernel"] == RANGE]
        if len(rows) != len(meta["ranges"]):
            raise ValueError(f"{p}: {len(rows)} ranges, meta has {len(meta['ranges'])}")
        runs.append(rows)
    v = score(metas, runs)
    summ = v.pop("_summary")
    print("verdict:", json.dumps(v))
    if a.json_out:
        a.json_out.write_text(json.dumps({"verdict": v, "summary": summ}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
