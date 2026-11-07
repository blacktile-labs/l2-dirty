#!/usr/bin/env python3
"""B9b scorer (queued_wb_probe.py). Rules Q-0..Q-3: research/HYPOTHESES.md "B9b pre-registration".

Normalised by the bytes the write kernel stored (w x L2); medians over all repetitions of all runs:
  kernel mode : W (write kernel), F (the flush after it), S_k = median of W + F per pair (both pairs)
  range mode  : A, B (split ranges), S_r = median of A + B;  S_c = median of C (write + flush in one range)
  model v4    : A_v4, B_v4 from cache_sim_v4.DirtyCapLRU replaying flush -> write -> flush

  python3 queued_wb_score.py predict
  python3 queued_wb_score.py score --kernel-meta k*.json --kernel-csv k*.csv --range-meta r*.json --range-csv r*.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from cache_sim_v4 import DirtyCapLRU
from counter_prereg import load_ncu_raw

LINE = 128
CTL_W_MAX = 4_000
BASE_W, BASE_LO, BASE_HI = 0.375, 0.95, 1.15
LOSS_YES, LOSS_NO = 0.05, 0.02
V4_TOL = 0.10
KERNELS = ("write_lines", "flush_l2_read")


def v4_predict(w: float, l2: int, flush_bytes: int) -> dict:
    c = DirtyCapLRU(l2 // LINE)
    F, D = 1 << 30, 0
    n = int(w * l2) // LINE
    for j in range(flush_bytes // LINE):
        c.access(F + j, False)
    c.reset_counters()
    for j in range(n):
        c.access(D + j, True)
    a = c.writebacks / n
    c.reset_counters()
    for j in range(flush_bytes // LINE):
        c.access(F + j, False)
    return {"A": a, "B": c.writebacks / n}


def kernel_pairs(meta: dict, rows: list[dict]) -> list[dict]:
    sel = [r for r in rows if r["kernel"] in KERNELS]
    ev = meta["events"]
    if len(sel) != 2 + 3 * len(ev):
        raise ValueError(f"{len(sel)} write/flush launches, expected {2 + 3 * len(ev)}")
    out = []
    for j, m in enumerate(ev):
        pre, w, f = sel[2 + 3 * j: 5 + 3 * j]
        if (pre["kernel"], w["kernel"], f["kernel"]) != ("flush_l2_read", "write_lines", "flush_l2_read"):
            raise ValueError(f"pair {j}: {pre['kernel']}, {w['kernel']}, {f['kernel']}")
        out.append({"w": m["w"], "W": w["write"] / m["bytes"], "F": f["write"] / m["bytes"]})
    return out


def score(kpairs: list[dict], rranges: list[tuple[dict, dict]], fails: int, preds: dict, log=print) -> dict:
    """kpairs: kernel-mode pairs over all runs; rranges: (event, row) over all range runs."""
    med = statistics.median
    ws = sorted({p["w"] for p in kpairs})
    ctl = [r["write"] for e, r in rranges if e["kind"] == "control"]
    s = {}
    for w in ws:
        kp = [p for p in kpairs if p["w"] == w]
        rr = {k: [r["write"] / e["bytes"] for e, r in rranges if e.get("w") == w and e["kind"] == k] for k in "ABC"}
        s[w] = {"W": med(p["W"] for p in kp), "F": med(p["F"] for p in kp), "S_k": med(p["W"] + p["F"] for p in kp),
                "A": med(rr["A"]), "B": med(rr["B"]), "S_r": med(a + b for a, b in zip(rr["A"], rr["B"])),
                "S_c": med(rr["C"]), "n_k": len(kp), "n_r": len(rr["C"]), **{f"{k}_v4": preds[w][k] for k in "AB"}}
    base = s[BASE_W]
    v: dict = {}
    i0 = {"a correctness": fails == 0, "b control writes": max(ctl) <= CTL_W_MAX,
          "c below-cap sums": all(BASE_LO <= base[k] <= BASE_HI for k in ("S_k", "S_r", "S_c"))}
    v["Q-0"], v["Q-0 parts"] = all(i0.values()), i0
    log(f"Q-0 correctness failures {fails}; control writes max {max(ctl):.0f} B (<= {CTL_W_MAX}); w {BASE_W}: "
        f"S_k {base['S_k']:.3f} S_r {base['S_r']:.3f} S_c {base['S_c']:.3f} (in [{BASE_LO}, {BASE_HI}]) "
        f"-> {'PASS' if v['Q-0'] else 'FAIL (stop)'}")
    above = [w for w in ws if w > BASE_W]

    def three(vals, yes, no):
        return "yes" if all(x >= yes for x in vals) else "no" if all(x <= no for x in vals) else "mixed"

    loss = {w: s[w]["S_c"] - s[w]["S_k"] for w in above}
    q1 = three(loss.values(), LOSS_YES, LOSS_NO)
    v["Q-1"] = {"yes": "per-kernel mode loses write-backs above the cap", "no": "no loss",
                "mixed": "mixed"}[q1]
    v["Q-1 loss"] = {f"{w:g}": round(x, 3) for w, x in loss.items()}
    log("Q-1 S_c - S_k: " + ", ".join(f"w {w:g}: {x:+.3f}" for w, x in loss.items())
        + f" (all >= {LOSS_YES}: loss; all <= {LOSS_NO}: none) -> {v['Q-1']}")
    da = {w: s[w]["A"] - s[w]["W"] for w in above}
    q2 = three(da.values(), LOSS_YES, LOSS_NO)
    v["Q-2"] = {"yes": "the kernel window misses part of its own write-backs", "no": "the kernel window counts them",
                "mixed": "mixed"}[q2]
    log("Q-2 A - W: " + ", ".join(f"w {w:g}: {x:+.3f}" for w, x in da.items()) + f" -> {v['Q-2']}")
    ok = all(abs(s[w]["A"] - s[w]["A_v4"]) <= V4_TOL for w in above)
    v["Q-3"] = "v4 predicts one write kernel's write-backs" if ok else "v4 does not"
    log("Q-3 A vs v4: " + ", ".join(f"w {w:g}: {s[w]['A']:.3f} vs {s[w]['A_v4']:.3f}" for w in above)
        + f" (within {V4_TOL}) -> {v['Q-3']}")
    log(f"{'w':>7}      W      F    S_k |      A      B    S_r    S_c |   A_v4   B_v4")
    for w in ws:
        x = s[w]
        log(f"{w:7g} {x['W']:6.3f} {x['F']:6.3f} {x['S_k']:6.3f} | {x['A']:6.3f} {x['B']:6.3f} {x['S_r']:6.3f} "
            f"{x['S_c']:6.3f} | {x['A_v4']:6.3f} {x['B_v4']:6.3f}")
    v["_summary"] = {f"{w:g}": x for w, x in s.items()}
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--l2-bytes", type=int, default=3145728)
    p.add_argument("--flush-bytes", type=int, default=16 << 20)
    s = sub.add_parser("score")
    for k in ("kernel-meta", "kernel-csv", "range-meta", "range-csv"):
        s.add_argument(f"--{k}", type=Path, nargs="+", required=True)
    s.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    if a.cmd == "predict":
        for w in (0.375, 0.75, 1.0, 1.5):
            print(f"w {w:g}: {v4_predict(w, a.l2_bytes, a.flush_bytes)}")
        return 0
    km = [json.loads(x.read_text()) for x in a.kernel_meta]
    rm = [json.loads(x.read_text()) for x in a.range_meta]
    kpairs = [p for m, c in zip(km, a.kernel_csv) for p in kernel_pairs(m, load_ncu_raw(c))]
    rranges = []
    for m, c in zip(rm, a.range_csv):
        rows = [r for r in load_ncu_raw(c) if r["kernel"] == "range"]
        if len(rows) != len(m["events"]):
            raise ValueError(f"{c}: {len(rows)} ranges, meta has {len(m['events'])}")
        rranges += list(zip(m["events"], rows))
    l2, fb = km[0]["l2_bytes"], km[0]["flush_bytes"]
    preds = {w: v4_predict(w, l2, fb) for w in sorted({p["w"] for p in kpairs})}
    fails = sum(m["correctness_failures"] for m in km + rm)
    v = score(kpairs, rranges, fails, preds)
    summ = v.pop("_summary")
    print("verdict:", json.dumps(v))
    if a.json_out:
        a.json_out.write_text(json.dumps({"verdict": v, "summary": summ}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
