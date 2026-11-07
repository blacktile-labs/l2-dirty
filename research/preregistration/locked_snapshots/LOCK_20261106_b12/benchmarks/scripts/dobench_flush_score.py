"""B12 scorer: do_bench's write flush vs a read flush before an LLM decode GEMV.
Rules G-0..G-2 pre-registered in research/HYPOTHESES.md, section "B12 pre-registration", before any
B12 measurement on any GPU. Inputs: the probe's range-mode metas + ncu raw CSVs (app-range: one row
per range, in event order) and its timing-mode metas.

  X(w)   = [median writes of the GEMV range after a "zero" flush - median after a "read" flush] / L2
  G-0  sanity: correctness 0 in every run; control ranges write <= 64 KiB; for every w after a read
       flush the GEMV writes <= 0.05 L2 and reads 0.9-1.2 x its weight bytes
  G-1  (primary) |X(w) - R_gpu| <= 0.15 for every w -> "residue = B9b-predicted dirty residue";
       for no w -> "not"; otherwise "mixed". R_gpu = 1.5 x (1 - A(1.5)) from that GPU's B9b run
       (the dirty bytes a single long write leaves in the L2); v4's 0.5 is reported beside it
  G-1b max X - min X <= 0.10 -> "residue independent of the GEMV size" (the GEMV evicts it all)
  G-2  Delta t(w) = median t(zero) - median t(read) (timing mode); predicted from bytes at the clean
       run's DRAM rate: Delta t_pred = X L2 / ((R_read + W_read) / t(read)).
       Delta t > 0 for every w AND |Delta t - Delta t_pred| <= 0.30 Delta t_pred at w = 2
       -> "the timing bias is the residue's write-back at the clean DRAM rate"
  reported: bias = Delta t / t(read) per w and for the fixed 4096 x 4096 weight; triton do_bench
  default vs read-patched (exploratory)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from counter_prereg import load_ncu_raw

# R_gpu = 1.5 x (1 - A(1.5)) from each GPU's B9b score (queued_wb_*/score.json, computed 2026-11-06
# before B12): the dirty residue a 1.5 x L2 write leaves in the L2.
R_B9B = {"RTX 3060 Laptop": 0.4722, "RTX 3090": 0.4797, "RTX 2060 SUPER": 0.5063, "RTX 4080 SUPER": 0.6861}
R_V4 = 0.5
CTL_MAX = 64 << 10
READ_W_MAX, READ_R_BAND = 0.05, (0.9, 1.2)
G1_TOL, G1B_SPREAD, G2_TOL, G2_W = 0.15, 0.10, 0.30, 2


def r_gpu(device: str) -> float | None:
    for k, v in R_B9B.items():
        if k in device:
            return v
    return None


def range_rows(metas: list[dict], rows_per_run: list[list[dict]]) -> list[tuple[dict, dict]]:
    out = []
    for m, rows in zip(metas, rows_per_run):
        if len(rows) != len(m["events"]):
            raise ValueError(f"{len(rows)} ranges in the CSV, {len(m['events'])} events in the meta")
        out += list(zip(m["events"], rows))
    return out


def score(rmetas: list[dict], rrows: list[tuple[dict, dict]], tmetas: list[dict], log=print) -> dict:
    med = statistics.median
    l2, dev = rmetas[0]["l2_bytes"], rmetas[0]["device"]
    R = r_gpu(dev)
    fails = sum(m["correctness_failures"] for m in rmetas + tmetas)
    ctl = [r["write"] for e, r in rrows if e["kind"] == "control"]
    ws = sorted({e["w_read"] for e, _ in rrows if e["kind"] == "gemv"})
    s: dict = {}
    for w in ws:
        g = {var: [(e, r) for e, r in rrows if e["kind"] == "gemv" and e["w_read"] == w and e["variant"] == var]
             for var in ("zero", "read")}
        wb = g["read"][0][0]["weight_bytes"]
        s[w] = {"Wz": med(r["write"] for _, r in g["zero"]), "Wr": med(r["write"] for _, r in g["read"]),
                "Rz": med(r["read"] for _, r in g["zero"]), "Rr": med(r["read"] for _, r in g["read"]),
                "weight": wb, "n": len(g["zero"])}
        s[w]["X"] = (s[w]["Wz"] - s[w]["Wr"]) / l2
    v: dict = {"device": dev, "l2_bytes": l2, "R_gpu": R}
    g0 = {"correctness": fails == 0, "control writes": max(ctl) <= CTL_MAX,
          "read flush leaves no dirty data": all(s[w]["Wr"] / l2 <= READ_W_MAX for w in ws),
          "GEMV reads its weight": all(READ_R_BAND[0] <= s[w]["Rr"] / s[w]["weight"] <= READ_R_BAND[1] for w in ws)}
    v["G-0"], v["G-0 parts"] = all(g0.values()), g0
    log(f"{dev}, L2 {l2} B; G-0 {g0} -> {'PASS' if v['G-0'] else 'FAIL'}")
    xs = {w: s[w]["X"] for w in ws}
    if R is None:
        v["G-1"] = "no prediction for this device"
    else:
        ok = [abs(x - R) <= G1_TOL for x in xs.values()]
        v["G-1"] = ("residue = B9b-predicted dirty residue" if all(ok) else "not" if not any(ok) else "mixed")
        mae_b9b = statistics.fmean(abs(x - R) for x in xs.values())
        mae_v4 = statistics.fmean(abs(x - R_V4) for x in xs.values())
        v["G-1 closer to"] = "B9b residue" if mae_b9b < mae_v4 else "v4 (0.5)" if mae_v4 < mae_b9b else "tie"
        v["G-1 MAE"] = {"B9b": round(mae_b9b, 4), "v4": round(mae_v4, 4)}
    v["X"] = {str(w): round(x, 4) for w, x in xs.items()}
    log("G-1 X = " + ", ".join(f"w {w}: {x:.3f}" for w, x in xs.items())
        + f" vs R_gpu {R} (tol {G1_TOL}) and v4 {R_V4} -> {v['G-1']}"
        + (f"; closer to {v['G-1 closer to']} {v['G-1 MAE']}" if R is not None else ""))
    spread = max(xs.values()) - min(xs.values())
    v["G-1b"] = "residue independent of the GEMV size" if spread <= G1B_SPREAD else "size-dependent"
    log(f"G-1b spread {spread:.3f} -> {v['G-1b']}")
    tim = [e for m in tmetas for e in m["events"] if e["kind"] == "timing"]
    t = {}
    for key in list(ws) + ["fixed"]:
        for var in ("zero", "read"):
            xs_t = [x for e in tim if e["w_read"] == key and e["variant"] == var for x in e["times_us"]]
            t[(key, var)] = med(xs_t) if xs_t else None
    dts, preds, bias = {}, {}, {}
    for w in ws:
        dt = t[(w, "zero")] - t[(w, "read")]
        rate = (s[w]["Rr"] + s[w]["Wr"]) / t[(w, "read")]           # bytes per microsecond
        dts[w], preds[w] = dt, s[w]["X"] * l2 / rate
        bias[str(w)] = round(dt / t[(w, "read")], 4)
    if t[("fixed", "zero")] is not None:
        bias["fixed 4096x4096"] = round((t[("fixed", "zero")] - t[("fixed", "read")]) / t[("fixed", "read")], 4)
    ok2 = all(dt > 0 for dt in dts.values()) and G2_W in dts and abs(dts[G2_W] - preds[G2_W]) <= G2_TOL * preds[G2_W]
    v["G-2"] = "timing bias = residue write-back at the clean DRAM rate" if ok2 else "not explained by bytes alone"
    v["G-2 detail"] = {str(w): {"dt_us": round(dts[w], 2), "dt_pred_us": round(preds[w], 2),
                                "t_read_us": round(t[(w, 'read')], 2)} for w in ws}
    v["bias"] = bias
    log("G-2 " + ", ".join(f"w {w}: dt {dts[w]:.2f} us vs pred {preds[w]:.2f} us" for w in ws) + f" -> {v['G-2']}")
    log("bias (t_zero / t_read - 1): " + json.dumps(bias))
    db = [e for m in tmetas for e in m["events"] if e["kind"] == "do_bench"]
    if db:
        v["do_bench (exploratory)"] = {str(e["w_read"]): round(e["default_us"] / e["read_flush_us"] - 1, 4) for e in db}
        log("do_bench default / read-patched - 1 (exploratory, last run): " + json.dumps(v["do_bench (exploratory)"]))
    log("summary per w: " + json.dumps({str(w): {k: (round(x, 1) if isinstance(x, float) else x)
                                                 for k, x in s[w].items()} for w in ws}))
    log("verdict: " + json.dumps({k: v[k] for k in ("G-0", "G-1", "G-1b", "G-2")}))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--range-meta", type=Path, nargs="+", required=True)
    p.add_argument("--range-csv", type=Path, nargs="+", required=True)
    p.add_argument("--timing-meta", type=Path, nargs="+", required=True)
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    rmetas = [json.loads(x.read_text()) for x in a.range_meta]
    tmetas = [json.loads(x.read_text()) for x in a.timing_meta]
    rrows = range_rows(rmetas, [load_ncu_raw(c) for c in a.range_csv])
    v = score(rmetas, rrows, tmetas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
