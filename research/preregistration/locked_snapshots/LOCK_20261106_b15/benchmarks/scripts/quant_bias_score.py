"""B14 scorer: does do_bench's write flush distort the measured speedup of weight-quantized decode
GEMVs (candidates reading different bytes)? Rules U-0..U-2 pre-registered in research/HYPOTHESES.md,
section "B14 pre-registration", before any B14 measurement. Input: the probe metas of one GPU.

Per shape and mode m (zero = do_bench as shipped, read = read-only flush), with medians over runs:
  best_f,m  = fastest candidate of format f (fp16 includes cuBLAS)
  S_q,m     = t(best_fp16,m) / t(best_q,m)       for q in {w8, w4}: the measured quantization speedup
  D_q       = S_q,zero / S_q,read - 1            the distortion of that speedup under do_bench
  U-0  sanity: 3 runs; >= 80 % of candidates valid in every shape; every format has a valid config
  U-1  (primary, per GPU) max |D| over shapes and q: <= 0.03 "faithful"; >= 0.10 "distorted";
       otherwise "mildly distorted"; the sign (under- or over-estimated) is reported
  U-2  the fastest format differs between modes in any shape -> "flips"; otherwise "no flip"
  reported: regret of the overall pick (as B13 T-1)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

FORMATS = ("fp16", "w8", "w4")
VALID_MIN = 0.80
FAITHFUL, DISTORTED = 0.03, 0.10


def score(metas: list[dict], log=print) -> dict:
    dev, n_cand = metas[0]["device"], metas[0]["n_candidates"]
    t: dict = {}
    fmt: dict = {}
    for m in metas:
        for r in m["rows"]:
            k = (tuple(r["shape"]), r["cand"])
            t.setdefault(k, {"zero": [], "read": []})
            t[k]["zero"].append(r["zero_ms"])
            t[k]["read"].append(r["read_ms"])
            fmt[r["cand"]] = r["format"]
    shapes = sorted({s for s, _ in t})
    ok0 = len(metas) >= 3
    per, ds = {}, []
    for s in shapes:
        cands = [c for (ss, c) in t if ss == s and len(t[(ss, c)]["zero"]) == len(metas)]
        ok0 &= len(cands) >= VALID_MIN * n_cand and all(any(fmt[c] == f for c in cands) for f in FORMATS)
        med = {m: {c: statistics.median(t[(s, c)][m]) for c in cands} for m in ("zero", "read")}
        best = {m: {f: min((c for c in cands if fmt[c] == f), key=med[m].get) for f in FORMATS} for m in med}
        tb = {m: {f: med[m][best[m][f]] for f in FORMATS} for m in med}
        p = {"t_us": {m: {f: round(tb[m][f] * 1e3, 2) for f in FORMATS} for m in med}}
        for q in ("w8", "w4"):
            sz, sr = tb["zero"]["fp16"] / tb["zero"][q], tb["read"]["fp16"] / tb["read"][q]
            d = sz / sr - 1
            ds.append(d)
            p[f"S_{q}"] = {"zero": round(sz, 3), "read": round(sr, 3), "D": round(d, 4)}
        p["fastest"] = {m: min(FORMATS, key=tb[m].get) for m in med}
        pz, pr = min(cands, key=med["zero"].get), min(cands, key=med["read"].get)
        p["regret"] = round(med["read"][pz] / med["read"][pr] - 1, 4)
        per[f"{s[0]}x{s[1]}"] = p
    worst = max(ds, key=abs)
    v = {"device": dev, "runs": len(metas), "U-0": ok0,
         "U-1": "faithful" if abs(worst) <= FAITHFUL else "distorted" if abs(worst) >= DISTORTED else "mildly distorted",
         "U-1 worst D": round(worst, 4),
         "U-1 sign": "speedup under-estimated" if all(d < 0 for d in ds) else
                     "speedup over-estimated" if all(d > 0 for d in ds) else "mixed signs",
         "U-2": "flips" if any(p["fastest"]["zero"] != p["fastest"]["read"] for p in per.values()) else "no flip",
         "per shape": per}
    log(f"{dev}: U-0 {'PASS' if ok0 else 'FAIL'} (runs {len(metas)})")
    for k, p in per.items():
        log(f"  {k}: t_us zero {p['t_us']['zero']} read {p['t_us']['read']}; S_w8 {p['S_w8']}; S_w4 {p['S_w4']}; "
            f"fastest {p['fastest']}; regret {p['regret']:+.2%}")
    log(f"U-1 worst D {worst:+.2%} -> {v['U-1']} ({v['U-1 sign']}); U-2 -> {v['U-2']}")
    log("verdict: " + json.dumps({k: v[k] for k in ("U-0", "U-1", "U-1 worst D", "U-1 sign", "U-2")}))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, nargs="+", required=True)
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    v = score([json.loads(m.read_text()) for m in a.meta])
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
