"""B13 scorer: does do_bench's write flush change the autotuner's pick for an LLM decode GEMV?
Rules T-0..T-2 pre-registered in research/HYPOTHESES.md, section "B13 pre-registration", before
any B13 measurement. Input: the probe metas of one GPU (3 runs); per (shape, candidate, mode) the
median over runs of do_bench's median time.

  pick_m        = argmin over candidates of t_m          (m = zero: do_bench as shipped; read)
  regret        = t_read(pick_zero) / t_read(pick_read) - 1      (how much slower the shipped pick
                  really is, timed without the dirty-L2 bias)
  T-0  sanity: >= 80 % of the 37 candidates valid in every shape, cuBLAS valid, 3 runs
  T-1  (primary, per GPU) max regret over shapes: <= 0.02 -> "harmless"; >= 0.05 -> "harmful";
       otherwise "minor"
  T-2  cuBLAS-vs-best-Triton decision (which is faster) differs between modes in any shape
       -> "flips"; otherwise "no flip"
  reported: Spearman rank correlation of t_zero vs t_read per shape; bias of the picks
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

VALID_MIN = 0.80
HARMLESS, HARMFUL = 0.02, 0.05


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 1.0 if n else float("nan")


def score(metas: list[dict], log=print) -> dict:
    dev, n_cand = metas[0]["device"], metas[0]["n_candidates"]
    t: dict = {}
    for m in metas:
        for r in m["rows"]:
            t.setdefault((tuple(r["shape"]), r["cand"]), {"zero": [], "read": []})
            t[(tuple(r["shape"]), r["cand"])]["zero"].append(r["zero_ms"])
            t[(tuple(r["shape"]), r["cand"])]["read"].append(r["read_ms"])
    shapes = sorted({s for s, _ in t})
    v: dict = {"device": dev, "runs": len(metas)}
    per = {}
    ok0 = len(metas) >= 3
    for s in shapes:
        cands = [c for (ss, c) in t if ss == s and len(t[(ss, c)]["zero"]) == len(metas)]
        ok0 &= len(cands) >= VALID_MIN * n_cand and "cublas" in cands
        tz = {c: statistics.median(t[(s, c)]["zero"]) for c in cands}
        tr = {c: statistics.median(t[(s, c)]["read"]) for c in cands}
        pz, pr = min(tz, key=tz.get), min(tr, key=tr.get)
        trit = [c for c in cands if c != "cublas"]
        bz, br = min(trit, key=tz.get), min(trit, key=tr.get)
        cub_z = tz["cublas"] < tz[bz] if "cublas" in tz else None
        cub_r = tr["cublas"] < tr[br] if "cublas" in tr else None
        per[f"{s[0]}x{s[1]}"] = {
            "pick_zero": pz, "pick_read": pr, "regret": tr[pz] / tr[pr] - 1,
            "t_read_pick_read_us": tr[pr] * 1e3, "t_read_pick_zero_us": tr[pz] * 1e3,
            "bias_of_pick_zero": tz[pz] / tr[pz] - 1,
            "cublas_faster_zero": cub_z, "cublas_faster_read": cub_r,
            "spearman": spearman([tz[c] for c in cands], [tr[c] for c in cands]), "n": len(cands)}
    v["T-0"] = ok0
    worst = max(p["regret"] for p in per.values())
    v["T-1"] = "harmless" if worst <= HARMLESS else "harmful" if worst >= HARMFUL else "minor"
    v["T-1 max regret"] = round(worst, 4)
    v["T-2"] = "flips" if any(p["cublas_faster_zero"] != p["cublas_faster_read"] for p in per.values()) else "no flip"
    v["per shape"] = {k: {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in p.items()} for k, p in per.items()}
    log(f"{dev}: T-0 {'PASS' if ok0 else 'FAIL'} (runs {len(metas)})")
    for k, p in per.items():
        log(f"  {k}: pick zero {p['pick_zero']} / read {p['pick_read']}; regret {p['regret']:+.3%}; "
            f"pick bias {p['bias_of_pick_zero']:+.1%}; cuBLAS faster: zero {p['cublas_faster_zero']} "
            f"read {p['cublas_faster_read']}; spearman {p['spearman']:.3f}; n {p['n']}")
    log(f"T-1 max regret {worst:+.3%} -> {v['T-1']}; T-2 -> {v['T-2']}")
    log("verdict: " + json.dumps({k: v[k] for k in ("T-0", "T-1", "T-1 max regret", "T-2")}))
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
