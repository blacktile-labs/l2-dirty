"""B16 scorer: do_bench's write-flush bias on production 4-bit decode kernels.
Rules P-0, V-1..V-3 pre-registered in research/HYPOTHESES.md, section "B16 pre-registration", before
any B16 measurement. Input: the probe metas of one GPU (3 runs); medians over runs per (shape, cand, mode).

Production 4-bit kernels q in {tinygemm (best of k2/k4/k8 per mode), bnb_nf4}; baseline cublas_fp16.
  S_q,m = t(cublas_fp16, m) / t(q, m);  D_q = S_q,zero / S_q,read - 1
  P-0  3 runs; cublas_fp16, >= 1 tinygemm config and bnb_nf4 valid in every shape
  V-1  (primary) max |D| over q and shapes: <= 0.03 "faithful"; >= 0.10 "distorted"; else "mildly
       distorted"; sign reported ("under-estimated" if every D < 0)
  V-2  the faster production 4-bit kernel (tinygemm-best vs bnb_nf4) differs between modes in any
       shape -> "flips"; otherwise "no flip"
  V-3  (reported) regret of the tinygemm innerKTiles pick: t_read(pick_zero) / t_read(pick_read) - 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

FAITHFUL, DISTORTED = 0.03, 0.10


def score(metas: list[dict], log=print) -> dict:
    t: dict = {}
    for m in metas:
        for r in m["rows"]:
            d = t.setdefault((tuple(r["shape"]), r["cand"]), {"zero": [], "read": []})
            d["zero"].append(r["zero_ms"])
            d["read"].append(r["read_ms"])
    med = {k: {m: statistics.median(v[m]) for m in v} for k, v in t.items() if len(v["zero"]) == len(metas)}
    shapes = sorted({s for s, _ in med})
    ok0, per, ds = len(metas) >= 3, {}, []
    for s in shapes:
        cands = [c for (ss, c) in med if ss == s]
        tiny = [c for c in cands if c.startswith("tinygemm_")]
        ok = "cublas_fp16" in cands and tiny and "bnb_nf4" in cands
        ok0 &= bool(ok)
        if not ok:
            per[f"{s[0]}x{s[1]}"] = {"missing": True, "cands": cands}
            continue
        p = {}
        best = {}
        for m in ("zero", "read"):
            best[m] = {"tinygemm": min(tiny, key=lambda c: med[(s, c)][m]), "bnb_nf4": "bnb_nf4"}
        for q in ("tinygemm", "bnb_nf4"):
            sz = med[(s, "cublas_fp16")]["zero"] / med[(s, best["zero"][q])]["zero"]
            sr = med[(s, "cublas_fp16")]["read"] / med[(s, best["read"][q])]["read"]
            ds.append(sz / sr - 1)
            p[f"S_{q}"] = {"zero": round(sz, 3), "read": round(sr, 3), "D": round(sz / sr - 1, 4)}
        p["t_us"] = {c: {m: round(med[(s, c)][m] * 1e3, 2) for m in ("zero", "read")} for c in cands}
        p["bias"] = {c: round(med[(s, c)]["zero"] / med[(s, c)]["read"] - 1, 4) for c in cands}
        p["faster_4bit"] = {m: min(("tinygemm", "bnb_nf4"), key=lambda q: med[(s, best[m][q])][m]) for m in ("zero", "read")}
        pz, pr = best["zero"]["tinygemm"], best["read"]["tinygemm"]
        p["tinygemm_pick"] = {"zero": pz, "read": pr, "regret": round(med[(s, pz)]["read"] / med[(s, pr)]["read"] - 1, 4)}
        per[f"{s[0]}x{s[1]}"] = p
    worst = max(ds, key=abs) if ds else float("nan")
    v = {"device": metas[0]["device"], "l2_bytes": metas[0]["l2_bytes"], "P-0": bool(ok0),
         "V-1": ("faithful" if abs(worst) <= FAITHFUL else "distorted" if abs(worst) >= DISTORTED else "mildly distorted")
         if ds else "not measured",
         "V-1 worst D": round(worst, 4) if ds else None,
         "V-1 sign": ("speedup under-estimated" if ds and all(d < 0 for d in ds) else
                      "speedup over-estimated" if ds and all(d > 0 for d in ds) else "mixed signs"),
         "V-2": "flips" if any(p.get("faster_4bit", {}).get("zero") != p.get("faster_4bit", {}).get("read")
                               for p in per.values() if not p.get("missing")) else "no flip",
         "V-3 max regret": max((p["tinygemm_pick"]["regret"] for p in per.values() if not p.get("missing")), default=None),
         "per shape": per, "notes": metas[0].get("notes", {}), "invalid": metas[0].get("invalid", [])}
    log(f"{v['device']} (L2 {v['l2_bytes'] >> 20} MiB): P-0 {'PASS' if ok0 else 'FAIL'}")
    for k, p in per.items():
        if p.get("missing"):
            log(f"  {k}: MISSING candidates {p['cands']}")
            continue
        log(f"  {k}: S_tinygemm {p['S_tinygemm']}; S_bnb {p['S_bnb_nf4']}; bias {p['bias']}; "
            f"faster 4-bit {p['faster_4bit']}; tinygemm pick {p['tinygemm_pick']}")
    log(f"V-1 worst D {v['V-1 worst D']} -> {v['V-1']} ({v['V-1 sign']}); V-2 -> {v['V-2']}; V-3 max regret {v['V-3 max regret']}")
    log("verdict: " + json.dumps({k: v[k] for k in ("P-0", "V-1", "V-1 worst D", "V-1 sign", "V-2", "V-3 max regret")}))
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, nargs="+", required=True)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    v = score([json.loads(m.read_text()) for m in a.meta])
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
