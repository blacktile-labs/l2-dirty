"""B18c scorer: does cleanbench time a kernel independently of the flush size, and does its steady mode charge
the kernel's own deferred write-back as the B18 counters predict? Rules T-0..T-2 pre-registered in
research/HYPOTHESES.md, section "B18c pre-registration", before any B18c measurement.

  M(k, config) = median over bench() calls (all runs) of the per-call mean time
  T-0  gate: correctness 0 in every run; every (kernel, config) cell has >= 1 bench() call
  T-1  max_k |M(k, cold_2L2) / M(k, cold_256) - 1| <= 0.03 "cold timing does not depend on the flush size";
       any >= 0.10 "depends"; else "partial"
  T-2  S(k) = M(k, steady) / M(k, cold_256) - 1: |S(gemv4)| <= 0.05, 0.30 <= S(e_mid) <= 1.00,
       0.05 <= S(e_large) <= 0.20 -> "steady mode charges the deferred write-back as predicted"; else "not as
       predicted" (the failing kernels are listed)
  reported: every M, S, median of medians, IQR, pads, host dispatch, do_bench as shipped
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

T1_OK, T1_BAD = 0.03, 0.10
T2_BANDS = {"gemv4": (-0.05, 0.05), "e_mid": (0.30, 1.00), "e_large": (0.05, 0.20)}


def score(metas: list[dict], log=print) -> dict:
    med = statistics.median
    cells: dict = {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "bench":
                cells.setdefault((e["kernel"], e["config"]), []).append(e)
    M = {c: med(e["mean_us"] for e in es) for c, es in cells.items()}
    kernels = sorted({k for k, _ in cells})
    v: dict = {"device": metas[0]["device"], "l2_bytes": metas[0]["l2_bytes"]}
    fails = sum(m["correctness_failures"] for m in metas)
    complete = all((k, c) in cells for k in kernels for c in ("cold_2L2", "cold_256", "steady"))
    v["T-0"] = fails == 0 and complete
    log(f"{v['device']}; T-0 correctness {fails == 0}, complete {complete} -> {'PASS' if v['T-0'] else 'FAIL'}")
    if not v["T-0"]:
        v["T-1"] = v["T-2"] = "not scored (T-0 failed)"
        return v
    t1 = {k: M[(k, "cold_2L2")] / M[(k, "cold_256")] - 1 for k in kernels}
    s = {k: M[(k, "steady")] / M[(k, "cold_256")] - 1 for k in kernels}
    v["T-1 ratio - 1"] = {k: round(x, 4) for k, x in t1.items()}
    v["T-2 S"] = {k: round(x, 4) for k, x in s.items()}
    if True:
        worst = max(abs(x) for x in t1.values())
        v["T-1"] = ("cold timing does not depend on the flush size" if worst <= T1_OK
                    else "depends" if worst >= T1_BAD else "partial")
        bad = [k for k, (lo, hi) in T2_BANDS.items() if not (lo <= s[k] <= hi)]
        v["T-2"] = "steady mode charges the deferred write-back as predicted" if not bad else f"not as predicted: {bad}"
    log(f"T-1 {v['T-1 ratio - 1']} -> {v['T-1']}")
    log(f"T-2 S {v['T-2 S']} (bands {T2_BANDS}) -> {v['T-2']}")
    v["M mean us"] = {f"{k} {c}": round(x, 3) for (k, c), x in sorted(M.items())}
    v["median of medians us"] = {f"{k} {c}": round(med(e["median_us"] for e in es), 3) for (k, c), es in sorted(cells.items())}
    v["IQR / median"] = {f"{k} {c}": round(med((e["p75_us"] - e["p25_us"]) / e["median_us"] for e in es), 4)
                         for (k, c), es in sorted(cells.items())}
    v["pad us"] = {k: round(med(e["pad_us"] for (kk, _), es in cells.items() if kk == k for e in es), 1) for k in kernels}
    v["host dispatch us"] = {k: round(med(e["host_dispatch_us"] for (kk, _), es in cells.items() if kk == k for e in es), 1)
                             for k in kernels}
    v["n bench() calls per cell"] = min(len(es) for es in cells.values())
    db = {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "do_bench":
                db.setdefault(e["kernel"], []).append(e["median_us"])
    v["do_bench shipped median us"] = {k: round(med(x), 3) for k, x in db.items()}
    v["do_bench shipped / cleanbench cold_256 - 1"] = {k: round(med(x) / M[(k, "cold_256")] - 1, 4) for k, x in db.items()}
    for key in ("M mean us", "IQR / median", "pad us", "host dispatch us", "do_bench shipped median us",
                "do_bench shipped / cleanbench cold_256 - 1"):
        log(f"{key}: " + json.dumps(v[key]))
    log("verdict: " + json.dumps({h: v[h] for h in ("T-0", "T-1", "T-2")}))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, nargs="+", required=True)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true", help="cell counts only; prints no measurement")
    a = p.parse_args()
    metas = [json.loads(x.read_text()) for x in a.meta]
    if a.check_only:
        cells = {(e["kernel"], e["config"]) for m in metas for e in m["events"] if e["kind"] == "bench"}
        print(f"check-only: {len(metas)} runs, {len(cells)} cells, "
              f"correctness_failures={sum(m['correctness_failures'] for m in metas)}")
        return 0
    v = score(metas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
