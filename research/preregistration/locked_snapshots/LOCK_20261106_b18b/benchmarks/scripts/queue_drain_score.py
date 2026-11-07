"""B18b scorer: does a short L2 flush make CUDA-event timing include host launch latency ("queue drain")?
Rules Q-0..Q-4 pre-registered in research/HYPOTHESES.md, section "B18b pre-registration", before any B18b
measurement. Input: queue_drain_probe.py metas (3 runs). T(k, cond, mode) = median of all samples pooled.

  I(k, cond, a, b) = T(k, cond, a) / T(k, cond, b) - 1
  Q-0  gate: correctness 0; pad_us >= 200 in every run; plain I(gemv4, prL2, pr256) >= 0.20 (B18 replicated)
  Q-1  (primary) padded I for gemv4 (prL2 vs pr256), e_large (prL2 vs pr256), t_small (pr2L2 vs pr256):
       max |I| <= 0.05 "queue drain explains the inflation"; min I >= 0.15 "not queue drain"; else "partial"
  Q-2  max_k |T(k, pad, pr256) / T(k, plain, pr256) - 1| <= 0.05 "a 256 MiB flush already keeps the queue full"
  Q-3  t_small: P = plain I(zeroL2, zero256), Q = padded I(zeroL2, zero256): P < 0.30 "no L2-sized-write anomaly
       on this GPU"; else Q <= 0.15 "the anomaly is queue drain"; Q >= 0.30 "not queue drain"; else "partial"
  Q-4  gemv4: padded I(zero256_wait, zero256): <= 0.10 "the wait slowdown was queue drain";
       >= 0.30 "the wait slowdown is something else (e.g. clocks)"; else "partial"
  reported: every T and I, flush GPU times, host enqueue times, real do_bench (exploratory)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PAD_MIN = 200.0
Q0_REPL = 0.20
Q1_OK, Q1_BAD = 0.05, 0.15
Q2_TOL = 0.05
Q3_ANOM, Q3_OK, Q3_BAD = 0.30, 0.15, 0.30
Q4_OK, Q4_BAD = 0.10, 0.30
Q1_PAIRS = {"gemv4": ("prL2", "pr256"), "e_large": ("prL2", "pr256"), "t_small": ("pr2L2", "pr256")}


def score(metas: list[dict], log=print) -> dict:
    med = statistics.median
    pool: dict = {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "timing":
                pool.setdefault((e["kernel"], e["cond"], e["mode"]), []).extend(e["times_us"])
    T = {c: med(ts) for c, ts in pool.items()}

    def I(k, c, a, b):
        return T[(k, c, a)] / T[(k, c, b)] - 1

    v: dict = {"device": metas[0]["device"], "l2_bytes": metas[0]["l2_bytes"]}
    fails = sum(m["correctness_failures"] for m in metas)
    pads = [m["pad_us"] for m in metas]
    repl = I("gemv4", "plain", "prL2", "pr256")
    g0 = {"correctness": fails == 0, "pad >= 200 us": min(pads) >= PAD_MIN,
          "plain gemv4 prL2 inflation >= 0.20 (B18 replicated)": repl >= Q0_REPL}
    v["Q-0"], v["Q-0 parts"], v["plain gemv4 prL2 vs pr256"] = all(g0.values()), g0, round(repl, 4)
    log(f"{v['device']}; Q-0 {g0} (repl {repl:.3f}, pad {min(pads):.0f} us) -> {'PASS' if v['Q-0'] else 'FAIL'}")
    q1 = {k: I(k, "pad", a, b) for k, (a, b) in Q1_PAIRS.items()}
    q1_plain = {k: I(k, "plain", a, b) for k, (a, b) in Q1_PAIRS.items()}
    v["Q-1 padded I"] = {k: round(x, 4) for k, x in q1.items()}
    v["Q-1 plain I"] = {k: round(x, 4) for k, x in q1_plain.items()}
    q2 = {k: T[(k, "pad", "pr256")] / T[(k, "plain", "pr256")] - 1 for k in Q1_PAIRS}
    v["Q-2 pad/plain - 1 at pr256"] = {k: round(x, 4) for k, x in q2.items()}
    P, Q = I("t_small", "plain", "zeroL2", "zero256"), I("t_small", "pad", "zeroL2", "zero256")
    v["Q-3 P (plain)"], v["Q-3 Q (padded)"] = round(P, 4), round(Q, 4)
    q4 = I("gemv4", "pad", "zero256_wait", "zero256")
    v["Q-4 padded wait vs zero256"] = round(q4, 4)
    if not v["Q-0"]:
        for h in ("Q-1", "Q-2", "Q-3", "Q-4"):
            v[h] = "not scored (Q-0 failed)"
    else:
        v["Q-1"] = ("queue drain explains the inflation" if max(abs(x) for x in q1.values()) <= Q1_OK
                    else "not queue drain" if min(q1.values()) >= Q1_BAD else "partial")
        v["Q-2"] = ("a 256 MiB flush already keeps the queue full" if max(abs(x) for x in q2.values()) <= Q2_TOL
                    else "padding changes the pr256 time")
        v["Q-3"] = ("no L2-sized-write anomaly on this GPU" if P < Q3_ANOM
                    else "the anomaly is queue drain" if Q <= Q3_OK else "not queue drain" if Q >= Q3_BAD else "partial")
        v["Q-4"] = ("the wait slowdown was queue drain" if q4 <= Q4_OK
                    else "the wait slowdown is something else (e.g. clocks)" if q4 >= Q4_BAD else "partial")
    log(f"Q-1 padded {v['Q-1 padded I']} (plain {v['Q-1 plain I']}) -> {v['Q-1']}")
    log(f"Q-2 {v['Q-2 pad/plain - 1 at pr256']} -> {v['Q-2']}")
    log(f"Q-3 P {P:.3f} Q {Q:.3f} -> {v['Q-3']}; Q-4 {q4:.3f} -> {v['Q-4']}")
    v["T median us"] = {f"{k} {c} {m}": round(t, 3) for (k, c, m), t in sorted(T.items())}
    v["n per cell"] = min(len(x) for x in pool.values())
    aux = [e for m in metas for e in m["events"] if e["kind"] == "aux"]
    if aux:
        v["flush GPU us (run medians)"] = {mo: round(med(a["flush_gpu_us"][mo] for a in aux), 2) for mo in aux[0]["flush_gpu_us"]}
        v["host enqueue us (run medians)"] = {k: round(med(a["host_enqueue_us"][k] for a in aux), 2) for k in aux[0]["host_enqueue_us"]}
    db = [e for m in metas for e in m["events"] if e["kind"] == "do_bench"]
    if db:
        v["do_bench (exploratory, per run)"] = [{"kernel": e["kernel"], **{n: round(x, 2) for n, x in e["median_us"].items()}} for e in db]
    log("T: " + json.dumps(v["T median us"]))
    for key in ("flush GPU us (run medians)", "host enqueue us (run medians)", "do_bench (exploratory, per run)"):
        if key in v:
            log(f"{key}: " + json.dumps(v[key]))
    log("verdict: " + json.dumps({h: v[h] for h in ("Q-0", "Q-1", "Q-2", "Q-3", "Q-4")}))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, nargs="+", required=True)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true", help="cell counts only; prints no measurement")
    a = p.parse_args()
    metas = [json.loads(x.read_text()) for x in a.meta]
    if a.check_only:
        cells = {(e["kernel"], e["cond"], e["mode"]) for m in metas for e in m["events"] if e["kind"] == "timing"}
        print(f"check-only: {len(metas)} runs, {len(cells)} timing cells, "
              f"correctness_failures={sum(m['correctness_failures'] for m in metas)}")
        return 0
    v = score(metas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
