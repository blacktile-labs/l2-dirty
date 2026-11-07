"""B23 scorer: does a single-kernel benchmark time each kernel within 1 % of its CUPTI time inside the model?
Rules W-0..W-3 pre-registered in research/HYPOTHESES.md, section "B23 pre-registration", before any B23 measurement.
Per GPU (--meta: the runs of one GPU) or across GPUs (--all: per-GPU score JSONs).

  rows: (variant, shape) for fp16, int4 and (if present) nf4 x 4 shapes
  per run: T_r = median of the row's truth samples; H_r = the harness's median (cupti_*) or value (event_*)
  T = median over runs of T_r; H = median over runs of H_r; E_h = H / T - 1
  W-0  gate: 3 full (non-smoke) runs with the same rows; correctness 0; every record check ok in every run; truth
       present for every fp16 and int4 row; tracing: |median(traced step) / median(untraced step) - 1| <= 0.005 for
       every variant (CUDA events, paired halves of the same segments, pooled over runs); truth stable: IQR / median
       of the row's pooled truth samples <= 0.05 for every row
  W-1  (primary) m = max over rows |E_cupti_rot|: <= 0.01 "every kernel within 1 %"; <= 0.02 "every kernel within
       2 %"; else "not" (rows within 1 % reported in every case)
  W-2  median over rows |E_cupti_cold| < median over rows |E_event_cold| "CUPTI closer than events"; else "not"
  W-3  (descriptive) w = median over rows of cupti_rot_idle / cupti_rot - 1: |w| >= 0.01 "warm-up changes kernel
       time by >= 1 %"; else "warm-up effect < 1 %"
  descriptive: rows within 1 % for every harness; kernel share of the step = layers x sum_s T(v, s) / step;
       tracing effect on the rotation graphs; spread of the truth's per-round medians; shortest warm-up done
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HARN = ("cupti_rot", "cupti_rot_idle", "cupti_cold", "event_cold", "event_graph")
TRACE_TOL, STAB, RUNS = 0.005, 0.05, 3


def quant(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def iqr_rel(xs):
    return (quant(xs, 0.75) - quant(xs, 0.25)) / quant(xs, 0.5)


def hval(row, h):
    x = row.get(h)
    return x["median"] if isinstance(x, dict) else x


def band(m, a, b):
    return a if m <= 0.01 else b if m <= 0.02 else "not"


def score(metas, log=print):
    med = statistics.median
    m0 = metas[0]
    per_run, samples, steps = {}, {}, {}
    oks, rowsets, rot_trace, rounds, warms = [], [], {}, {}, []
    for m in metas:
        rowsets.append(sorted((e["variant"], e["shape"]) for e in m["events"] if e["kind"] == "row"))
        for e in m["events"]:
            if e["kind"] == "step":
                st = steps.setdefault(e["variant"], {"untraced": [], "traced": []})
                st["untraced"].extend(e["untraced_us"])
                st["traced"].extend(e["traced_us"] or [])
                oks.append(e["count_ok"])
                warms += [w["warm_s"] for w in e.get("warm", [])]
            elif e["kind"] == "row":
                key = (e["variant"], e["shape"])
                oks.extend(e["count_ok"].values())
                warms += [w["warm_s"] for w in e.get("warm", [])]
                if e.get("rot_tracing_effect") is not None:
                    rot_trace.setdefault(key, []).append(e["rot_tracing_effect"])
                rounds.setdefault(key, []).extend(x for x in e["truth"].get("round_medians", []) if x is not None)
                t = e["truth"]["samples"]
                samples.setdefault(key, []).extend(t)
                r = per_run.setdefault(key, {"truth": [], **{h: [] for h in HARN}})
                if t:
                    r["truth"].append(med(t))
                for h in HARN:
                    if hval(e, h) is not None:
                        r[h].append(hval(e, h))
    rows = sorted(k for k, r in per_run.items() if r["truth"])
    T = {k: med(per_run[k]["truth"]) for k in rows}
    H = {(k, h): med(per_run[k][h]) for k in rows for h in HARN if per_run[k][h]}
    Err = {(k, h): H[(k, h)] / T[k] - 1 for (k, h) in H}
    trace = {v: med(s["traced"]) / med(s["untraced"]) - 1 for v, s in steps.items() if s["traced"]}
    stab = {k: iqr_rel(samples[k]) for k in rows}
    fails = sum(m["correctness_failures"] for m in metas)
    need = {(v, s) for (v, s) in per_run if v in ("fp16", "int4")}
    g0 = {"3 full runs": len(metas) == RUNS and not any(m.get("smoke") for m in metas)
                         and all(r == rowsets[0] for r in rowsets),
          "correctness": fails == 0, "record checks": bool(oks) and all(oks),
          "fp16 and int4 truth": bool(need) and need <= set(rows) and {"fp16", "int4"} <= {v for v, _ in need},
          "tracing <= 0.5 %": bool(trace) and set(trace) == set(steps) and all(abs(x) <= TRACE_TOL for x in trace.values()),
          "truth stable": bool(rows) and all(x <= STAB for x in stab.values())}
    name = lambda k: f"{k[0]} {k[1]}"  # noqa: E731
    out = {"device": m0["device"], "l2_bytes": m0["l2_bytes"], "runs": len(metas),
           "truth us": {name(k): round(T[k], 3) for k in rows},
           "E": {h: {name(k): round(Err[(k, h)], 4) for k in rows if (k, h) in Err} for h in HARN},
           "within 1 %": {h: f"{sum(abs(Err[(k, h)]) <= 0.01 for k in rows if (k, h) in Err)}/"
                             f"{sum((k, h) in Err for k in rows)}" for h in HARN},
           "tracing effect": {v: round(x, 4) for v, x in trace.items()},
           "truth IQR/median": {name(k): round(x, 4) for k, x in stab.items()},
           "kernel share of step": {v: round(m0["layers"] * sum(T[k] for k in rows if k[0] == v)
                                             / med(steps[v]["untraced"]), 4) for v in steps
                                    if steps[v]["untraced"] and any(k[0] == v for k in rows)},
           "rotation tracing effect": {name(k): round(med(x), 4) for k, x in rot_trace.items()},
           "truth round spread": {name(k): round((max(x) - min(x)) / med(x), 4) for k, x in rounds.items() if x},
           "shortest warm-up s": round(min(warms), 3) if warms else None,
           "W-0 parts": g0, "W-0": all(g0.values())}
    log(f"{out['device']} (L2 {out['l2_bytes'] >> 20} MiB): W-0 {g0} -> {'PASS' if out['W-0'] else 'FAIL'}")
    if not out["W-0"]:
        out["W-1"] = out["W-2"] = out["W-3"] = "not scored (W-0 failed)"
        return out
    m = max(abs(Err[(k, "cupti_rot")]) for k in rows)
    out["W-1 max |E_cupti_rot|"] = round(m, 4)
    out["W-1"] = band(m, "every kernel within 1 %", "every kernel within 2 %")
    mc = med([abs(Err[(k, "cupti_cold")]) for k in rows])
    me = med([abs(Err[(k, "event_cold")]) for k in rows])
    out["W-2 median |E| cupti_cold / event_cold"] = [round(mc, 4), round(me, 4)]
    out["W-2"] = "CUPTI closer than events" if mc < me else "not"
    w = med([H[(k, "cupti_rot_idle")] / H[(k, "cupti_rot")] - 1 for k in rows])
    out["W-3 median idle / warm - 1"] = round(w, 4)
    out["W-3"] = "warm-up changes kernel time by >= 1 %" if abs(w) >= 0.01 else "warm-up effect < 1 %"
    log(f"truth {out['truth us']}")
    for h in HARN:
        log(f"E {h}: {out['E'][h]}  (within 1 %: {out['within 1 %'][h]})")
    log(f"tracing {out['tracing effect']} · kernel share {out['kernel share of step']}")
    log("verdict: " + json.dumps({k: out[k] for k in ("W-0", "W-1", "W-2", "W-3")}))
    return out


def score_all(vs, log=print):
    ok = [v for v in vs if v.get("W-0") is True]
    out = {"gpus": [v["device"] for v in vs], "W-0 pass": len(ok),
           "W-1": [f"{v['device']}: {v['W-1']} ({v['within 1 %']['cupti_rot']} rows within 1 %)" for v in ok],
           "W-1 every kernel within 1 %": f"{sum(v['W-1'] == 'every kernel within 1 %' for v in ok)}/{len(ok)}",
           "W-2": f"{sum(v['W-2'] == 'CUPTI closer than events' for v in ok)}/{len(ok)}",
           "W-3": [f"{v['device']}: {v['W-3']}" for v in ok]}               # lists: two hosts may share a GPU model
    log("across GPUs: " + json.dumps(out))
    return out


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--meta", type=Path, nargs="+")
    g.add_argument("--all", type=Path, nargs="+")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true", help="counts only; prints no measurement")
    a = p.parse_args()
    if a.all:
        v = score_all([json.loads(x.read_text()) for x in a.all])
    else:
        metas = [json.loads(x.read_text()) for x in a.meta]
        if a.check_only:
            ev = [e for m in metas for e in m["events"]]
            rows = [e for e in ev if e["kind"] == "row"]
            oks = [e["count_ok"] for e in ev if e["kind"] == "step"] + [x for r in rows for x in r["count_ok"].values()]
            print(f"check-only: {len(metas)} runs, steps {sorted({e['variant'] for e in ev if e['kind'] == 'step'})}, "
                  f"{len(rows)} rows, record checks ok {sum(oks)}/{len(oks)}, truth samples per row "
                  f"{sorted({len(r['truth']['samples']) for r in rows})}, copies {metas[0]['ncopies']}, "
                  f"correctness_failures={sum(m['correctness_failures'] for m in metas)}, notes {metas[0]['notes']}")
            return 0
        v = score(metas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
