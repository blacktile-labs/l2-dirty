"""B22 scorer: can a per-kernel timing predict an in-model decode step within 1 %?
Rules V-0..V-3 pre-registered in research/HYPOTHESES.md, section "B22 pre-registration", before any B22 measurement.
Per GPU (--meta: the runs of one GPU) or across GPUs (--all: per-GPU score JSONs).

  T(v) = median of all step replays; P_h(v) = layers x sum over shapes of the median over runs of harness h
  E_h(v) = P_h(v) / T(v) - 1;  D_h(int4) = [P_h(fp16) / P_h(int4)] / [T(fp16) / T(int4)] - 1
  V-0  gate: correctness 0; fp16 and int4 steps present; IQR / median of each step <= 0.03
  V-1  (primary) m = max(|E_graph(fp16)|, |E_graph(int4)|): m <= 0.01 "within 1 %"; m <= 0.02 "within 2 %"; else "not"
  V-2  |D_graph(int4)| <= 0.01 "speedup within 1 %"; <= 0.02 "speedup within 2 %"; else "not"
  V-3  |E_graph(v)| < |E_cold(v)| for fp16 and int4 "graph timing beats cold"; else "not"
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HARN = ("graph", "cold", "write")
STAB = 0.03


def iqr_rel(xs):
    xs = sorted(xs)
    q = lambda p: xs[int(round((len(xs) - 1) * p))]  # noqa: E731
    return (q(0.75) - q(0.25)) / q(0.5)


def band(x, one, two, a, b):
    return a if x <= one else b if x <= two else "not"


def score(metas, log=print):
    med = statistics.median
    m0 = metas[0]
    steps, kern = {}, {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "step":
                steps.setdefault(e["variant"], []).extend(e["times_us"])
            elif e["kind"] == "kernel":
                for h in HARN:
                    kern.setdefault((e["variant"], e["shape"], h), []).append(e[f"{h}_us"])
    T = {v: med(x) for v, x in steps.items()}
    stab = {v: iqr_rel(x) for v, x in steps.items()}
    shapes = sorted({s for _, s, _ in kern})
    variants = sorted({v for v, _, _ in kern})
    P = {(v, h): m0["layers"] * sum(med(kern[(v, s, h)]) for s in shapes) for v in variants for h in HARN}
    E = {(v, h): P[(v, h)] / T[v] - 1 for v in variants for h in HARN if v in T}
    out = {"device": m0["device"], "l2_bytes": m0["l2_bytes"], "T step us": {v: round(t, 2) for v, t in T.items()},
           "step IQR/median": {v: round(x, 4) for v, x in stab.items()},
           "E": {f"{v} {h}": round(x, 4) for (v, h), x in E.items()}}
    fails = sum(m["correctness_failures"] for m in metas)
    g0 = {"correctness": fails == 0, "steps": {"fp16", "int4"} <= set(T),
          "stable": all(stab.get(v, 1) <= STAB for v in ("fp16", "int4"))}
    out["V-0"], out["V-0 parts"] = all(g0.values()), g0
    D = {}
    for v in variants:
        if v != "fp16" and v in T and "fp16" in T:
            for h in HARN:
                D[(v, h)] = (P[("fp16", h)] / P[(v, h)]) / (T["fp16"] / T[v]) - 1
    out["D"] = {f"{v} {h}": round(x, 4) for (v, h), x in D.items()}
    log(f"{out['device']} (L2 {out['l2_bytes'] >> 20} MiB): V-0 {g0} -> {'PASS' if out['V-0'] else 'FAIL'}")
    if not out["V-0"]:
        out["V-1"] = out["V-2"] = out["V-3"] = "not scored (V-0 failed)"
        return out
    m = max(abs(E[("fp16", "graph")]), abs(E[("int4", "graph")]))
    out["V-1 max |E_graph|"] = round(m, 4)
    out["V-1"] = band(m, 0.01, 0.02, "within 1 %", "within 2 %")
    d = abs(D[("int4", "graph")])
    out["V-2"] = band(d, 0.01, 0.02, "speedup within 1 %", "speedup within 2 %")
    out["V-3"] = ("graph timing beats cold" if all(abs(E[(v, "graph")]) < abs(E[(v, "cold")]) for v in ("fp16", "int4"))
                  else "not")
    log(f"T {out['T step us']} IQR {out['step IQR/median']}")
    log(f"E {out['E']}")
    log(f"D {out['D']}")
    log("verdict: " + json.dumps({k: out[k] for k in ("V-0", "V-1", "V-2", "V-3")}))
    return out


def score_all(vs, log=print):
    ok = [v for v in vs if v.get("V-0") is True]
    out = {"gpus": [v["device"] for v in vs], "V-0 pass": len(ok),
           "V-1": {v["device"]: v["V-1"] for v in ok}, "V-2": {v["device"]: v["V-2"] for v in ok},
           "V-3": f"{sum(v['V-3'] == 'graph timing beats cold' for v in ok)}/{len(ok)}"}
    log("across GPUs: " + json.dumps(out))
    return out


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--meta", type=Path, nargs="+")
    g.add_argument("--all", type=Path, nargs="+")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if a.all:
        v = score_all([json.loads(x.read_text()) for x in a.all])
    else:
        metas = [json.loads(x.read_text()) for x in a.meta]
        if a.check_only:
            ev = [e for m in metas for e in m["events"]]
            print(f"check-only: {len(metas)} runs, steps {sorted({e['variant'] for e in ev if e['kind'] == 'step'})}, "
                  f"{sum(e['kind'] == 'kernel' for e in ev)} kernel rows, copies {metas[0]['ncopies']}, "
                  f"correctness_failures={sum(m['correctness_failures'] for m in metas)}, notes {metas[0]['notes']}")
            return 0
        v = score(metas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
