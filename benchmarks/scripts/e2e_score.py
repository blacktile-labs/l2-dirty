"""B21 scorer: which per-kernel timing predicts an in-model (CUDA-graph) decode step?
Rules U-0..U-3 pre-registered in research/HYPOTHESES.md, section "B21 pre-registration", before any B21
measurement. Per GPU (--meta: the runs of one GPU) or across GPUs (--all: per-GPU score JSONs).

  T(v)      = median of all step replays (all runs) of variant v
  P_h(v)    = layers x sum over shapes of the median (over runs) of harness h's time for (v, shape)
  E_h(v)    = P_h(v) / T(v) - 1                         (prediction error of the step time)
  S_e2e(v)  = T(fp16) / T(v);  S_h(v) = P_h(fp16) / P_h(v);  D_h(v) = S_h(v) / S_e2e(v) - 1
  U-0  gate: correctness 0; fp16 and int4 steps captured; IQR / median of the fp16 and int4 steps <= 0.05
  U-1  |E_cold(fp16)| <= 0.10 and |E_cold(int4)| <= 0.10 "cold predicts the in-model step"; else "not"
  U-2  only if L2 >= 40 MiB: E_write(fp16) - E_cold(fp16) >= 0.05 "the shipped write flush over-predicts the
       step"; else "not"; smaller L2: "n/a (small L2)"
  U-3  |D_cold(int4)| <= 0.05 and |D_cold(int4)| < |D_write(int4)| "cold predicts the int4 speedup better than the
       write flush"; else "not"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MIB = 1 << 20
STAB, U1_TOL, U2_GAP, U2_L2, U3_TOL = 0.05, 0.10, 0.05, 40 * MIB, 0.05
HARN = ("write", "cold", "steady", "l2w")


def quart(xs):
    xs = sorted(xs)
    q = lambda p: xs[int(round((len(xs) - 1) * p))]  # noqa: E731
    return q(0.25), q(0.5), q(0.75)


def score(metas: list[dict], log=print) -> dict:
    med = statistics.median
    m0 = metas[0]
    dev, l2, L = m0["device"], m0["l2_bytes"], m0["layers"]
    steps, kern = {}, {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "step":
                steps.setdefault(e["variant"], []).extend(e["times_us"])
            elif e["kind"] == "kernel":
                for h in HARN:
                    kern.setdefault((e["variant"], e["shape"], h), []).append(e[f"{h}_us"])
    T = {v: med(ts) for v, ts in steps.items()}
    stab = {v: (quart(ts)[2] - quart(ts)[0]) / quart(ts)[1] for v, ts in steps.items()}
    variants = sorted({v for v, _, _ in kern})
    shapes = sorted({s for _, s, _ in kern})
    P = {(v, h): L * sum(med(kern[(v, s, h)]) for s in shapes) for v in variants for h in HARN}
    E = {(v, h): P[(v, h)] / T[v] - 1 for v in variants for h in HARN if v in T}
    v_out: dict = {"device": dev, "l2_bytes": l2, "T step us": {v: round(t, 2) for v, t in T.items()},
                   "step IQR/median": {v: round(x, 4) for v, x in stab.items()},
                   "E": {f"{v} {h}": round(x, 4) for (v, h), x in E.items()}}
    fails = sum(m["correctness_failures"] for m in metas)
    g0 = {"correctness": fails == 0, "fp16 and int4 captured": {"fp16", "int4"} <= set(T),
          "stable steps": all(stab.get(v, 1) <= STAB for v in ("fp16", "int4"))}
    v_out["U-0"], v_out["U-0 parts"] = all(g0.values()), g0
    log(f"{dev} (L2 {l2 / MIB:.0f} MiB): U-0 {g0} -> {'PASS' if v_out['U-0'] else 'FAIL'}")
    D = {}
    for v in variants:
        if v == "fp16" or v not in T or "fp16" not in T:
            continue
        s_e2e = T["fp16"] / T[v]
        v_out[f"S_e2e {v}"] = round(s_e2e, 4)
        for h in HARN:
            s_h = P[("fp16", h)] / P[(v, h)]
            D[(v, h)] = s_h / s_e2e - 1
    v_out["D"] = {f"{v} {h}": round(x, 4) for (v, h), x in D.items()}
    if not v_out["U-0"]:
        for r in ("U-1", "U-2", "U-3"):
            v_out[r] = "not scored (U-0 failed)"
        return v_out
    v_out["U-1"] = ("cold predicts the in-model step" if abs(E[("fp16", "cold")]) <= U1_TOL
                    and abs(E[("int4", "cold")]) <= U1_TOL else "not")
    gap = E[("fp16", "write")] - E[("fp16", "cold")]
    v_out["U-2 gap"] = round(gap, 4)
    v_out["U-2"] = ("n/a (small L2)" if l2 < U2_L2 else
                    "the shipped write flush over-predicts the step" if gap >= U2_GAP else "not")
    dc, dw = D[("int4", "cold")], D[("int4", "write")]
    v_out["U-3"] = ("cold predicts the int4 speedup better than the write flush"
                    if abs(dc) <= U3_TOL and abs(dc) < abs(dw) else "not")
    log(f"T {v_out['T step us']} · E {v_out['E']}")
    log(f"D {v_out['D']}")
    log("verdict: " + json.dumps({r: v_out[r] for r in ("U-0", "U-1", "U-2", "U-3")}))
    return v_out


def score_all(vs: list[dict], log=print) -> dict:
    ok = [v for v in vs if v.get("U-0") is True]
    out = {"gpus": [v["device"] for v in vs], "U-0 pass": len(ok),
           "U-1": f"{sum(v['U-1'].startswith('cold predicts') for v in ok)}/{len(ok)}",
           "U-2": f"{sum(v['U-2'].startswith('the shipped') for v in ok)}/{sum(not v['U-2'].startswith('n/a') for v in ok)}",
           "U-3": f"{sum(v['U-3'].startswith('cold predicts') for v in ok)}/{len(ok)}"}
    log("across GPUs: " + json.dumps(out))
    return out


def main() -> int:
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
            print(f"check-only: {len(metas)} runs, steps {sorted({e['variant'] for e in ev if e['kind'] == 'step'})}, "
                  f"{sum(e['kind'] == 'kernel' for e in ev)} kernel rows, correctness_failures="
                  f"{sum(m['correctness_failures'] for m in metas)}, notes {metas[0]['notes']}")
            return 0
        v = score(metas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
