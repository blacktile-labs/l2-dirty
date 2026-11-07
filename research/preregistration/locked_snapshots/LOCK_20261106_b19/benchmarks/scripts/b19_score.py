"""B19 scorer: B17's L2-sized-flush anomaly vs queue drain, and cleanbench's cold/steady gap on large-L2 GPUs.
Rules R-0..R-4 pre-registered in research/HYPOTHESES.md, section "B19 pre-registration", before any B19 GPU was
rented. Per GPU (--meta, the runs of one GPU) or across GPUs (--all, per-GPU score JSONs).

  T(k, cond, mode) = median of the pooled part-A samples; I(k, cond, a, b) = T(k, cond, a) / T(k, cond, b) - 1
  M(k, cfg) = median over bench() calls of the per-call mean; S(k) = M(k, steady) / M(k, cold) - 1
  R-0  gate: correctness 0, Liger present (rms16 and rms64 built), every pad >= 200 us
  R-1  plain I(rms16, zeroL2, zero256) >= 0.5 "the B17 anomaly replicates"; else "no anomaly"
  R-2  if R-1 replicates: padded I(rms16, zeroL2, zero256) <= 0.15 "queue drain"; >= 0.5 "not queue drain";
       else "partial"; if not: "n/a"
  R-3  padded T(gemv16, none) <= 0.85 x padded T(gemv16, pr256) "no flush is warm once the queue is busy";
       else "not warm"
  R-4a S(e_q) in [0.30, 1.20] and |S(gemv)| <= 0.05 "steady charges the deferred output write-back"; else "not"
  R-4b |S(rms64) - P| <= 0.10 with P = r L2 / (128 MiB - r L2), r = this GPU's B17 implied residue / L2
       "the residue predicts the cold/steady gap"; else "not"
  across GPUs: R-2 "queue drain explains the anomaly on n/m GPUs where it replicates"; R-3, R-4a counts;
       R-4b "residue predicts the gap" if >= 3 of 4 GPUs pass
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# B17 implied residue / L2 (benchmarks/results/b17_summary_20261106/all.json, "residue / L2 (B_ref x 112 MiB / L2)")
RESIDUE_B17 = {"A100": 0.448, "H100": 0.431, "L40S": 0.295, "RTX 5090": 0.126, "RTX 4090": 0.271,
               "RTX 4080 SUPER": 0.279, "RTX 4060 Ti": 0.301, "RTX 4060": 0.311, "RTX 5060 Ti": 0.229}
PAD_MIN = 200.0
R1_REPL, R2_OK, R2_BAD, R3_WARM = 0.5, 0.15, 0.5, 0.85
R4A_EQ, R4A_GEMV, R4B_TOL = (0.30, 1.20), 0.05, 0.10
MIB = 1 << 20


def residue_for(device: str) -> float | None:
    for name in sorted(RESIDUE_B17, key=len, reverse=True):     # "RTX 4060 Ti" before "RTX 4060"
        if name in device:
            return RESIDUE_B17[name]
    return None


def score(metas: list[dict], log=print) -> dict:
    med = statistics.median
    dev, l2 = metas[0]["device"], metas[0]["l2_bytes"]
    pool, bench = {}, {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "loop":
                pool.setdefault((e["kernel"], e["cond"], e["mode"]), []).extend(e["times_us"])
            elif e["kind"] == "bench":
                bench.setdefault((e["kernel"], e["config"]), []).append(e["mean_us"])
    T = {c: med(x) for c, x in pool.items()}
    M = {c: med(x) for c, x in bench.items()}

    def I(k, c, a, b):
        return T[(k, c, a)] / T[(k, c, b)] - 1
    v: dict = {"device": dev, "l2_bytes": l2}
    kernels = set(metas[0]["kernels"])
    fails = sum(m["correctness_failures"] for m in metas)
    pads = min(p for m in metas for p in m["pads_us"].values())
    g0 = {"correctness": fails == 0, "liger": {"rms16", "rms64"} <= kernels, "pad >= 200 us": pads >= PAD_MIN}
    v["R-0"], v["R-0 parts"] = all(g0.values()), g0
    log(f"{dev} (L2 {l2 / MIB:.0f} MiB): R-0 {g0} -> {'PASS' if v['R-0'] else 'FAIL'}")
    if not v["R-0"]:
        for h in ("R-1", "R-2", "R-3", "R-4a", "R-4b"):
            v[h] = "not scored (R-0 failed)"
        return v
    p1, p2 = I("rms16", "plain", "zeroL2", "zero256"), I("rms16", "pad", "zeroL2", "zero256")
    v["R-1 plain I"], v["R-2 padded I"] = round(p1, 4), round(p2, 4)
    v["R-1"] = "the B17 anomaly replicates" if p1 >= R1_REPL else "no anomaly"
    if v["R-1"] == "no anomaly":
        v["R-2"] = "n/a"
    else:
        v["R-2"] = "queue drain" if p2 <= R2_OK else "not queue drain" if p2 >= R2_BAD else "partial"
    warm_pad = T[("gemv16", "pad", "none")] / T[("gemv16", "pad", "pr256")]
    warm_plain = T[("gemv16", "plain", "none")] / T[("gemv16", "plain", "pr256")]
    v["R-3 none/pr256 padded"], v["R-3 none/pr256 plain"] = round(warm_pad, 4), round(warm_plain, 4)
    v["R-3"] = "no flush is warm once the queue is busy" if warm_pad <= R3_WARM else "not warm"
    S = {k: M[(k, "steady")] / M[(k, "cold")] - 1 for k in {k for k, _ in M}}
    v["S"] = {k: round(x, 4) for k, x in sorted(S.items())}
    ok4a = R4A_EQ[0] <= S["e_q"] <= R4A_EQ[1] and abs(S["gemv"]) <= R4A_GEMV
    v["R-4a"] = "steady charges the deferred output write-back" if ok4a else "not"
    r = residue_for(dev)
    if r is None:
        v["R-4b"] = "no B17 residue for this GPU"
    else:
        rb = r * l2
        P = rb / (128 * MIB - rb)
        v["R-4b predicted S(rms64)"] = round(P, 4)
        v["R-4b"] = "the residue predicts the cold/steady gap" if abs(S["rms64"] - P) <= R4B_TOL else "not"
    log(f"R-1 plain I {p1:.3f} -> {v['R-1']}; R-2 padded I {p2:.3f} -> {v['R-2']}")
    log(f"R-3 none/pr256 padded {warm_pad:.3f} (plain {warm_plain:.3f}) -> {v['R-3']}")
    log(f"R-4 S {v['S']}; R-4a -> {v['R-4a']}; R-4b pred {v.get('R-4b predicted S(rms64)')} -> {v['R-4b']}")
    v["T median us"] = {f"{k} {c} {m}": round(t, 3) for (k, c, m), t in sorted(T.items())}
    v["M mean us"] = {f"{k} {c}": round(x, 3) for (k, c), x in sorted(M.items())}
    db: dict = {}
    for m in metas:
        for e in m["events"]:
            if e["kind"] == "do_bench":
                db.setdefault(e["kernel"], []).append(e["median_us"])
    v["do_bench shipped median us"] = {k: round(med(x), 3) for k, x in sorted(db.items())}
    v["do_bench shipped / cleanbench cold - 1"] = {k: round(med(x) / M[(k, "cold")] - 1, 4)
                                                   for k, x in sorted(db.items()) if (k, "cold") in M}
    v["host dispatch us"] = {k: round(med(m["host_dispatch_us"][k] for m in metas), 1) for k in metas[0]["host_dispatch_us"]}
    for key in ("T median us", "M mean us", "do_bench shipped / cleanbench cold - 1", "host dispatch us"):
        log(f"{key}: " + json.dumps(v[key]))
    log("verdict: " + json.dumps({h: v[h] for h in ("R-0", "R-1", "R-2", "R-3", "R-4a", "R-4b")}))
    return v


def score_all(vs: list[dict], log=print) -> dict:
    ok = [v for v in vs if v.get("R-0") is True]
    repl = [v for v in ok if v["R-1"] == "the B17 anomaly replicates"]
    qd = [v for v in repl if v["R-2"] == "queue drain"]
    out = {"gpus": [v["device"] for v in vs], "R-0 pass": len(ok),
           "R-2": f"queue drain explains the anomaly on {len(qd)}/{len(repl)} GPUs where it replicates",
           "R-2 all": bool(repl) and len(qd) == len(repl),
           "R-3": f"{sum(v['R-3'].startswith('no flush is warm') for v in ok)}/{len(ok)} warm",
           "R-4a": f"{sum(v['R-4a'].startswith('steady charges') for v in ok)}/{len(ok)}"}
    n4b = sum(v["R-4b"].startswith("the residue predicts") for v in ok)
    out["R-4b"] = ("the residue predicts the gap" if n4b >= 3 else "not") + f" ({n4b}/{len(ok)})"
    log("across GPUs: " + json.dumps(out))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--meta", type=Path, nargs="+")
    g.add_argument("--all", type=Path, nargs="+", help="per-GPU score JSONs")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true", help="counts only; prints no measurement")
    a = p.parse_args()
    if a.all:
        v = score_all([json.loads(x.read_text()) for x in a.all])
    else:
        metas = [json.loads(x.read_text()) for x in a.meta]
        if a.check_only:
            ev = [e for m in metas for e in m["events"]]
            print(f"check-only: {len(metas)} runs, kernels {metas[0]['kernels']}, "
                  f"{len({(e['kernel'], e['cond'], e['mode']) for e in ev if e['kind'] == 'loop'})} loop cells, "
                  f"{len({(e['kernel'], e['config']) for e in ev if e['kind'] == 'bench'})} bench cells, "
                  f"correctness_failures={sum(m['correctness_failures'] for m in metas)}, notes {metas[0]['notes']}")
            return 0
        v = score(metas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
