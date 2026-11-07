"""B15 scorer: is do_bench's per-config write-flush penalty explained by bandwidth slack?
Rules K-0..K-3 pre-registered in research/HYPOTHESES.md, section "B15 pre-registration", before any
B15 measurement. Inputs per GPU: one range-mode meta + its ncu raw CSV (app-range, one row per range
in meta order), and the timing-mode metas.

Per shape and config c (medians over repetitions / runs):
  R_c   DRAM read bytes of the kernel after a read flush       (counter)
  E_c   extra DRAM write bytes: writes after zero - writes after read   (counter; the residue)
  t_c   do_bench time with the read flush; dt_c = t(zero) - t(read)     (timing)
  U_c   bandwidth use = (R_c / t_c) / max over configs of (R / t)
  K-0  sanity: every range has counters; >= 80 % configs valid; median read-flush writes <= 0.05 L2;
       >= 3 timing runs
  K-1  (primary) Spearman rho(dt, U) in every shape >= 0.5 -> "penalty grows with bandwidth use";
       <= 0 in every shape -> "no"; otherwise "weak"
  K-2  coefficient of variation of E over configs <= 0.20 in every shape -> "same write-back bytes";
       otherwise "bytes differ"
  K-3  (reported) slack model: dt_pred = max(0, E / BWmax - (t - R / BWmax)); R^2 of dt_pred vs dt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from autotune_bias_score import spearman
from counter_prereg import load_ncu_raw

VALID_MIN, READ_W_MAX, K1_YES, K2_CV = 0.80, 0.05, 0.5, 0.20


def r2(pred: list[float], obs: list[float]) -> float:
    m = statistics.fmean(obs)
    ss_tot = sum((o - m) ** 2 for o in obs)
    ss_res = sum((o - p) ** 2 for p, o in zip(pred, obs))
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def score(rmeta: dict, rrows: list[dict], tmetas: list[dict], log=print) -> dict:
    med = statistics.median
    dev, l2, n_cand = rmeta["device"], rmeta["l2_bytes"], rmeta["n_candidates"]
    rr = [r for r in rmeta["rows"] if r["kind"] == "range"]
    if len(rr) != len(rrows):
        raise ValueError(f"{len(rrows)} ranges in the CSV, {len(rr)} in the meta")
    cnt: dict = {}
    for e, row in zip(rr, rrows):
        d = cnt.setdefault((tuple(e["shape"]), e["cand"]), {"zero": [], "read": []})
        d[e["mode"]].append(row)
    tim: dict = {}
    for m in tmetas:
        for r in m["rows"]:
            d = tim.setdefault((tuple(r["shape"]), r["cand"]), {"zero": [], "read": []})
            d["zero"].append(r["zero_ms"])
            d["read"].append(r["read_ms"])
    shapes = sorted({s for s, _ in cnt})
    ok0 = len(tmetas) >= 3
    per, rhos, cvs = {}, [], []
    for s in shapes:
        cands = [c for (ss, c) in cnt if ss == s and (ss, c) in tim]
        ok0 &= len(cands) >= VALID_MIN * n_cand
        R = {c: med(x["read"] for x in cnt[(s, c)]["read"]) for c in cands}
        Wr = {c: med(x["write"] for x in cnt[(s, c)]["read"]) for c in cands}
        E = {c: med(x["write"] for x in cnt[(s, c)]["zero"]) - Wr[c] for c in cands}
        t = {c: med(tim[(s, c)]["read"]) for c in cands}
        dt = {c: med(tim[(s, c)]["zero"]) - t[c] for c in cands}
        bw = {c: R[c] / t[c] for c in cands}
        bwmax = max(bw.values())
        U = {c: bw[c] / bwmax for c in cands}
        ok0 &= med(Wr.values()) <= READ_W_MAX * l2
        rho = spearman([dt[c] for c in cands], [U[c] for c in cands])
        ev = list(E.values())
        cv = statistics.pstdev(ev) / statistics.fmean(ev) if statistics.fmean(ev) > 0 else float("inf")
        pred = {c: max(0.0, E[c] / bwmax - (t[c] - R[c] / bwmax)) for c in cands}
        rhos.append(rho)
        cvs.append(cv)
        top = sorted(cands, key=lambda c: -U[c])
        per[f"{s[0]}x{s[1]}"] = {
            "n": len(cands), "rho_dt_U": round(rho, 3), "cv_E": round(cv, 3),
            "E_median_over_L2": round(med(ev) / l2, 4), "r2_slack_model": round(r2(list(pred.values()), list(dt.values())), 3),
            "most_bw_used": [(c, round(U[c], 3), round(dt[c] * 1e3, 1)) for c in top[:3]],
            "least_bw_used": [(c, round(U[c], 3), round(dt[c] * 1e3, 1)) for c in top[-3:]]}
    k1 = ("penalty grows with bandwidth use" if all(r >= K1_YES for r in rhos) else
          "no" if all(r <= 0 for r in rhos) else "weak")
    k2 = "same write-back bytes" if all(cv <= K2_CV for cv in cvs) else "bytes differ"
    v = {"device": dev, "K-0": ok0, "K-1": k1, "K-2": k2, "per shape": per}
    log(f"{dev}: K-0 {'PASS' if ok0 else 'FAIL'}")
    for k, p in per.items():
        log(f"  {k}: rho(dt, U) {p['rho_dt_U']:+.3f}; CV(E) {p['cv_E']:.3f}; E median {p['E_median_over_L2']:.3f} L2; "
            f"slack-model R^2 {p['r2_slack_model']:.3f}; most BW (cfg, U, dt us) {p['most_bw_used']}; "
            f"least {p['least_bw_used']}")
    log(f"K-1 -> {k1}; K-2 -> {k2}")
    log("verdict: " + json.dumps({k: v[k] for k in ("K-0", "K-1", "K-2")}))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--range-meta", type=Path, required=True)
    p.add_argument("--range-csv", type=Path, required=True)
    p.add_argument("--timing-meta", type=Path, nargs="+", required=True)
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    v = score(json.loads(a.range_meta.read_text()), load_ncu_raw(a.range_csv),
              [json.loads(m.read_text()) for m in a.timing_meta])
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
