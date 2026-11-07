"""B17 scorer: do_bench's write flush vs the read flush of triton-lang/triton PR #12556.
Rules E-0, F-1..F-6 pre-registered in research/HYPOTHESES.md, section "B17 pre-registration", before any
B17 measurement. Medians over runs per (family, case, cand, mode); B = t_zero / t_pr - 1.

Per GPU (score_gpu, input: the probe metas of one GPU):
  E-0  validity: >= 4 runs; the PR flush's PTX has >= 1 volatile global load and no global store and
       left its buffer unchanged; on GPUs with L2 >= 64 MiB the PR flush evicts: t_pr / t_none >= 1.2
       for cublas_fp16 on gemv 4096x4096 (32 MiB of weights, L2-resident without a flush)
  F-1  B_ref = B of cublas_fp16 on gemv 14336x4096 (112 MiB, larger than every L2 here):
       >= 0.05 "affected", <= 0.02 "not affected", else "marginal"
  F-3  D_q = S_q,zero / S_q,pr - 1, S_q = t(cublas_fp16) / t(q), q in {tinygemm (best of k2/k4/k8 per
       mode), bnb_nf4}, over the gemv shapes: worst |D| >= 0.10 "distorted", <= 0.03 "faithful",
       else "mildly distorted" (+ sign)
  F-4  per family in {attn, rmsnorm, swiglu}: B at the family's largest size, max over candidates:
       >= 0.05 "affected", <= 0.02 "not affected", else "marginal"
  F-6  (reported) R = t_sum / t_pr - 1 (our B12-B16 read flush vs the PR flush), gemv cases with
       t_pr >= 38 us: median and max |R|
  F-7  (B17a amendment) L2-sized write flush (Inductor / nvbench / CUTLASS): B_l2 = t_l2w / t_pr - 1 for
       the reference case; B_l2 - B_ref >= 0.02 "biases more than do_bench", <= -0.02 "less", else
       "similar"; B_l2 also reported for every case
  E-0' (B17b amendment, written after seeing ONLY the 4080 SUPER result, before any other B17 GPU result):
       (a) and (b) as E-0; (c') replaces the residency clause: the PR flush leaves cublas_fp16 on gemv
       4096x4096 at least as cold as the sum flush (validated with DRAM counters in B12 G-0):
       t_pr >= t_sum - max(0.03 t_sum, 1.1 us), on every GPU. Rules are scored under E-0 and E-0'.
Across GPUs (score_all, input: the score_gpu outputs):
  F-2  Spearman rho(L2 size, B_ref) over the Ada GPUs (cc 8.9) that pass E-0, needs >= 4:
       >= 0.8 "scales with L2 within Ada", <= 0.4 "not explained by L2 size", else "mixed"
  F-5  4080 SUPER replication on a second machine, with the sum flush as in B16:
       |B_sum(cublas_fp16, 14336x4096) - 0.1661| <= 0.05 and |D_sum(tinygemm, 14336x4096) - (-0.1696)| <= 0.05
       -> "replicates", else "differs"
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REF = ("gemv", "14336x4096", "cublas_fp16")
CTRL = ("gemv", "4096x4096", "cublas_fp16")
GEMV = ["4096x4096", "14336x4096", "4096x14336"]
FAMILY_LAST = {"attn": "L65536", "rmsnorm": "T8192", "swiglu": "T2048"}
B16_4080S = {"B_sum": 0.1661, "D_sum": -0.1696}
TICK_US = 1.024


def verdict3(x, hi, lo, names):
    return names[0] if x >= hi else names[1] if x <= lo else names[2]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def score_gpu(metas: list, log=print) -> dict:
    t: dict = {}
    for m in metas:
        for r in m["rows"]:
            for mode, ms in r["ms"].items():
                t.setdefault((r["family"], r["case"], r["cand"]), {}).setdefault(mode, []).append(ms)
    need = max(1, len(metas) - 1)                      # present in all runs but at most one
    med = {k: {m: statistics.median(v) for m, v in d.items() if len(v) >= need} for k, d in t.items()}
    m0 = metas[0]
    l2 = m0["l2_bytes"]
    ptx = m0.get("ptx_check", {})
    e0 = {"runs": len(metas), "runs_ok": len(metas) >= 4,
          "ptx_ok": ptx.get("ld_volatile_global", 0) >= 1 and ptx.get("st_global", 1) == 0
          and ptx.get("buffer_unchanged") is True}
    c = med.get(CTRL, {})
    if l2 >= 64 << 20:
        ev = c["pr"] / c["none"] if "pr" in c and "none" in c else None
        e0["evict_ratio"] = round(ev, 3) if ev else None
        e0["evict_ok"] = bool(ev and ev >= 1.2)
    else:
        e0["evict_ratio"] = round(c["pr"] / c["none"], 3) if "pr" in c and "none" in c else None
        e0["evict_ok"] = None                          # not applicable below 64 MiB
    e0["pass"] = e0["runs_ok"] and e0["ptx_ok"] and e0["evict_ok"] is not False
    if "pr" in c and "sum" in c:
        e0["cold_as_sum_ok"] = c["pr"] >= c["sum"] - max(0.03 * c["sum"], 1.1e-3)
    else:
        e0["cold_as_sum_ok"] = False
    e0["pass_amended"] = e0["runs_ok"] and e0["ptx_ok"] and e0["cold_as_sum_ok"]
    bias = {f"{k[0]} {k[1]} {k[2]}": round(v["zero"] / v["pr"] - 1, 4) for k, v in med.items()
            if "zero" in v and "pr" in v}
    v: dict = {"device": m0["device"], "l2_bytes": l2, "cc": m0["cc"], "versions": m0.get("versions"),
               "E-0": e0, "bias": bias}
    b_ref = bias.get(" ".join(REF))
    v["B_ref"] = b_ref
    v["F-1"] = verdict3(b_ref, 0.05, 0.02, ("affected", "not affected", "marginal")) if b_ref is not None else "not measured"
    # F-3 quantization speedups, pr baseline (and the same with the sum baseline for F-5)
    ds, dsum, per = [], {}, {}
    for s in GEMV:
        base = med.get(("gemv", s, "cublas_fp16"), {})
        tiny = [k for k in med if k[0] == "gemv" and k[1] == s and k[2].startswith("tinygemm_")]
        bnb = ("gemv", s, "bnb_nf4")
        p = {}
        for q, keys in (("tinygemm", tiny), ("bnb_nf4", [bnb] if bnb in med else [])):
            keys = [k for k in keys if all(m in med[k] for m in ("zero", "pr", "sum"))]
            if not keys or not all(m in base for m in ("zero", "pr", "sum")):
                continue
            S = {m: base[m] / min(med[k][m] for k in keys) for m in ("zero", "pr", "sum")}
            d = S["zero"] / S["pr"] - 1
            ds.append(d)
            p[q] = {"S_zero": round(S["zero"], 3), "S_pr": round(S["pr"], 3), "D": round(d, 4),
                    "D_sum": round(S["zero"] / S["sum"] - 1, 4)}
            dsum[(s, q)] = S["zero"] / S["sum"] - 1
        per[s] = p
    worst = max(ds, key=abs) if ds else None
    v["F-3"] = (verdict3(abs(worst), 0.10, 0.03, ("distorted", "faithful", "mildly distorted"))
                if worst is not None else "not measured")
    v["F-3 worst D"] = round(worst, 4) if worst is not None else None
    v["F-3 sign"] = ("under-estimated" if ds and all(d < 0 for d in ds) else
                     "over-estimated" if ds and all(d > 0 for d in ds) else "mixed" if ds else None)
    v["F-3 per shape"] = per
    # F-4 other families
    f4 = {}
    for fam, last in FAMILY_LAST.items():
        bs = {k[2]: bias[f"{fam} {last} {k[2]}"] for k in med if k[0] == fam and k[1] == last
              and f"{fam} {last} {k[2]}" in bias}
        if bs:
            mx = max(bs.values())
            f4[fam] = {"case": last, "B": bs, "verdict": verdict3(mx, 0.05, 0.02, ("affected", "not affected", "marginal"))}
        else:
            f4[fam] = {"case": last, "verdict": "not measured"}
    v["F-4"] = f4
    # F-6 sum flush vs pr flush
    rs = {f"{k[1]} {k[2]}": round(x["sum"] / x["pr"] - 1, 4) for k, x in med.items()
          if k[0] == "gemv" and "sum" in x and "pr" in x and x["pr"] * 1e3 >= 38}
    v["F-6"] = {"median R": round(statistics.median(rs.values()), 4) if rs else None,
                "max |R|": round(max(map(abs, rs.values())), 4) if rs else None, "R": rs}
    # F-7 L2-sized write flush
    bl2 = {f"{k[0]} {k[1]} {k[2]}": round(x["l2w"] / x["pr"] - 1, 4) for k, x in med.items() if "l2w" in x and "pr" in x}
    b_l2 = bl2.get(" ".join(REF))
    v["B_l2"] = b_l2
    v["F-7"] = ("not measured" if b_l2 is None or b_ref is None else
                "biases more than do_bench" if b_l2 - b_ref >= 0.02 else
                "biases less than do_bench" if b_l2 - b_ref <= -0.02 else "similar")
    v["bias_l2w"] = bl2
    # F-5 inputs (sum baseline), timer grid
    bsum = med.get(REF, {})
    v["F-5 inputs"] = {"B_sum": round(bsum["zero"] / bsum["sum"] - 1, 4) if "zero" in bsum and "sum" in bsum else None,
                       "D_sum": round(dsum[("14336x4096", "tinygemm")], 4) if ("14336x4096", "tinygemm") in dsum else None}
    vals = [ms * 1e3 for d in med.values() for ms in d.values()]
    v["tick fraction"] = round(sum(abs(x / TICK_US - round(x / TICK_US)) < 0.01 for x in vals) / len(vals), 3) if vals else None
    v["t_us"] = {" ".join(k): {m: round(x * 1e3, 2) for m, x in d.items()} for k, d in med.items()}
    v["invalid"] = sorted({f"{r['family']} {r['case']} {r['cand']}" for m in metas for r in m.get("invalid", [])})
    v["notes"] = m0.get("notes", {})
    log(f"{v['device']} (L2 {l2 >> 20} MiB, cc {v['cc']}): E-0 {'PASS' if e0['pass'] else 'FAIL'}, "
        f"E-0' {'PASS' if e0['pass_amended'] else 'FAIL'} {e0}")
    log(f"  F-1 B_ref {b_ref} -> {v['F-1']}")
    log(f"  F-3 worst D {v['F-3 worst D']} -> {v['F-3']} ({v['F-3 sign']}); per shape {per}")
    log(f"  F-4 {{fam: (B, verdict)}}: { {f: (x.get('B'), x['verdict']) for f, x in f4.items()} }")
    log(f"  F-6 sum vs pr flush: {v['F-6']}")
    log(f"  F-7 B_l2 {b_l2} vs B_ref {b_ref} -> {v['F-7']}")
    log(f"  F-5 inputs {v['F-5 inputs']}; tick fraction {v['tick fraction']}; invalid {v['invalid']}")
    log("verdict: " + json.dumps({k: v[k] for k in ("device", "B_ref", "F-1", "F-3", "F-3 worst D", "F-3 sign", "B_l2", "F-7")}
                                 | {"E-0": e0["pass"]}))
    return v


def f2(gpus, key):
    ada = [g for g in gpus if g["cc"] == "8.9" and g["E-0"].get(key) and g["B_ref"] is not None]
    r = {"ada": [(g["device"], g["l2_bytes"] >> 20, g["B_ref"]) for g in ada]}
    if len(ada) >= 4:
        rho = spearman([g["l2_bytes"] for g in ada], [g["B_ref"] for g in ada])
        r["rho"] = round(rho, 3)
        r["verdict"] = ("scales with L2 within Ada" if rho >= 0.8 else
                        "not explained by L2 size" if rho <= 0.4 else "mixed")
    else:
        r["verdict"] = f"not enough Ada GPUs ({len(ada)} < 4)"
    return r


def score_all(gpus: list, log=print) -> dict:
    locked, amended = f2(gpus, "pass"), f2(gpus, "pass_amended")
    out: dict = {"n_gpus": len(gpus), "ada": locked["ada"], "F-2": locked["verdict"], "F-2 rho": locked.get("rho"),
                 "F-2 (E-0')": amended["verdict"], "F-2 rho (E-0')": amended.get("rho"), "ada (E-0')": amended["ada"]}
    rep = [g for g in gpus if "4080 SUPER" in g["device"]]
    if rep and rep[0]["F-5 inputs"]["B_sum"] is not None and rep[0]["F-5 inputs"]["D_sum"] is not None:
        x = rep[0]["F-5 inputs"]
        ok = abs(x["B_sum"] - B16_4080S["B_sum"]) <= 0.05 and abs(x["D_sum"] - B16_4080S["D_sum"]) <= 0.05
        out["F-5"] = "replicates" if ok else "differs"
        out["F-5 detail"] = {"B17": x, "B16": B16_4080S}
    else:
        out["F-5"] = "not measured"
    out["residue / L2 (B_ref x 112 MiB / L2)"] = {g["device"]: round(g["B_ref"] * 14336 * 4096 * 2 / g["l2_bytes"], 3)
                                                 for g in gpus if g["B_ref"] is not None}
    log(f"F-2 (E-0 locked): {out['F-2']} (rho {out.get('F-2 rho')}; Ada {out['ada']})")
    ka, kr, kd = "F-2 (E-0')", "F-2 rho (E-0')", "ada (E-0')"
    log(f"F-2 (E-0' amended): {out[ka]} (rho {out.get(kr)}; Ada {out[kd]})")
    log(f"F-5: {out['F-5']} {out.get('F-5 detail', '')}")
    log(f"residue / L2: {out['residue / L2 (B_ref x 112 MiB / L2)']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, nargs="+", help="probe metas of ONE GPU")
    ap.add_argument("--all", type=Path, nargs="+", help="score_gpu JSON outputs of several GPUs")
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    if a.meta:
        v = score_gpu([json.loads(m.read_text()) for m in a.meta])
    elif a.all:
        v = score_all([json.loads(m.read_text()) for m in a.all])
    else:
        ap.error("--meta or --all")
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
