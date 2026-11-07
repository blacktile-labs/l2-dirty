"""B18 scorer: which L2-flush fixes work, and who pays for a kernel's own writes?
Rules H-0..H-4 pre-registered in research/HYPOTHESES.md, section "B18 pre-registration", before any B18
measurement. Inputs: the probe's range-mode metas + ncu raw CSVs (app-range: one row per range, in event
order) and its timing-mode metas.

Per (kernel k, mode m): W[k,m], R[k,m] = median DRAM write / read bytes of the timed-call range.
  w = W / L2      r = R / input bytes      ow = W / output bytes
  H-0  gate: correctness 0 (kernels, PR flush unchanged buffer, no st.global); control ranges write
       <= 64 KiB; gemv4 after pr256 writes <= 0.05 L2 and reads 0.9-1.2 x its weight; e_large after pr256
       reads 0.9-1.2 x its input. H-0e: X = w(gemv4, zero256) - w(gemv4, pr256) >= 0.05, else H-1/H-2
       are not scorable (no write-flush residue on this GPU)
  H-1  (W(gemv4, zero256_wait) - W(gemv4, pr256)) / (W(gemv4, zero256) - W(gemv4, pr256))
       >= 0.7 "waiting does not remove the residue"; <= 0.3 "waiting removes it"; else "partial"
  H-2  w(gemv4, zero256_read) - w(gemv4, pr256): <= 0.02 "write-then-read removes the residue";
       >= 0.05 "does not"; else "partial"
  H-3  needs r(e_tiny, pr256) >= 0.9 and r(e_small, pr256) >= 0.9
       a: r(e_tiny, prL2) <= 0.6 "an L2-sized read flush leaves a tiny kernel warm"; >= 0.9 "sufficient"
       b: r(e_small, prL2) >= 0.9 "sufficient at footprint 0.25 L2"; <= 0.6 "leaves it warm"
       c: min r(e_tiny|e_small, pr2L2) >= 0.9 "a 2 x L2 read flush suffices"; else "not"
  H-4  a: ow(e_small, pr256) <= 0.1 "a read flush keeps a small kernel's own write-back out of its window";
          >= 0.5 "charges it"; else "partial"
       b: ow(e_small, rotate) >= 0.7 "rotation charges about one output per call"; <= 0.3 "does not"
       c: ow(e_large, pr256) >= 0.6 "a kernel with output > L2/2 pays most of its own write-back"; <= 0.3 "does not"
  reported: every w / r / ow; zeroL2 and sum256 against pr256; timing medians, means and bias vs pr256
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from counter_prereg import load_ncu_raw

CTL_MAX = 64 << 10
READ_W_MAX, READ_R_BAND = 0.05, (0.9, 1.2)
X_MIN = 0.05
H1_HI, H1_LO = 0.7, 0.3
H2_OK, H2_BAD = 0.02, 0.05
COLD, WARM = 0.9, 0.6
H4A_OK, H4A_BAD = 0.1, 0.5
H4B_OK, H4B_BAD = 0.7, 0.3
H4C_OK, H4C_BAD = 0.6, 0.3


def range_rows(metas: list[dict], rows_per_run: list[list[dict]]) -> list[tuple[dict, dict]]:
    out = []
    for m, rows in zip(metas, rows_per_run):
        if len(rows) != len(m["events"]):
            raise ValueError(f"{len(rows)} ranges in the CSV, {len(m['events'])} events in the meta")
        out += list(zip(m["events"], rows))
    return out


def three_way(x: float, hi: float, lo: float, hi_txt: str, lo_txt: str, higher_is_hi: bool = True) -> str:
    if higher_is_hi:
        return hi_txt if x >= hi else lo_txt if x <= lo else "partial"
    return hi_txt if x <= hi else lo_txt if x >= lo else "partial"


def score(rmetas: list[dict], rrows: list[tuple[dict, dict]], tmetas: list[dict], log=print) -> dict:
    med = statistics.median
    l2, dev = rmetas[0]["l2_bytes"], rmetas[0]["device"]
    kinfo = rmetas[0]["kernels"]
    fails = sum(m["correctness_failures"] for m in rmetas + tmetas)
    pr_ok = all(m["pr_flush_check"].get("buffer_unchanged") and m["pr_flush_check"].get("st_global", 1) == 0
                for m in rmetas + tmetas)
    ctl = [r["write"] for e, r in rrows if e["kind"] == "control"]
    cells: dict = {}
    for e, r in rrows:
        if e["kind"] == "call":
            cells.setdefault((e["kernel"], e["mode"]), []).append(r)
    W = {c: med(x["write"] for x in rows) for c, rows in cells.items()}
    R = {c: med(x["read"] for x in rows) for c, rows in cells.items()}
    n = {c: len(rows) for c, rows in cells.items()}

    def w(k, m): return W[(k, m)] / l2
    def r(k, m): return R[(k, m)] / kinfo[k]["in_bytes"]
    def ow(k, m): return W[(k, m)] / kinfo[k]["out_bytes"]

    v: dict = {"device": dev, "l2_bytes": l2}
    g0 = {"correctness": fails == 0 and pr_ok, "control writes": max(ctl) <= CTL_MAX,
          "gemv4 after pr256 leaves no dirty data": w("gemv4", "pr256") <= READ_W_MAX,
          "gemv4 after pr256 reads its weight": READ_R_BAND[0] <= r("gemv4", "pr256") <= READ_R_BAND[1],
          "e_large after pr256 reads its input": READ_R_BAND[0] <= r("e_large", "pr256") <= READ_R_BAND[1]}
    v["H-0"], v["H-0 parts"] = all(g0.values()), g0
    log(f"{dev}, L2 {l2} B; H-0 {g0} -> {'PASS' if v['H-0'] else 'FAIL'}")
    X = w("gemv4", "zero256") - w("gemv4", "pr256")
    v["X (residue / L2)"] = round(X, 4)
    v["H-0e"] = X >= X_MIN
    log(f"H-0e X = w(zero256) - w(pr256) = {X:.4f} (>= {X_MIN}: {v['H-0e']})")
    if not v["H-0"]:
        for h in ("H-1", "H-2", "H-3a", "H-3b", "H-3c", "H-4a", "H-4b", "H-4c"):
            v[h] = "not scored (H-0 failed)"
    else:
        if v["H-0e"]:
            base = W[("gemv4", "zero256")] - W[("gemv4", "pr256")]
            h1 = (W[("gemv4", "zero256_wait")] - W[("gemv4", "pr256")]) / base
            v["H-1 ratio"] = round(h1, 4)
            v["H-1"] = three_way(h1, H1_HI, H1_LO, "waiting does not remove the residue", "waiting removes the residue")
            h2 = w("gemv4", "zero256_read") - w("gemv4", "pr256")
            v["H-2 d"] = round(h2, 4)
            v["H-2"] = three_way(h2, H2_OK, H2_BAD, "write-then-read removes the residue", "write-then-read does not",
                                 higher_is_hi=False)
        else:
            v["H-1"] = v["H-2"] = "not scorable (no write-flush residue on this GPU)"
        log(f"H-1 ratio {v.get('H-1 ratio')} -> {v['H-1']}; H-2 d {v.get('H-2 d')} -> {v['H-2']}")
        if r("e_tiny", "pr256") >= COLD and r("e_small", "pr256") >= COLD:
            a = r("e_tiny", "prL2")
            v["H-3a r"] = round(a, 4)
            v["H-3a"] = three_way(a, WARM, COLD, "an L2-sized read flush leaves a tiny kernel warm", "sufficient",
                                  higher_is_hi=False)
            b = r("e_small", "prL2")
            v["H-3b r"] = round(b, 4)
            v["H-3b"] = three_way(b, COLD, WARM, "sufficient at footprint 0.25 L2", "leaves it warm")
            c = min(r("e_tiny", "pr2L2"), r("e_small", "pr2L2"))
            v["H-3c min r"] = round(c, 4)
            v["H-3c"] = "a 2 x L2 read flush suffices" if c >= COLD else "not"
        else:
            v["H-3a"] = v["H-3b"] = v["H-3c"] = "not scorable (pr256 does not cool the small kernels)"
        log(f"H-3a r {v.get('H-3a r')} -> {v['H-3a']}; H-3b r {v.get('H-3b r')} -> {v['H-3b']}; "
            f"H-3c {v.get('H-3c min r')} -> {v['H-3c']}")
        a4 = ow("e_small", "pr256")
        v["H-4a ow"] = round(a4, 4)
        v["H-4a"] = three_way(a4, H4A_OK, H4A_BAD, "a read flush keeps a small kernel's own write-back out of its window",
                              "charges it", higher_is_hi=False)
        b4 = ow("e_small", "rotate")
        v["H-4b ow"] = round(b4, 4)
        v["H-4b"] = three_way(b4, H4B_OK, H4B_BAD, "rotation charges about one output per call", "does not")
        c4 = ow("e_large", "pr256")
        v["H-4c ow"] = round(c4, 4)
        v["H-4c"] = three_way(c4, H4C_OK, H4C_BAD, "a kernel with output > L2/2 pays most of its own write-back",
                              "does not")
        log(f"H-4a ow {a4:.4f} -> {v['H-4a']}; H-4b ow {b4:.4f} -> {v['H-4b']}; H-4c ow {c4:.4f} -> {v['H-4c']}")
    table = {}
    for (k, m) in sorted(cells):
        table[f"{k} {m}"] = {"w (write/L2)": round(w(k, m), 4), "r (read/input)": round(r(k, m), 4),
                             "ow (write/output)": round(ow(k, m), 4), "n": n[(k, m)]}
    v["cells"] = table
    v["zeroL2 residue / L2 (gemv4, descriptive)"] = round(w("gemv4", "zeroL2") - w("gemv4", "pr256"), 4)
    v["sum256 - pr256 write / L2 (gemv4, descriptive)"] = round(w("gemv4", "sum256") - w("gemv4", "pr256"), 4)
    tim: dict = {}
    for m in tmetas:
        for e in m["events"]:
            if e["kind"] == "timing":
                tim.setdefault((e["kernel"], e["mode"]), []).extend(e["times_us"])
    tv = {}
    for k in sorted({k for k, _ in tim}):
        ref = med(tim[(k, "pr256")]) if (k, "pr256") in tim else None
        tv[k] = {}
        for mode in sorted({mm for kk, mm in tim if kk == k}):
            ts = tim[(k, mode)]
            tv[k][mode] = {"median_us": round(med(ts), 3), "mean_us": round(statistics.fmean(ts), 3),
                           "bias_vs_pr256_median": round(med(ts) / ref - 1, 4) if ref else None, "n": len(ts)}
    v["timing (descriptive)"] = tv
    log("cells: " + json.dumps(table))
    log("timing: " + json.dumps(tv))
    log("verdict: " + json.dumps({h: v.get(h) for h in ("H-0", "H-0e", "H-1", "H-2", "H-3a", "H-3b", "H-3c",
                                                         "H-4a", "H-4b", "H-4c")}))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--range-meta", type=Path, nargs="+", required=True)
    p.add_argument("--range-csv", type=Path, nargs="+", required=True)
    p.add_argument("--timing-meta", type=Path, nargs="+", required=True)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--check-only", action="store_true", help="row counts only; prints no measurement")
    a = p.parse_args()
    rmetas = [json.loads(x.read_text()) for x in a.range_meta]
    tmetas = [json.loads(x.read_text()) for x in a.timing_meta]
    rrows = range_rows(rmetas, [load_ncu_raw(c) for c in a.range_csv])
    if a.check_only:
        calls = sum(1 for e, _ in rrows if e["kind"] == "call")
        modes = {e["mode"] for e, _ in rrows if e["kind"] == "call"}
        ks = {e["kernel"] for e, _ in rrows if e["kind"] == "call"}
        tcells = {(e["kernel"], e["mode"]) for m in tmetas for e in m["events"] if e["kind"] == "timing"}
        fails = sum(m["correctness_failures"] for m in rmetas + tmetas)
        print(f"check-only: {len(rrows)} ranges aligned ({calls} calls, {len(ks)} kernels, {len(modes)} modes), "
              f"{len(tcells)} timing cells, correctness_failures={fails}")
        return 0
    v = score(rmetas, rrows, tmetas)
    if a.json_out:
        a.json_out.write_text(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
