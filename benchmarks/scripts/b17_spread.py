#!/usr/bin/env python3
"""Run-to-run spread of the B17 headline numbers (post hoc, descriptive; D-088).

For every B17 GPU directory, per run: B = t(mode) / t(pr) - 1 from triton.testing.do_bench medians, for the
reference GEMV (cuBLAS fp16, 14336 x 4096) and the Liger RMSNorm cases; reports median, min and max over runs.
The locked B17 scores use the ratio of median times over runs, so this "median" (of per-run ratios) is close to
them but not identical; the report quotes the locked score and uses min/max from here.
"""
import glob
import json
import statistics
import sys

CASES = [("gemv", "14336x4096", "cublas_fp16", "zero"), ("gemv", "14336x4096", "cublas_fp16", "l2w"),
         ("rmsnorm", "T8192", "liger_rmsnorm", "zero"), ("rmsnorm", "T2048", "liger_rmsnorm", "zero"),
         ("rmsnorm", "T2048", "liger_rmsnorm", "l2w")]


def main() -> int:
    out = {}
    for d in sorted(glob.glob("benchmarks/results/b17_*_20261106")):
        metas = sorted(glob.glob(f"{d}/benchmarks/results/flush_bias_2*/run_r*_meta.json"))
        if not metas:
            continue
        runs = [json.load(open(p)) for p in metas]
        dev = runs[0]["device"]
        res = {}
        for fam, case, cand, mode in CASES:
            bs = []
            for r in runs:
                for row in r["rows"]:
                    if row["family"] == fam and row["case"] == case and row["cand"] == cand and mode in row["ms"]:
                        bs.append(row["ms"][mode] / row["ms"]["pr"] - 1)
            if bs:
                res[f"{fam} {case} {cand} {mode}"] = {"median": round(statistics.median(bs), 4), "min": round(min(bs), 4),
                                                      "max": round(max(bs), 4), "n": len(bs)}
        out[f"{d.split('/')[-1]} ({dev})"] = res
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
