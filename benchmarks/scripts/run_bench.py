#!/usr/bin/env python3
"""Run, record and summarize the B2-B4 baselines (transpose, gather, fusion).

  python3 benchmarks/scripts/run_bench.py transpose --smoke --dry-run   # plan + ESTIMATES only
  python3 benchmarks/scripts/run_bench.py gather --smoke                # needs an NVIDIA GPU
  python3 benchmarks/scripts/run_bench.py fusion --summarize-only benchmarks/results/fusion/<run>
  python3 benchmarks/scripts/run_bench.py transpose --smoke -- --shapes 2048x2048   # extra binary args

Each run writes benchmarks/results/<bench>/<run_id>/{env.json, reps.csv, meta.json,
stdout.log, summary.md}. Columns prefixed est_/pred_ are MODEL ESTIMATES (bench_common.py);
everything else is measured or derived from measurements. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import bench_common as bc
from harness_env import ROOT, capture_env, nvidia_smi_snapshot

BENCHES = ("transpose", "gather", "fusion", "stencil")
RUN_TIMEOUT_S = 900
ASSUMED_L2 = 2_359_296  # dry-run only: L2 attribute measured on the RTX 3060 (2026-11-04)


def med_us(s: bc.Series) -> float:
    return statistics.median(s.times_ms) * 1e3


def fmt(v, nd=3) -> str:
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.{nd}f}"
    return str(v)


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def dry_run(bench: str, smoke: bool) -> None:
    print(f"DRY RUN ({bench}) -- no GPU touched. Numbers are MODEL ESTIMATES, not measurements.")
    if bench == "transpose":
        shapes = [(4096, 4096), (1000, 777)] if smoke else [(1024, 1024), (2048, 2048), (4096, 4096),
                                                             (8192, 8192), (16384, 1024), (1024, 16384),
                                                             (4097, 4095)]
        rows = []
        for R, C in shapes:
            for v in ("copy_2d", "naive", "tiled", "tiled_pad"):
                e = bc.model_transpose(R, C, v)
                rows.append([f"{R}x{C}", v, e["dram_bytes"] / 2**20, e["l2_store_sectors"], e["smem_conflict_ways"]])
        print("\n".join(table(["shape", "variant", "est_dram_MiB", "est_l2_store_sectors", "est_smem_conflict_ways"], rows)))
    elif bench == "fusion":
        factors = [0.25, 0.5, 1, 4, 16] if smoke else [0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 64]
        print(f"(assuming L2 = {ASSUMED_L2} B, the RTX 3060 attribute; the binary uses the real value)")
        rows = []
        for f in factors:
            n = max(256, int(f * ASSUMED_L2 / 4) // 256 * 256)
            fu = bc.model_fusion_bytes(n, ASSUMED_L2, "fused")["dram"]
            for m in ("lru_cliff", "proportional"):
                ma = bc.model_fusion_bytes(n, ASSUMED_L2, "materialized", m)["dram"]
                rows.append([f, n, m, fu / 2**20, ma / 2**20, ma / fu])
        print("\n".join(table(["l2x", "n", "residency_model", "est_dram_MiB_fused", "est_dram_MiB_mat",
                               "est_mat/fused (DRAM-bound)"], rows)))
    elif bench == "stencil":
        print("stencil: see benchmarks/scripts/b5_model.py for the B5 model (pre-registered predictions).")
    else:
        print("gather: random-index estimates need the index statistics the binary computes "
              "(meta.json -> index_stats); only the bounds below are available up front.")
        m, t = (4 << 20, 16 << 20) if smoke else (16 << 20, 16 << 20)
        print(f"M={m} lookups, T={t} elements: no_reuse table bytes = {bc.SECTOR * m / 2**20:.0f} MiB, "
              f"seq table bytes = {4 * min(m, t) / 2**20:.0f} MiB")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def common_rows(series: list[bc.Series]) -> list[str]:
    rows = []
    for s in series:
        st = bc.summarize(s.times_ms)
        gbps = s.logical_bytes / (st.median * 1e-3) / 1e9 if s.logical_bytes else 0.0
        rows.append([s.variant, s.params_raw, s.timing_mode, st.n, st.median * 1e3, st.p95 * 1e3,
                     st.cv * 100, gbps])
    return table(["variant", "params", "mode", "reps", "median_us", "p95_us", "cv_pct", "useful_GBps"], rows)


def summarize_transpose(series, meta) -> list[str]:
    by = {(s.variant, s.params_raw, s.timing_mode): s for s in series}
    rows = []
    for (v, p, mode), s in by.items():
        ref = by.get(("copy_2d", p, mode))
        if v == "copy_2d" or ref is None:
            continue
        R, C = s.param("R", int), s.param("C", int)
        e = bc.model_transpose(R, C, v)
        rows.append([p, mode, v, med_us(s), med_us(ref), 100 * med_us(ref) / med_us(s),
                     e["l2_store_sectors"] / bc.model_transpose(R, C, "copy_2d")["l2_store_sectors"],
                     e["smem_conflict_ways"]])
    return ["", "### Relative to copy_2d (same shape, same timing mode)", "",
            "`pct_of_copy` = copy time / variant time x 100 (MEASURED ratio). `est_*` are model estimates.", ""] + \
        table(["shape", "mode", "variant", "median_us", "copy_us", "pct_of_copy", "est_l2_store_sectors_vs_copy",
               "est_smem_conflict_ways"], rows)


def summarize_gather(series, meta, peak_gbps) -> list[str]:
    stats = meta.get("index_stats", {})
    l2 = meta.get("l2_cache_bytes", 0)
    rows = []
    for s in series:
        if s.timing_mode != "isolated":
            continue
        p = s.params
        m, t, dist = int(p["M"]), int(p["T"]), p["dist"]
        ist = stats.get(s.params_raw, {})
        try:
            tb = bc.model_gather_table_bytes(m, t, dist, l2, ist.get("distinct_sectors"), ist.get("hot_sector_share"))
        except ValueError:
            continue
        if s.variant not in ("gather", "scatter_add"):
            continue
        tot = bc.model_gather_bytes(m, tb) if s.variant == "gather" else bc.model_scatter_add_bytes(m, tb)
        sec = med_us(s) * 1e-6
        implied = {k: v / sec / 1e9 for k, v in tot.items()}
        rows.append([s.variant, dist, f"{t}", med_us(s), ist.get("hot_sector_share", float("nan")),
                     ist.get("max_element_count", float("nan")), tot["compulsory"] / 2**20, tot["hot_set"] / 2**20,
                     tot["no_reuse"] / 2**20, implied["compulsory"], implied["no_reuse"],
                     "YES" if implied["compulsory"] > peak_gbps else "no"])
    # fetch-granularity probe: pair / single time ratios vs. what each granularity predicts
    by = {(s.variant, s.params_raw, s.timing_mode): s for s in series}
    probe = []
    for s in series:
        if s.variant not in bc.PAIR_FLIPS or s.timing_mode != "isolated":
            continue
        single = by.get(("gather", s.params_raw, "isolated"))
        if single is None:
            continue
        probe.append([s.variant, s.params["dist"], med_us(s), med_us(single), med_us(s) / med_us(single)] +
                     [bc.predicted_pair_ratio(s.variant, g) for g in (32, 64, 128)])
    probe_md = []
    if probe:
        probe_md = ["", "### Fetch-granularity probe (isolated mode)", "",
                    f"`meas_ratio` = pair-variant time / gather time (MEASURED). `pred_g*` = ratio expected if "
                    f"DRAM moves g-byte chunks per miss with no reuse (ESTIMATE; meaningful for random dists). "
                    f"l2_fetch_granularity reported by driver: {meta.get('l2_fetch_granularity_bytes')} B "
                    f"(requested: {meta.get('l2_fetch_granularity_requested', -1)}).", ""] + \
            table(["variant", "dist", "pair_us", "gather_us", "meas_ratio", "pred_g32", "pred_g64", "pred_g128"], probe)
    rows = [r for r in rows if r[0] in ("gather", "scatter_add")]
    return probe_md + ["", "### Traffic bounds (isolated mode)", "",
            f"`est_*_MiB` are model bytes; `impl_GBps_*` = estimate / measured median. If the *compulsory* "
            f"(lower-bound) bytes imply more than the estimated DRAM peak ({peak_gbps:.0f} GB/s), the model's "
            "32-B-sector assumption is contradicted.", ""] + \
        table(["variant", "dist", "T", "median_us", "hot_sector_share", "max_elem_count", "est_MiB_compulsory",
               "est_MiB_hot_set", "est_MiB_no_reuse", "impl_GBps_compulsory", "impl_GBps_no_reuse",
               "compulsory_exceeds_peak"], rows)


def summarize_fusion(series, meta, peak_gbps, peak_gflops) -> list[str]:
    by = {(s.variant, s.params_raw, s.timing_mode): s for s in series}
    params = sorted({s.params_raw for s in series}, key=lambda r: (float(bc.parse_params(r)["l2x"]),
                                                                    int(bc.parse_params(r)["K"])))
    rows = []
    for p in params:
        pp = bc.parse_params(p)
        n, k, l2 = int(pp["n"]), int(pp["K"]), int(pp["l2_bytes"])
        get = lambda v, mode: by.get((v, p, mode))
        fi, mi = get("fused", "isolated"), get("materialized", "isolated")
        fs, ms = get("fused", "steady"), get("materialized", "steady")
        cold, warm = get("stage2_cold", "isolated"), get("stage2_warm", "isolated")
        ai_f = bc.fusion_flops(n, k, "fused") / (8 * n)
        rows.append([pp["l2x"], n, k,
                     med_us(fi) if fi else float("nan"), med_us(mi) if mi else float("nan"),
                     med_us(mi) / med_us(fi) if fi and mi else float("nan"),
                     med_us(ms) / med_us(fs) if fs and ms else float("nan"),
                     med_us(warm) / med_us(cold) if warm and cold else float("nan"),
                     bc.fusion_residency(4 * n, l2, "lru_cliff"), bc.fusion_residency(4 * n, l2, "proportional"),
                     ai_f])
    ridge = peak_gflops / peak_gbps if peak_gbps == peak_gbps and peak_gbps > 0 else float("nan")
    return ["", "### Fusion payoff by size and arithmetic intensity", "",
            "`mat/fused` = materialized time / fused time (MEASURED; >1 means fusion helps). "
            "`warm/cold` = stage2 time right after stage1 / after an L2 flush (MEASURED; <1 means t was "
            "partly served from L2). `est_r_*` = fraction of t predicted to survive in L2 by the two "
            f"residency models. `AI_fused` = FLOP per algorithmic DRAM byte. Estimated ridge point "
            f"= {peak_gflops:.0f} GFLOP/s / {peak_gbps:.0f} GB/s = {ridge:.1f} FLOP/B (ESTIMATE).", ""] + \
        table(["l2x", "n", "K", "fused_us(iso)", "mat_us(iso)", "mat/fused(iso)", "mat/fused(steady)",
               "warm/cold", "est_r_lru_cliff", "est_r_proportional", "AI_fused"], rows)


def summarize_stencil(series) -> list[str]:
    by = {(s.variant, s.params_raw, s.timing_mode): s for s in series}
    params = sorted({s.params_raw for s in series}, key=lambda r: float(bc.parse_params(r)["l2x"]))
    rows = []
    for p in params:
        pp = bc.parse_params(p)
        g = lambda v, m: by.get((v, p, m))
        mi, fi, ms, fs = g("mat4", "isolated"), g("fused2", "isolated"), g("mat4", "steady"), g("fused2", "steady")
        fci = g("fused2c", "isolated")
        cold, warm = g("sweep_cold", "isolated"), g("sweep_warm", "isolated")
        q = lambda a, b: med_us(a) / med_us(b) if a and b else float("nan")
        rows.append([pp["l2x"], pp["H"], int(pp["n"]) * 4 / 2**20, med_us(mi) if mi else float("nan"),
                     med_us(fi) if fi else float("nan"), q(mi, fi), q(mi, fci), q(ms, fs), q(warm, cold)])
    return ["", "### B5 ratios (MEASURED)", "",
            "`mat4/fused2` > 1 means fusing sweeps pays off; `warm/cold` < 1 means the producer's output was "
            "served from L2, > 1 means its dirty lines were written back inside the consumer's window.", ""] + \
        table(["l2x", "H", "MiB_per_buffer", "mat4_us(iso)", "fused2_us(iso)", "mat4/fused2(iso)",
               "mat4/fused2c(iso)", "mat4/fused2(steady)", "warm/cold"], rows)


def summarize(bench: str, run_dir: Path) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    series = bc.load_generic_csv(run_dir / "reps.csv")
    peak_gbps = bc.theoretical_dram_bw_gbps(meta.get("mem_clock_khz_attr", -1), meta.get("mem_bus_width_bits", -1))
    peak_gflops = bc.fp32_peak_gflops(meta.get("sm_count", -1), meta.get("sm_clock_khz_attr", -1))
    md = [
        f"# {bench} summary -- run `{run_dir.name}`", "",
        f"- Device: {meta.get('device_name')} (cc {meta.get('compute_capability')}), driver "
        f"{meta.get('cuda_driver_version')}, runtime {meta.get('cuda_runtime_version')}, "
        f"L2 {meta.get('l2_cache_bytes')} B",
        f"- Correctness failures: {meta.get('correctness_failures')} "
        f"({sum(1 for v in meta.get('correctness', {}).values() if v == 0)} checks passed)",
        f"- reps {meta.get('reps')}, warmup {meta.get('warmup')}, steady iters {meta.get('steady_iters')}, "
        f"flush_l2 {meta.get('flush_l2')}",
        f"- ESTIMATED theoretical DRAM BW {peak_gbps:.1f} GB/s; ESTIMATED FP32 peak {peak_gflops:.0f} GFLOP/s "
        "(device attributes)",
        "", "`useful_GBps` = algorithmic bytes of the task / median time (DERIVED). "
        "`isolated` = L2 flushed before each rep; `steady` = back-to-back calls.", "",
    ] + common_rows(series)
    if bench == "transpose":
        md += summarize_transpose(series, meta)
    elif bench == "gather":
        md += summarize_gather(series, meta, peak_gbps)
    elif bench == "stencil":
        md += summarize_stencil(series)
    else:
        md += summarize_fusion(series, meta, peak_gbps, peak_gflops)
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ---------------------------------------------------------------------------
def main() -> int:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", choices=BENCHES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize-only", type=Path)
    args = ap.parse_args(argv)

    if args.summarize_only:
        summarize(args.bench, args.summarize_only)
        return 0
    if args.dry_run:
        dry_run(args.bench, args.smoke)
        return 0

    binary = args.binary or ROOT / "benchmarks" / "build" / args.bench
    if not binary.exists():
        print(f"binary not found: {binary}\nbuild first: bash benchmarks/scripts/build.sh", file=sys.stderr)
        return 2
    if shutil.which("nvidia-smi") is None:
        print("warning: nvidia-smi not on PATH; clocks/temperature recorded as unavailable", file=sys.stderr)
    out_dir = args.out_dir or ROOT / "benchmarks" / "results" / args.bench
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "") + (
        f"-{args.tag}" if args.tag else "")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    cmd = [str(binary)] + (["--smoke"] if args.smoke else []) + passthrough + \
          ["--csv", str(run_dir / "reps.csv"), "--meta", str(run_dir / "meta.json")]
    src = ROOT / "benchmarks" / "src"
    env = capture_env(binary, [src / f"{args.bench}.cu", src / "common.cuh", src / "dmk_host.hpp"] +
                      ([src / "stencil_host.hpp"] if args.bench == "stencil" else []))
    env["command"] = cmd
    env["nvidia_smi_before"] = nvidia_smi_snapshot()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = None, f"TIMEOUT after {RUN_TIMEOUT_S}s\n{e.stdout or ''}{e.stderr or ''}"
    env["nvidia_smi_after"] = nvidia_smi_snapshot()
    env["binary_rc"] = rc
    (run_dir / "stdout.log").write_text(out if isinstance(out, str) else str(out))
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(out)
    if rc not in (0, 1) or not (run_dir / "meta.json").exists():
        print(f"binary failed (rc={rc}); see {run_dir / 'stdout.log'}", file=sys.stderr)
        return 3
    summarize(args.bench, run_dir)
    if rc == 1:
        print("CORRECTNESS FAILURE: some variants were not timed; see meta.json", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
