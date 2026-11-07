#!/usr/bin/env python3
"""Run, record and summarize the copy_stride baseline.

  # Print the plan and the traffic-model ESTIMATES; touches no GPU:
  python3 benchmarks/scripts/run_copy_stride.py --smoke --dry-run

  # Bounded smoke run on a real CUDA GPU (build first: benchmarks/scripts/build.sh):
  python3 benchmarks/scripts/run_copy_stride.py --smoke

Every run writes to benchmarks/results/copy_stride/<run_id>/:
  env.json         environment capture (versions, nvidia-smi before/after)
  reps.csv         MEASURED per-repetition times (written by the CUDA binary)
  meta.json        device attributes + correctness results (written by the binary)
  stdout.log       binary stdout/stderr
  summary.csv/.md  DERIVED statistics next to model ESTIMATES (clearly labelled)

Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import bench_common as bc
from harness_env import ROOT, capture_env, nvidia_smi_snapshot
DEFAULT_BIN = ROOT / "benchmarks" / "build" / "copy_stride"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "copy_stride"

SMOKE = dict(footprint_mib=64, strides=[1, 2, 8, 16, 32], reps=10, warmup=3)
FULL = dict(footprint_mib=256, strides=[1, 2, 4, 8, 16, 32, 64], reps=50, warmup=10)
MODEL_GRANULARITIES = (32, 64, 128)
RUN_TIMEOUT_S = 600


def model_rows(n_src: int, strides: list[int]) -> list[dict]:
    rows = []
    for s in strides:
        n_out = n_src // s
        row = {"stride": s, "n_out": n_out, "logical_bytes": bc.logical_bytes(n_out)}
        for g in MODEL_GRANULARITIES:
            row[f"est_bytes_g{g}"] = bc.model_copy_stride_bytes(n_src, s, 4, g)
            row[f"est_amp_g{g}"] = bc.amplification(n_src, s, 4, g)
        rows.append(row)
    return rows


def print_dry_run(cfg: dict) -> None:
    n_src = cfg["footprint_mib"] * 1024 * 1024 // 4
    n_src -= n_src % 4
    print("DRY RUN -- no GPU touched. Numbers below are MODEL ESTIMATES, not measurements.")
    print(f"config: {json.dumps(cfg)}  n_src={n_src}")
    print("variants: empty_kernel, memcpy_d2d, copy_contig_u32, copy_contig_u32x4, "
          + ", ".join(f"copy_strided_read_u32_s{s}" for s in cfg["strides"]))
    hdr = "stride  n_out       logical_MiB  " + "  ".join(
        f"est_MiB(g={g})  amp(g={g})" for g in MODEL_GRANULARITIES)
    print(hdr)
    for r in model_rows(n_src, cfg["strides"]):
        cells = "  ".join(
            f"{r[f'est_bytes_g{g}'] / 2**20:12.2f}  {r[f'est_amp_g{g}']:9.3f}"
            for g in MODEL_GRANULARITIES)
        print(f"{r['stride']:6d}  {r['n_out']:10d}  {r['logical_bytes'] / 2**20:11.2f}  {cells}")


def summarize_run(run_dir: Path) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    groups = bc.load_reps_csv(run_dir / "reps.csv")
    by_name = {g.variant: g for g in groups}

    # Reference bandwidth for predictions: MEASURED cudaMemcpy D2D (reads + writes n_src).
    ref_bw = float("nan")
    if "memcpy_d2d" in by_name:
        m = by_name["memcpy_d2d"]
        ref_bw = m.logical_bytes / (bc.summarize(m.times_ms).median * 1e-3) / 1e9
    theo_bw = bc.theoretical_dram_bw_gbps(meta.get("mem_clock_khz_attr", -1),
                                          meta.get("mem_bus_width_bits", -1))
    empty_med_us = (bc.summarize(by_name["empty_kernel"].times_ms).median * 1e3
                    if "empty_kernel" in by_name else float("nan"))

    rows = []
    for g in groups:
        s = bc.summarize(g.times_ms)
        row = {
            "variant": g.variant, "stride": g.stride, "n_out": g.n_out, "reps": s.n,
            "median_us": s.median * 1e3, "p95_us": s.p95 * 1e3, "min_us": s.min * 1e3,
            "max_us": s.max * 1e3, "cv_pct": s.cv * 100,
            "useful_GBps": (g.logical_bytes / (s.median * 1e-3) / 1e9) if g.logical_bytes else 0.0,
        }
        if g.kernel == "copy_strided_read_u32":
            for gran in (32, 64):
                est = bc.model_copy_stride_bytes(g.n_src, g.stride, g.elem_bytes, gran)
                pred_us = est / (ref_bw * 1e9) * 1e6
                row[f"est_MiB_g{gran}"] = est / 2**20
                row[f"pred_us_g{gran}"] = pred_us
                row[f"meas_over_pred_g{gran}"] = row["median_us"] / pred_us
        rows.append(row)

    cols = ["variant", "stride", "n_out", "reps", "median_us", "p95_us", "min_us", "max_us",
            "cv_pct", "useful_GBps", "est_MiB_g32", "pred_us_g32", "meas_over_pred_g32",
            "est_MiB_g64", "pred_us_g64", "meas_over_pred_g64"]
    with open(run_dir / "summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(c, "")) for c in cols) + "\n")

    md = [
        f"# copy_stride summary -- run `{run_dir.name}`",
        "",
        f"- Device: {meta.get('device_name')} (cc {meta.get('compute_capability')}), "
        f"driver {meta.get('cuda_driver_version')}, runtime {meta.get('cuda_runtime_version')}",
        f"- n_src = {meta.get('n_src')} uint32 ({meta.get('footprint_mib')} MiB), "
        f"flush_l2 = {meta.get('flush_l2')}, reps = {meta.get('reps')}, warmup = {meta.get('warmup')}",
        f"- Correctness failures: {meta.get('correctness_failures')} "
        f"(per-variant mismatches: {meta.get('correctness')})",
        f"- L2 fetch granularity reported by driver: {meta.get('l2_fetch_granularity_bytes')} B",
        f"- MEASURED reference BW (cudaMemcpy D2D, logical bytes / median): {ref_bw:.1f} GB/s",
        f"- ESTIMATED theoretical DRAM BW (device attributes, x2 DDR assumed): {theo_bw:.1f} GB/s",
        f"- MEASURED empty-kernel event floor (median): {empty_med_us:.2f} us",
        "",
        "Columns `est_*` and `pred_*` are MODEL ESTIMATES (bench_common.py, assumptions A1-A5);",
        "`pred_us` = estimated bytes / measured memcpy BW. Everything else is measured or derived.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        md.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="small bounded configuration")
    ap.add_argument("--dry-run", action="store_true", help="print plan + model estimates only")
    ap.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--footprint-mib", type=int)
    ap.add_argument("--strides", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--reps", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--no-flush-l2", action="store_true")
    ap.add_argument("--l2-fetch-granularity", type=int, default=-1,
                    help="hint passed to cudaLimitMaxL2FetchGranularity; -1 = driver default")
    ap.add_argument("--tag", default="", help="free-text label appended to the run id")
    ap.add_argument("--summarize-only", type=Path, help="re-summarize an existing run directory")
    args = ap.parse_args()

    if args.summarize_only:
        summarize_run(args.summarize_only)
        return 0

    cfg = dict(SMOKE if args.smoke else FULL)
    for k in ("footprint_mib", "strides", "reps", "warmup"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    if args.dry_run:
        print_dry_run(cfg)
        return 0

    if not args.binary.exists():
        print(f"binary not found: {args.binary}\nbuild first: bash benchmarks/scripts/build.sh",
              file=sys.stderr)
        return 2
    if shutil.which("nvidia-smi") is None:
        print("warning: nvidia-smi not on PATH; clocks/temperature will be recorded as unavailable",
              file=sys.stderr)

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "") + (
        f"-{args.tag}" if args.tag else "")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    cmd = [str(args.binary), "--footprint-mib", str(cfg["footprint_mib"]),
           "--strides", ",".join(map(str, cfg["strides"])), "--reps", str(cfg["reps"]),
           "--warmup", str(cfg["warmup"]), "--csv", str(run_dir / "reps.csv"),
           "--meta", str(run_dir / "meta.json")]
    if args.smoke:
        cmd.insert(1, "--smoke")
    if args.no_flush_l2:
        cmd.append("--no-flush-l2")
    if args.l2_fetch_granularity >= 0:
        cmd += ["--l2-fetch-granularity", str(args.l2_fetch_granularity)]

    env = capture_env(args.binary, [ROOT / "benchmarks" / "src" / "copy_stride.cu"])
    env["command"] = cmd
    env["config"] = cfg
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
    summarize_run(run_dir)
    if rc == 1:
        print("CORRECTNESS FAILURE: some variants were not timed; see meta.json", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
