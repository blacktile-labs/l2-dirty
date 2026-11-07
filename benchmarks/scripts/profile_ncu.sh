#!/usr/bin/env bash
# Collect hardware counters for the copy_stride kernels with Nsight Compute.
# Bounded: --smoke config, 1 rep, 0 warmup; ncu replays each kernel to collect metrics
# and (default --cache-control all) flushes caches between replay passes.
# If ncu is missing or counters are permission-blocked (ERR_NVGPUCTRPERM, common on WSL
# and on Linux without the NVreg_RestrictProfilingToAdminUsers=0 setting), this script
# records that fact and exits non-zero. It never substitutes estimated numbers.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(dirname "$HERE")"
BIN="${BIN:-$BENCH/build/copy_stride}"
RUN_DIR="$BENCH/results/ncu/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
METRICS=(
  gpu__time_duration.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  lts__t_sectors_srcunit_tex_op_read.sum
  lts__t_sectors_srcunit_tex_op_write.sum
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  dram__throughput.avg.pct_of_peak_sustained_elapsed
)
if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu: unavailable (not on PATH)" | tee "$RUN_DIR/STATUS"
  exit 2
fi
ncu --version > "$RUN_DIR/ncu_version.txt" 2>&1
ncu --kernel-name regex:'^copy_' --metrics "$(IFS=,; echo "${METRICS[*]}")" \
    --export "$RUN_DIR/copy_stride" --force-overwrite \
    "$BIN" --smoke --reps 1 --warmup 0 \
    --csv "$RUN_DIR/reps.csv" --meta "$RUN_DIR/meta.json" > "$RUN_DIR/stdout.txt" 2> "$RUN_DIR/stderr.txt"
rc=$?
if [ -f "$RUN_DIR/copy_stride.ncu-rep" ]; then
  ncu --import "$RUN_DIR/copy_stride.ncu-rep" --csv --page raw > "$RUN_DIR/ncu_raw.csv" 2>> "$RUN_DIR/stderr.txt"
fi
if grep -q "ERR_NVGPUCTRPERM" "$RUN_DIR/stdout.txt" "$RUN_DIR/stderr.txt" 2>/dev/null; then
  echo "ncu: permission-blocked (ERR_NVGPUCTRPERM); metrics = unavailable" | tee "$RUN_DIR/STATUS"
  exit 3
fi
echo "ncu rc=$rc; raw metrics in $RUN_DIR/ncu_raw.csv" | tee "$RUN_DIR/STATUS"
exit $rc
