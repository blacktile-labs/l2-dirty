#!/usr/bin/env bash
# Counter protocol C-1..C-4 (pre-registered 2026-11-04; research/HYPOTHESES.md "Counter
# pre-registration"; predictions in research/preregistration/counters_rtx3060laptop.json).
# Needs GPU performance-counter access. On WSL2 this is a *Windows host* setting; if ncu
# reports ERR_NVGPUCTRPERM the script records `unavailable` and exits 3 without numbers.
#
# Measurement choices (fixed here, before any counter has been read):
#   --replay-mode application  every pass reruns the whole binary, so no save/restore memcpy
#                              between passes disturbs the L2 state a launch sees
#   --cache-control none       ncu must not flush caches: the L2 state left by the previous
#                              launch is exactly what is being tested
#   --clock-control none       bytes, not time, are scored; leave clocks as they are
#   binary: --reps 1 --warmup 0 --steady-iters 3 (the launch plan in counter_prereg.py)
# Known difference from the timed runs: ncu serializes launches with idle gaps; if the L2
# writes dirty lines back while idle, C-2 will show it (that is part of what is tested).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
OUT="benchmarks/results/counters_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
PREREG=research/preregistration/counters_rtx3060laptop.json
SIZES="0.25,0.46875,0.5625,0.75,1,2,4"
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem --format=csv
  ncu --version
  sha256sum "$0" "$PREREG" benchmarks/scripts/counter_prereg.py benchmarks/scripts/cache_sim.py \
    benchmarks/scripts/cache_sim_b5.py benchmarks/src/fusion.cu benchmarks/src/stencil.cu benchmarks/src/common.cuh
} > "$OUT/environment.txt" 2>&1

for b in fusion stencil; do
  [ -x "benchmarks/build/$b" ] || { echo "missing benchmarks/build/$b (run build.sh)" | tee "$OUT/STATUS"; exit 2; }
done

ARGS_fusion="--sizes-l2x $SIZES --ks 1 --reps 1 --warmup 0 --steady-iters 3"
ARGS_stencil="--sizes-l2x $SIZES --reps 1 --warmup 0 --steady-iters 3"
for b in fusion stencil; do
  args_var="ARGS_$b"
  # shellcheck disable=SC2086
  ncu --replay-mode application --cache-control none --clock-control none --metrics "$METRICS" \
      --export "$OUT/$b" --force-overwrite \
      "benchmarks/build/$b" ${!args_var} --csv "$OUT/${b}_reps.csv" --meta "$OUT/${b}_meta.json" \
      > "$OUT/${b}_ncu_stdout.txt" 2> "$OUT/${b}_ncu_stderr.txt"
  echo "$b ncu rc=$?" | tee -a "$OUT/STATUS"
  if grep -q "ERR_NVGPUCTRPERM" "$OUT/${b}_ncu_stdout.txt" "$OUT/${b}_ncu_stderr.txt" 2>/dev/null; then
    echo "ncu: permission-blocked (ERR_NVGPUCTRPERM); counters = unavailable" | tee -a "$OUT/STATUS"
    exit 3
  fi
  ncu --import "$OUT/$b.ncu-rep" --csv --page raw --print-units base > "$OUT/${b}_raw.csv" 2>> "$OUT/${b}_ncu_stderr.txt"
  python3 benchmarks/scripts/counter_prereg.py score --bench "$b" --ncu-csv "$OUT/${b}_raw.csv" \
      --prereg "$PREREG" --verbose > "$OUT/${b}_score.txt" 2>&1
  echo "$b score rc=$?" | tee -a "$OUT/STATUS"
  grep -E "^S-0|^C-|verdict" "$OUT/${b}_score.txt" | tee -a "$OUT/STATUS"
done
