#!/usr/bin/env bash
# Model v4 counter test K-0..K-3 (pre-registered 2026-11-05 before any counter at these sizes was
# read; research/HYPOTHESES.md "Model v4 counter pre-registration"; predictions in
# research/preregistration/counters_v4_rtx3060laptop.json).
# Identical to local_counters.sh (D-026) except: new sizes, 3 runs per benchmark, scored by
# counter_v4.py (v3 vs v4 per launch). Same binaries, same binary args, same ncu flags:
#   --replay-mode application --cache-control none --clock-control none
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
OUT="benchmarks/results/counters_v4_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
PREREG=research/preregistration/counters_v4_rtx3060laptop.json
SIZES="0.1875,0.3125,0.375,0.4375,0.5,2,3"
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
python3 -c "import json,sys; s=json.load(open('$PREREG'))['benches']['fusion']['sizes_l2x']; sys.exit(0 if s==[float(x) for x in '$SIZES'.split(',')] else 1)" \
  || { echo "sizes differ from the pre-registration" | tee "$OUT/STATUS"; exit 2; }
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD
  sha256sum "$0" "$PREREG" benchmarks/scripts/counter_v4.py benchmarks/scripts/counter_prereg.py \
    benchmarks/scripts/cache_sim.py benchmarks/scripts/cache_sim_v4.py benchmarks/scripts/cache_sim_b5.py \
    benchmarks/src/fusion.cu benchmarks/src/stencil.cu benchmarks/src/common.cuh benchmarks/build/fusion benchmarks/build/stencil
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

ARGS_fusion="--sizes-l2x $SIZES --ks 1 --reps 1 --warmup 0 --steady-iters 3"
ARGS_stencil="--sizes-l2x $SIZES --reps 1 --warmup 0 --steady-iters 3"
for b in fusion stencil; do
  args_var="ARGS_$b"
  for i in 1 2 3; do
    # shellcheck disable=SC2086
    ncu --replay-mode application --cache-control none --clock-control none --metrics "$METRICS" \
        --export "$OUT/${b}_r$i" --force-overwrite \
        "benchmarks/build/$b" ${!args_var} --csv "$OUT/${b}_r${i}_reps.csv" --meta "$OUT/${b}_r${i}_meta.json" \
        > "$OUT/${b}_r${i}_ncu.txt" 2>&1
    echo "$b run $i ncu rc=$?" | tee -a "$OUT/STATUS"
    grep -q "ERR_NVGPUCTRPERM" "$OUT/${b}_r${i}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
    ncu --import "$OUT/${b}_r$i.ncu-rep" --csv --page raw --print-units base > "$OUT/${b}_r${i}_raw.csv"
  done
  python3 benchmarks/scripts/counter_v4.py score --bench "$b" --ncu-csv "$OUT/${b}"_r{1,2,3}_raw.csv \
      --prereg "$PREREG" --json-out "$OUT/${b}_score.json" --verbose > "$OUT/${b}_score.txt" 2>&1
  echo "$b score rc=$?" | tee -a "$OUT/STATUS"
  grep -E "^S-0|^K-|verdict" "$OUT/${b}_score.txt" | tee -a "$OUT/STATUS"
done
echo "results: $OUT"
