#!/usr/bin/env bash
# B7r2: rerun of B7r with the flag gate (B7r failed Y-0c, D-049); rules Y-0..Y-3 as B7r with Y-0c for the
# flag gate (research/HYPOTHESES.md "B7r2 pre-registration";
# predictions are B7's locked research/preregistration/b7_predictions.json, unchanged).
#   kernel mode: ncu --replay-mode application        (per launch, profiler gaps between launches)
#   range mode : ncu --replay-mode app-range + torch_opchain.py --profiler-ranges --range-gate flag
#                (one range per segment; a gate kernel holds the GPU until the CPU has enqueued it)
# 5 runs of each, interleaved (K1 G1 K2 G2 ...), same harness, same sizes as B7.
# Python: isolated venv ~/.cache/dmkl-venv-torch (benchmarks/requirements-torch.txt).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
OUT="benchmarks/results/b7r2_opchain_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
PREREG=research/preregistration/b7_predictions.json
SIZES="0.125,0.1875,0.25,0.3125,0.375,0.5,1,2,3"
RUNS=5
H=benchmarks/scripts/torch_opchain.py
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
NCU=(ncu --target-processes application-only --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD; git status --short
  "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" "$PREREG" benchmarks/scripts/counter_b7r.py benchmarks/scripts/counter_b7.py \
    benchmarks/scripts/cache_sim.py benchmarks/scripts/cache_sim_v4.py benchmarks/requirements-torch.txt
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

"$PY" "$H" --mode counters --sizes-l2x "$SIZES" --plan "$OUT/warm_plan.json" --meta "$OUT/warm_meta.json" > "$OUT/warm.txt" 2>&1
echo "warm rc=$?" | tee -a "$OUT/STATUS"
for i in $(seq 1 $RUNS); do
  for mode in kernel range; do
    tag="${mode}_r$i" rm=application extra=()
    [ "$mode" = range ] && { rm=app-range; extra=(--profiler-ranges --range-gate flag); }
    "${NCU[@]}" --replay-mode "$rm" --export "$OUT/$tag" \
      "$PY" "$H" --mode counters --sizes-l2x "$SIZES" "${extra[@]}" --plan "$OUT/${tag}_plan.json" \
      --meta "$OUT/${tag}_meta.json" > "$OUT/${tag}_ncu.txt" 2>&1
    echo "$tag rc=$?" | tee -a "$OUT/STATUS"
    grep -q "ERR_NVGPUCTRPERM" "$OUT/${tag}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
    ncu --import "$OUT/$tag.ncu-rep" --csv --page raw --print-units base > "$OUT/${tag}_raw.csv"
  done
done
S=benchmarks/scripts
python3 $S/counter_b7r.py score --prereg "$PREREG" \
  --kernel-plan "$OUT"/kernel_r*_plan.json --kernel-csv "$OUT"/kernel_r*_raw.csv \
  --range-plan "$OUT"/range_r*_plan.json --range-csv "$OUT"/range_r*_raw.csv \
  --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
grep -E "^plans|^Y-|^descriptive|verdict" "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
