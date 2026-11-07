#!/usr/bin/env bash
# B18 L2-flush fixes, rules H-0..H-4 (research/HYPOTHESES.md "B18 pre-registration").
# 3 x range mode (ncu --replay-mode app-range: DRAM bytes of each timed-call range) interleaved with
# 3 x timing mode (no profiler: CUDA events). Python: $PY (torch 2.7.1, triton 3.3.1). Local GPU only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/flush_fix_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
H=benchmarks/scripts/flush_fix_probe.py
RUNS=3; [ -n "$SMOKE" ] && RUNS=1
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
NCU=(ncu --target-processes application-only --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version
  "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" benchmarks/scripts/flush_fix_score.py benchmarks/scripts/counter_prereg.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
for i in $(seq 1 $RUNS); do
  "${NCU[@]}" --replay-mode app-range --export "$OUT/range_r$i" "$PY" "$H" --mode range $SMOKE \
    --meta "$OUT/range_r${i}_meta.json" > "$OUT/range_r${i}_ncu.txt" 2>&1
  echo "range_r$i rc=$?" | tee -a "$OUT/STATUS"
  grep -q "ERR_NVGPUCTRPERM" "$OUT/range_r${i}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
  ncu --import "$OUT/range_r$i.ncu-rep" --csv --page raw --print-units base > "$OUT/range_r${i}_raw.csv"
  "$PY" "$H" --mode timing $SMOKE --meta "$OUT/timing_r${i}_meta.json" > "$OUT/timing_r${i}.txt" 2>&1
  echo "timing_r$i rc=$?" | tee -a "$OUT/STATUS"
done
if [ -n "$SMOKE" ]; then
  # pipeline check only: row counts, no measurement is printed
  python3 benchmarks/scripts/flush_fix_score.py --range-meta "$OUT"/range_r*_meta.json \
    --range-csv "$OUT"/range_r*_raw.csv --timing-meta "$OUT"/timing_r*_meta.json --check-only
  echo "smoke results (not to be read before the lock): $OUT"
  exit 0
fi
python3 benchmarks/scripts/flush_fix_score.py --range-meta "$OUT"/range_r*_meta.json \
  --range-csv "$OUT"/range_r*_raw.csv --timing-meta "$OUT"/timing_r*_meta.json \
  --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
