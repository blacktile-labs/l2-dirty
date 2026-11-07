#!/usr/bin/env bash
# B15 absorption protocol, rules K-0..K-3 (research/HYPOTHESES.md "B15 pre-registration").
# 1 range run under ncu --replay-mode app-range (3 repetitions per config and flush mode inside) +
# 3 timing runs (real do_bench, no profiler), then the scorer.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/absorb_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
H=benchmarks/scripts/absorb_probe.py
RUNS=3; [ -n "$SMOKE" ] && RUNS=1
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
NCU=(ncu --target-processes application-only --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" benchmarks/scripts/quant_bias_probe.py benchmarks/scripts/absorb_score.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
"${NCU[@]}" --replay-mode app-range --export "$OUT/range" "$PY" "$H" --mode range $SMOKE --run-seed 7 \
  --meta "$OUT/range_meta.json" > "$OUT/range_ncu.txt" 2>&1
echo "range rc=$?" | tee -a "$OUT/STATUS"
grep -q "ERR_NVGPUCTRPERM" "$OUT/range_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
ncu --import "$OUT/range.ncu-rep" --csv --page raw --print-units base > "$OUT/range_raw.csv"
for i in $(seq 1 $RUNS); do
  "$PY" "$H" --mode timing $SMOKE --run-seed "$i" --meta "$OUT/timing_r${i}_meta.json" > "$OUT/timing_r$i.txt" 2>&1
  echo "timing_r$i rc=$?" | tee -a "$OUT/STATUS"
done
python3 benchmarks/scripts/absorb_score.py --range-meta "$OUT/range_meta.json" --range-csv "$OUT/range_raw.csv" \
  --timing-meta "$OUT"/timing_r*_meta.json --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
