#!/usr/bin/env bash
# B14 quantized-GEMV speedup bias, rules U-0..U-2 (research/HYPOTHESES.md "B14 pre-registration").
# 3 runs of quant_bias_probe.py (no profiler; candidate order shuffled per run), then the scorer.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/quant_bias_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
H=benchmarks/scripts/quant_bias_probe.py
RUNS=3; [ -n "$SMOKE" ] && RUNS=1
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" benchmarks/scripts/quant_bias_score.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
for i in $(seq 1 $RUNS); do
  "$PY" "$H" $SMOKE --run-seed "$i" --meta "$OUT/run_r${i}_meta.json" > "$OUT/run_r$i.txt" 2>&1
  echo "run_r$i rc=$?" | tee -a "$OUT/STATUS"
done
python3 benchmarks/scripts/quant_bias_score.py --meta "$OUT"/run_r*_meta.json --json-out "$OUT/score.json" \
  > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
