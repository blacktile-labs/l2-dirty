#!/usr/bin/env bash
# B23 (research/HYPOTHESES.md "B23 pre-registration"): 3 runs of kernel_time_probe.py on one GPU, scored here.
# Timing only (CUPTI activity tracing through torch.profiler; no counters, so no profiling permission needed).
# Descriptive logs, not scored: nvidia-smi every 1 s (clocks, power, power limit, clock-event reasons) and
# `nvidia-smi -q -d PERFORMANCE` (cumulative clock-event counters, e.g. SW power capping) before and after each run.
# Each run is capped at 900 s. Python: $PY (torch 2.7.1, triton 3.3.1, bitsandbytes 0.50.2).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/kernel_time_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
RUNS=3; [ -n "$SMOKE" ] && RUNS=1
{
  date -u; date +%z
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate,power.limit,utilization.gpu --format=csv
  nvidia-smi -q -d POWER,CLOCK,PERFORMANCE
  lscpu | grep -E "Model name|^CPU\(s\)"
  "$PY" -c "import torch, triton, importlib.metadata as m; print(torch.__version__, torch.version.cuda, triton.__version__, m.version('bitsandbytes'))"
  sha256sum "$0" benchmarks/scripts/kernel_time_probe.py benchmarks/scripts/kernel_time_score.py \
    benchmarks/scripts/cupti_records.py benchmarks/scripts/graph_time_probe.py benchmarks/scripts/e2e_probe.py \
    benchmarks/scripts/cleanbench.py
} > "$OUT/environment.txt" 2>&1
F="timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,pstate"
if nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader > /dev/null 2>&1; then
  F="$F,clocks_event_reasons.active"
elif nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader > /dev/null 2>&1; then
  F="$F,clocks_throttle_reasons.active"
fi
nvidia-smi --query-gpu="$F" --format=csv -l 1 > "$OUT/nvsmi.csv" 2>&1 &
NV=$!
trap 'kill $NV 2> /dev/null' EXIT
for i in $(seq 1 $RUNS); do
  nvidia-smi -q -d PERFORMANCE > "$OUT/perf_before_r$i.txt" 2>&1
  (cd benchmarks/scripts && timeout 900 "$PY" kernel_time_probe.py $SMOKE --meta "../../$OUT/run_r${i}_meta.json") \
    > "$OUT/run_r${i}.txt" 2>&1
  echo "run_r$i rc=$?" | tee -a "$OUT/STATUS"
  nvidia-smi -q -d PERFORMANCE > "$OUT/perf_after_r$i.txt" 2>&1
done
kill $NV 2> /dev/null
if [ -n "$SMOKE" ]; then
  python3 benchmarks/scripts/kernel_time_score.py --meta "$OUT"/run_r*_meta.json --check-only
  echo "smoke results (not to be read before the lock): $OUT"
  exit 0
fi
python3 benchmarks/scripts/kernel_time_score.py --meta "$OUT"/run_r*_meta.json --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
