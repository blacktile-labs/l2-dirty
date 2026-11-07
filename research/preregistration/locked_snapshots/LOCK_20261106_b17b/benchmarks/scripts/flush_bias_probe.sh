#!/usr/bin/env bash
# B17 write flush vs PR #12556 read flush (rules E-0, F-1..F-6, research/HYPOTHESES.md "B17 pre-registration").
# 5 runs of flush_bias_probe.py (no profiler counters; case and mode order shuffled per run), then the
# per-GPU scorer. The cross-GPU rules (F-2, F-5) are scored locally with flush_bias_score.py --all.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/flush_bias_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
H=benchmarks/scripts/flush_bias_probe.py
RUNS=5; [ -n "$SMOKE" ] && RUNS=1
{
  date -u; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,temperature.gpu,clocks.sm,clocks.mem,clocks.max.sm,pstate --format=csv
  "$PY" -c "import torch, triton, importlib.metadata as m; print(torch.__version__, torch.version.cuda, triton.__version__, m.version('bitsandbytes'), m.version('liger-kernel'))"
  sha256sum "$0" "$H" benchmarks/scripts/flush_bias_score.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
for i in $(seq 1 $RUNS); do
  "$PY" "$H" $SMOKE --run-seed "$i" --meta "$OUT/run_r${i}_meta.json" > "$OUT/run_r$i.txt" 2>&1
  echo "run_r$i rc=$?" | tee -a "$OUT/STATUS"
done
"$PY" benchmarks/scripts/flush_bias_score.py --meta "$OUT"/run_r*_meta.json --json-out "$OUT/score.json" \
  > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
