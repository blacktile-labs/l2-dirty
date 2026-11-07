#!/usr/bin/env bash
# B18c cleanbench validation, rules T-0..T-2 (research/HYPOTHESES.md "B18c pre-registration"). Timing only.
# 3 runs of cleanbench_validate.py. Python: $PY (torch 2.7.1, triton 3.3.1). Local GPU only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/cleanbench_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
H=benchmarks/scripts/cleanbench_validate.py
RUNS=3; [ -n "$SMOKE" ] && RUNS=1
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" benchmarks/scripts/cleanbench.py benchmarks/scripts/cleanbench_score.py
} > "$OUT/environment.txt" 2>&1
for i in $(seq 1 $RUNS); do
  (cd benchmarks/scripts && "$PY" cleanbench_validate.py $SMOKE --meta "../../$OUT/run_r${i}_meta.json") > "$OUT/run_r${i}.txt" 2>&1
  echo "run_r$i rc=$?" | tee -a "$OUT/STATUS"
done
if [ -n "$SMOKE" ]; then
  python3 benchmarks/scripts/cleanbench_score.py --meta "$OUT"/run_r*_meta.json --check-only
  echo "smoke results (not to be read before the lock): $OUT"
  exit 0
fi
python3 benchmarks/scripts/cleanbench_score.py --meta "$OUT"/run_r*_meta.json --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
