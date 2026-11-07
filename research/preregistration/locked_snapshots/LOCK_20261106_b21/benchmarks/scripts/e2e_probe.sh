#!/usr/bin/env bash
# B21 (research/HYPOTHESES.md "B21 pre-registration"): 3 runs of e2e_probe.py on one GPU, scored here.
# Timing only. Python: $PY (torch 2.7.1, triton 3.3.1, bitsandbytes 0.50.2).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
OUT="benchmarks/results/e2e_$([ -n "$SMOKE" ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
RUNS=3; [ -n "$SMOKE" ] && RUNS=1
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate,power.limit --format=csv
  lscpu | grep -E "Model name|^CPU\(s\)"
  "$PY" -c "import torch, triton, importlib.metadata as m; print(torch.__version__, torch.version.cuda, triton.__version__, m.version('bitsandbytes'))"
  sha256sum "$0" benchmarks/scripts/e2e_probe.py benchmarks/scripts/e2e_score.py benchmarks/scripts/cleanbench.py
} > "$OUT/environment.txt" 2>&1
for i in $(seq 1 $RUNS); do
  (cd benchmarks/scripts && "$PY" e2e_probe.py $SMOKE --meta "../../$OUT/run_r${i}_meta.json") > "$OUT/run_r${i}.txt" 2>&1
  echo "run_r$i rc=$?" | tee -a "$OUT/STATUS"
done
if [ -n "$SMOKE" ]; then
  python3 benchmarks/scripts/e2e_score.py --meta "$OUT"/run_r*_meta.json --check-only
  echo "smoke results (not to be read before the lock): $OUT"
  exit 0
fi
python3 benchmarks/scripts/e2e_score.py --meta "$OUT"/run_r*_meta.json --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
