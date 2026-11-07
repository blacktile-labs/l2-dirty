#!/usr/bin/env bash
# B9b queued write-back probe, rules Q-0..Q-3 (research/HYPOTHESES.md "B9b pre-registration").
# 3 x kernel mode (ncu --replay-mode application) interleaved with 3 x range mode (app-range).
# Python: isolated venv ~/.cache/dmkl-venv-torch (benchmarks/requirements-torch.txt).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
OUT="benchmarks/results/queued_wb_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
H=benchmarks/scripts/queued_wb_probe.py
RUNS=3
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
NCU=(ncu --target-processes application-only --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD; git status --short
  "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" benchmarks/scripts/idle_wb_probe.py benchmarks/scripts/torch_opchain.py \
    benchmarks/scripts/queued_wb_score.py benchmarks/scripts/cache_sim_v4.py benchmarks/scripts/counter_prereg.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

"$PY" "$H" --mode range --meta "$OUT/warm_meta.json" > "$OUT/warm.txt" 2>&1
echo "warm rc=$?" | tee -a "$OUT/STATUS"
for i in $(seq 1 $RUNS); do
  for mode in kernel range; do
    rm=application; [ "$mode" = range ] && rm=app-range
    "${NCU[@]}" --replay-mode "$rm" --export "$OUT/${mode}_r$i" "$PY" "$H" --mode "$mode" \
      --meta "$OUT/${mode}_r${i}_meta.json" > "$OUT/${mode}_r${i}_ncu.txt" 2>&1
    echo "${mode}_r$i rc=$?" | tee -a "$OUT/STATUS"
    grep -q "ERR_NVGPUCTRPERM" "$OUT/${mode}_r${i}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
    ncu --import "$OUT/${mode}_r$i.ncu-rep" --csv --page raw --print-units base > "$OUT/${mode}_r${i}_raw.csv"
  done
done
python3 benchmarks/scripts/queued_wb_score.py score \
  --kernel-meta "$OUT"/kernel_r*_meta.json --kernel-csv "$OUT"/kernel_r*_raw.csv \
  --range-meta "$OUT"/range_r*_meta.json --range-csv "$OUT"/range_r*_raw.csv \
  --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
cat "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
