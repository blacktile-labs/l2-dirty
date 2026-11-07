#!/usr/bin/env bash
# B7 real PyTorch op chain, rules R-0..R-3 and T-1 (pre-registered 2026-11-05 before any B7 cell at
# these sizes was measured; research/HYPOTHESES.md "B7 pre-registration"; locked predictions in
# research/preregistration/b7_predictions.json).
#   1. warm the Inductor cache (counters mode, no profiler): ncu application replay needs an identical
#      launch sequence in every pass
#   2. 3 x ncu --replay-mode application --target-processes application-only (per-launch DRAM bytes)
#   3. score (counter_b7.py score): R-0..R-3, and check that each run's plan reproduces the locked predictions
#   4. 3 x time mode (no profiler), then counter_b7.py time: T-1 + descriptive table
# Python: isolated venv ~/.cache/dmkl-venv-torch (benchmarks/requirements-torch.txt).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
PY="${PY:-$HOME/.cache/dmkl-venv-torch/bin/python}"
OUT="benchmarks/results/b7_opchain_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
PREREG=research/preregistration/b7_predictions.json
SIZES="0.125,0.1875,0.25,0.3125,0.375,0.5,1,2,3"
H=benchmarks/scripts/torch_opchain.py
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD; "$PY" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
  sha256sum "$0" "$H" "$PREREG" benchmarks/scripts/counter_b7.py benchmarks/scripts/cache_sim.py \
    benchmarks/scripts/cache_sim_v4.py benchmarks/requirements-torch.txt
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

"$PY" "$H" --mode counters --sizes-l2x "$SIZES" --plan "$OUT/warm_plan.json" --meta "$OUT/warm_meta.json" > "$OUT/warm.txt" 2>&1
echo "warm rc=$?" | tee -a "$OUT/STATUS"
for i in 1 2 3; do
  ncu --replay-mode application --target-processes application-only --cache-control none --clock-control none \
      --metrics "$METRICS" --export "$OUT/counters_r$i" --force-overwrite \
      "$PY" "$H" --mode counters --sizes-l2x "$SIZES" --plan "$OUT/counters_r${i}_plan.json" \
      --meta "$OUT/counters_r${i}_meta.json" > "$OUT/counters_r${i}_ncu.txt" 2>&1
  echo "counters run $i rc=$?" | tee -a "$OUT/STATUS"
  grep -q "ERR_NVGPUCTRPERM" "$OUT/counters_r${i}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
  ncu --import "$OUT/counters_r$i.ncu-rep" --csv --page raw --print-units base > "$OUT/counters_r${i}_raw.csv"
done
for i in 2 3; do
  python3 benchmarks/scripts/counter_b7.py score --prereg "$PREREG" --plan "$OUT/counters_r${i}_plan.json" \
    --ncu-csv "$OUT/counters_r${i}_raw.csv" 2>&1 | grep "identical predictions" | sed "s/^/run $i: /" | tee -a "$OUT/STATUS"
done
python3 benchmarks/scripts/counter_b7.py score --prereg "$PREREG" --plan "$OUT/counters_r1_plan.json" \
  --ncu-csv "$OUT"/counters_r{1,2,3}_raw.csv --json-out "$OUT/score.json" --verbose > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
grep -E "identical predictions|^R-|^   (eager|compiled)|verdict" "$OUT/score.txt" | tee -a "$OUT/STATUS"
for i in 1 2 3; do
  "$PY" "$H" --mode time --sizes-l2x "$SIZES" --csv "$OUT/time_r$i.csv" --meta "$OUT/time_r${i}_meta.json" \
    > "$OUT/time_r$i.txt" 2>&1
  echo "time run $i rc=$?" | tee -a "$OUT/STATUS"
done
python3 benchmarks/scripts/counter_b7.py time --csv "$OUT"/time_r{1,2,3}.csv > "$OUT/time_summary.txt" 2>&1
cat "$OUT/time_summary.txt" | tee -a "$OUT/STATUS"
python3 - "$OUT" <<'PY' | tee -a "$OUT/STATUS"
import json, pathlib, sys
print("correctness_failures per run:", {p.name: json.loads(p.read_text()).get("correctness_failures") for p in sorted(pathlib.Path(sys.argv[1]).glob("*_meta.json"))})
PY
echo "results: $OUT"
