#!/usr/bin/env bash
# B6 write-back probe protocol W-0..W-5 (pre-registered 2026-11-05 before any grid cell was
# measured; research/HYPOTHESES.md "B6 write-back probe pre-registration").
#   bash benchmarks/scripts/wb_probe.sh           # binding run: 3 x per-kernel + 3 x range replay
#   bash benchmarks/scripts/wb_probe.sh --smoke   # 3 off-grid cells, 1 run each, table only
# Fixed before any grid cell was measured:
#   --replay-mode application  per-launch counters; every pass reruns the whole binary
#   --replay-mode app-range    steady launches 2..S of each cell as one range, launched back to
#                              back (no per-kernel serialization gaps) -> W-5
#   --cache-control none       ncu must not flush caches between launches
#   --clock-control none       bytes, not time, are scored
# Needs GPU performance-counter access (DECISION_LOG D-031); exits 3 on ERR_NVGPUCTRPERM.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
RUNS=3; BIN_ARGS=(); [ "$SMOKE" = 1 ] && { RUNS=1; BIN_ARGS=(--smoke); }
OUT="benchmarks/results/wb_probe_$([ "$SMOKE" = 1 ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
BIN=benchmarks/build/wb_probe
[ -x "$BIN" ] || { echo "missing $BIN (ONLY=wb_probe CUDA_ARCH=sm_86 bash benchmarks/scripts/build.sh)"; exit 2; }
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
NCU=(ncu --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)

{
  date -u; uname -a; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD; git status --short
  sha256sum "$0" "$BIN" benchmarks/src/wb_probe.cu benchmarks/src/common.cuh benchmarks/src/dmk_host.hpp \
    benchmarks/scripts/wb_probe_score.py benchmarks/scripts/counter_prereg.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

blocked() { grep -q "ERR_NVGPUCTRPERM" "$1" && { echo "ERR_NVGPUCTRPERM: counters unavailable" | tee -a "$OUT/STATUS"; exit 3; }; }
KCSV=(); RCSV=()
for i in $(seq 1 "$RUNS"); do
  "${NCU[@]}" --replay-mode application --export "$OUT/kernel_r$i" "$BIN" "${BIN_ARGS[@]}" \
    --meta "$OUT/kernel_r${i}_meta.json" > "$OUT/kernel_r${i}_ncu.txt" 2>&1
  echo "kernel run $i rc=$?" | tee -a "$OUT/STATUS"; blocked "$OUT/kernel_r${i}_ncu.txt"
  ncu --import "$OUT/kernel_r$i.ncu-rep" --csv --page raw --print-units base > "$OUT/kernel_r$i.csv"
  KCSV+=("$OUT/kernel_r$i.csv")
done
for i in $(seq 1 "$RUNS"); do
  "${NCU[@]}" --replay-mode app-range --export "$OUT/range_r$i" "$BIN" "${BIN_ARGS[@]}" --profiler-ranges \
    --meta "$OUT/range_r${i}_meta.json" > "$OUT/range_r${i}_ncu.txt" 2>&1
  echo "range run $i rc=$?" | tee -a "$OUT/STATUS"; blocked "$OUT/range_r${i}_ncu.txt"
  ncu --import "$OUT/range_r$i.ncu-rep" --csv --page raw --print-units base > "$OUT/range_r$i.csv"
  RCSV+=("$OUT/range_r$i.csv")
done

python3 - "$OUT" <<'EOF' | tee -a "$OUT/STATUS"
import json, pathlib, sys
fails = {p.name: json.loads(p.read_text())["correctness_failures"] for p in pathlib.Path(sys.argv[1]).glob("*_meta.json")}
print("correctness_failures per run:", fails)
EOF
SCORE_ARGS=(--kernel-csv "${KCSV[@]}" --range-csv "${RCSV[@]}" --meta "$OUT/kernel_r1_meta.json")
[ "$SMOKE" = 1 ] && SCORE_ARGS+=(--table-only)
python3 benchmarks/scripts/wb_probe_score.py "${SCORE_ARGS[@]}" --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
grep -E "^W-|verdict" "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
