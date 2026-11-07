#!/usr/bin/env bash
# B8 replacement near L2 capacity, rules Z-0..Z-3 (pre-registered 2026-11-05 before any B8 cell was
# measured; research/HYPOTHESES.md "B8 pre-registration"; predictions in
# research/preregistration/b8_predictions.json). Same binary, ncu flags and launch sequence as
# B6b (full lines): 3x --replay-mode application + 3x --replay-mode app-range.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
OUT="benchmarks/results/wb_probe_b8_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
BIN=benchmarks/build/wb_probe
PREREG=research/preregistration/b8_predictions.json
CELLS=$(python3 benchmarks/scripts/wb_probe_b8_score.py cells)
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
NCU=(ncu --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD; echo "CELLS=$CELLS"
  sha256sum "$0" "$BIN" "$PREREG" benchmarks/src/wb_probe.cu benchmarks/scripts/wb_probe_b8_score.py \
    benchmarks/scripts/wb_probe_score.py benchmarks/scripts/cache_sim_v4.py benchmarks/scripts/cache_sim_v5cand.py
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
for mode in kernel range; do
  for i in 1 2 3; do
    tag="${mode}_r$i"; rm=application; extra=()
    [ "$mode" = range ] && { rm=app-range; extra=(--profiler-ranges); }
    "${NCU[@]}" --replay-mode "$rm" --export "$OUT/$tag" "$BIN" --cells "$CELLS" "${extra[@]}" \
      --meta "$OUT/${tag}_meta.json" > "$OUT/${tag}_ncu.txt" 2>&1
    echo "$tag rc=$?" | tee -a "$OUT/STATUS"
    grep -q "ERR_NVGPUCTRPERM" "$OUT/${tag}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
    ncu --import "$OUT/$tag.ncu-rep" --csv --page raw --print-units base > "$OUT/$tag.csv"
  done
done
python3 - "$OUT" <<'PY' | tee -a "$OUT/STATUS"
import json, pathlib, sys
print("correctness_failures per run:", {p.name: json.loads(p.read_text())["correctness_failures"] for p in sorted(pathlib.Path(sys.argv[1]).glob("*_meta.json"))})
PY
python3 benchmarks/scripts/wb_probe_b8_score.py score --prereg "$PREREG" --meta "$OUT/kernel_r1_meta.json" \
  --kernel-csv "$OUT"/kernel_r{1,2,3}.csv --range-csv "$OUT"/range_r{1,2,3}.csv --json-out "$OUT/score.json" \
  > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
grep -E "^Z-|verdict" "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
