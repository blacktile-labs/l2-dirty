#!/usr/bin/env bash
# B6b confirmatory write-back probe, rules V-0..V-5 (pre-registered 2026-11-05 before any B6b cell
# was measured; research/HYPOTHESES.md "B6b pre-registration").
#   bash benchmarks/scripts/wb_probe_b6b.sh           # binding run
#   bash benchmarks/scripts/wb_probe_b6b.sh --smoke   # off-grid smoke cells, 1 run per set, tables only
# Two cell sets (cells are constants in wb_probe_b6b_score.py), each 3x per-kernel
# (--replay-mode application) and 3x back-to-back (--replay-mode app-range, steady 2..5):
#   full    whole 128-B lines written (rw_kernel, unchanged since B6: identical SASS)
#   sector  one 32-B sector written per 128-B line (rw1s_kernel, --one-sector)
# ncu flags as in B6: --cache-control none --clock-control none.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
RUNS=3; [ "$SMOKE" = 1 ] && RUNS=1
OUT="benchmarks/results/wb_probe_b6b_$([ "$SMOKE" = 1 ] && echo smoke_)$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
BIN=benchmarks/build/wb_probe
[ -x "$BIN" ] || { echo "missing $BIN"; exit 2; }
read -r FULL SECT < <(python3 -c "import sys; sys.path.insert(0, 'benchmarks/scripts'); import wb_probe_b6b_score as s; print(s.FULL_CELLS, s.SECTOR_CELLS)")
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
NCU=(ncu --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)

{
  date -u; uname -a; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; git rev-parse HEAD; git status --short
  sha256sum "$0" "$BIN" benchmarks/src/wb_probe.cu benchmarks/src/common.cuh benchmarks/src/dmk_host.hpp \
    benchmarks/scripts/wb_probe_b6b_score.py benchmarks/scripts/wb_probe_score.py benchmarks/scripts/counter_prereg.py
  echo "FULL=$FULL"; echo "SECT=$SECT"
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
blocked() { grep -q "ERR_NVGPUCTRPERM" "$1" && { echo "ERR_NVGPUCTRPERM: counters unavailable" | tee -a "$OUT/STATUS"; exit 3; }; }

run_set() {  # name, cells, extra binary args...
  local name="$1" cells="$2"; shift 2
  local cell_args=(--cells "$cells"); [ "$SMOKE" = 1 ] && cell_args=(--smoke)
  for mode in kernel range; do
    for i in $(seq 1 "$RUNS"); do
      local tag="${name}_${mode}_r$i" rm=application extra=()
      [ "$mode" = range ] && { rm=app-range; extra=(--profiler-ranges); }
      "${NCU[@]}" --replay-mode "$rm" --export "$OUT/$tag" "$BIN" "${cell_args[@]}" "$@" "${extra[@]}" \
        --meta "$OUT/${tag}_meta.json" > "$OUT/${tag}_ncu.txt" 2>&1
      echo "$tag rc=$?" | tee -a "$OUT/STATUS"; blocked "$OUT/${tag}_ncu.txt"
      ncu --import "$OUT/$tag.ncu-rep" --csv --page raw --print-units base > "$OUT/$tag.csv"
    done
  done
}
run_set full "$FULL"
run_set sector "$SECT" --one-sector

python3 - "$OUT" <<'EOF' | tee -a "$OUT/STATUS"
import json, pathlib, sys
fails = {p.name: json.loads(p.read_text())["correctness_failures"] for p in sorted(pathlib.Path(sys.argv[1]).glob("*_meta.json"))}
print("correctness_failures per run:", fails)
EOF
S=benchmarks/scripts
if [ "$SMOKE" = 1 ]; then
  for name in full sector; do
    python3 $S/wb_probe_score.py --kernel-csv "$OUT/${name}_kernel_r1.csv" --range-csv "$OUT/${name}_range_r1.csv" \
      --meta "$OUT/${name}_kernel_r1_meta.json" --table-only > "$OUT/${name}_table.txt" 2>&1
    echo "== $name"; cat "$OUT/${name}_table.txt"
  done
else
  python3 $S/wb_probe_b6b_score.py \
    --full-meta "$OUT/full_kernel_r1_meta.json" --full-kernel-csv "$OUT"/full_kernel_r{1,2,3}.csv \
    --full-range-csv "$OUT"/full_range_r{1,2,3}.csv \
    --sector-meta "$OUT/sector_kernel_r1_meta.json" --sector-kernel-csv "$OUT"/sector_kernel_r{1,2,3}.csv \
    --sector-range-csv "$OUT"/sector_range_r{1,2,3}.csv --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
  echo "score rc=$?" | tee -a "$OUT/STATUS"
  grep -E "^V-|verdict" "$OUT/score.txt" | tee -a "$OUT/STATUS"
fi
echo "results: $OUT"
