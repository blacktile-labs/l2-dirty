#!/usr/bin/env bash
# B10b first-cell test, rules F-0..F-2 and V-4' (research/HYPOTHESES.md "B11 pre-registration").
# Same binary, ncu flags and launch sequence as the B6b sector set (wb_probe_b6b.sh), with ONE change:
# a dummy cell 0:0.25 is prepended, so the V-4 control 0:0.375 is no longer the first cell of the
# process. 3x --replay-mode application + 3x --replay-mode app-range.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v ncu >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
OUT="benchmarks/results/wb_probe_firstcell_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
BIN=benchmarks/build/wb_probe
[ -x "$BIN" ] || { echo "missing $BIN"; exit 2; }
CELLS=$(python3 benchmarks/scripts/wb_probe_firstcell_score.py cells)
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,gpu__time_duration.sum
NCU=(ncu --cache-control none --clock-control none --metrics "$METRICS" --force-overwrite)
{
  date -u; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv
  ncu --version; echo "CELLS=$CELLS"
  sha256sum "$0" "$BIN" benchmarks/src/wb_probe.cu benchmarks/scripts/wb_probe_firstcell_score.py \
    benchmarks/scripts/wb_probe_b6b_score.py benchmarks/scripts/wb_probe_score.py
} > "$OUT/environment.txt" 2>&1
for mode in kernel range; do
  for i in 1 2 3; do
    tag="sector_${mode}_r$i"; rm=application; extra=()
    [ "$mode" = range ] && { rm=app-range; extra=(--profiler-ranges); }
    "${NCU[@]}" --replay-mode "$rm" --export "$OUT/$tag" "$BIN" --cells "$CELLS" --one-sector "${extra[@]}" \
      --meta "$OUT/${tag}_meta.json" > "$OUT/${tag}_ncu.txt" 2>&1
    echo "$tag rc=$?" | tee -a "$OUT/STATUS"
    grep -q "ERR_NVGPUCTRPERM" "$OUT/${tag}_ncu.txt" && { echo "ERR_NVGPUCTRPERM" | tee -a "$OUT/STATUS"; exit 3; }
    ncu --import "$OUT/$tag.ncu-rep" --csv --page raw --print-units base > "$OUT/$tag.csv"
  done
done
python3 benchmarks/scripts/wb_probe_firstcell_score.py score --meta "$OUT/sector_kernel_r1_meta.json" \
  --kernel-csv "$OUT"/sector_kernel_r{1,2,3}.csv --range-csv "$OUT"/sector_range_r{1,2,3}.csv \
  --meta-glob "$OUT/*_meta.json" --json-out "$OUT/score.json" > "$OUT/score.txt" 2>&1
echo "score rc=$?" | tee -a "$OUT/STATUS"
grep -E "^F-|^V-4|verdict" "$OUT/score.txt" | tee -a "$OUT/STATUS"
echo "results: $OUT"
