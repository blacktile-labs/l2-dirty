#!/usr/bin/env bash
# B20 (research/HYPOTHESES.md "B20 pre-registration"): the B9b protocol (queued_wb_probe.sh, unchanged) run
# with the newest Nsight Compute first and the original 2024.2.1 second, back to back on the laptop GPU.
# NEW_NCU_DIR: directory holding the extracted newer `ncu` (not installed system-wide).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
NEW_NCU_DIR="${NEW_NCU_DIR:-$HOME/.cache/ncu-2026.4.0/root/opt/nvidia/nsight-compute/2026.4.0}"
OLD_NCU_DIR=/usr/local/cuda-12.5/bin
OUT="benchmarks/results/ncu_version_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
{ date -u; "$NEW_NCU_DIR/ncu" --version; "$OLD_NCU_DIR/ncu" --version; sha256sum "$0" benchmarks/scripts/queued_wb_probe.sh \
  benchmarks/scripts/queued_wb_probe.py benchmarks/scripts/queued_wb_score.py benchmarks/scripts/ncu_version_score.py; } > "$OUT/environment.txt" 2>&1
for tag in new old; do
  dir=$NEW_NCU_DIR; [ "$tag" = old ] && dir=$OLD_NCU_DIR
  PATH="$dir:$PATH" bash benchmarks/scripts/queued_wb_probe.sh > "$OUT/${tag}.log" 2>&1
  res=$(grep -o "results: .*" "$OUT/${tag}.log" | tail -1 | cut -d' ' -f2)
  echo "$tag -> $res" | tee -a "$OUT/STATUS"
  echo "$res" > "$OUT/${tag}_dir.txt"
done
python3 benchmarks/scripts/ncu_version_score.py --old "$(cat "$OUT/old_dir.txt")/score.json" \
  --new "$(cat "$OUT/new_dir.txt")/score.json" --json-out "$OUT/score.json" | tee "$OUT/score.txt"
echo "results: $OUT"
