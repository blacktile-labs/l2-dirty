#!/usr/bin/env bash
# Follow-up experiments for a rented GPU host (prepared 2026-11-04; NOT yet run).
# Requires PI approval of a rental. Run from the project root after remote_smoke.sh succeeded:
#   bash benchmarks/scripts/remote_followup.sh
# 1. B4 fine sweep around the L2 cliff, K=1, 3 independent repetitions
#    -> scored against research/preregistration/B4_v2_predictions.json
# 2. B3 fetch-granularity probe (gather vs. gather_pair64 / gather_pair128), random dists,
#    with cudaLimitMaxL2FetchGranularity left at default and set to 32 / 64 / 128,
#    default repeated 3 times
# 3. B1 smoke once more (card identity / run-to-run drift check)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="benchmarks/results/followup_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
step() { echo "== $(date -u +%H:%M:%S) $*" | tee -a "$OUT/steps.log"; }
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,driver_version,temperature.gpu,pstate --format=csv > "$OUT/gpu.txt" 2>&1

[ -x benchmarks/build/fusion ] || CUDA_ARCH="${CUDA_ARCH:-sm_86}" bash benchmarks/scripts/build.sh > "$OUT/build.txt" 2>&1

for r in 1 2 3; do
  step "B4 sweep rep $r"
  python3 benchmarks/scripts/run_bench.py fusion --tag "sweep-r$r" -- \
    --sizes-l2x 0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2 --ks 1 --reps 30 --warmup 5 --steady-iters 10 \
    > "$OUT/fusion_sweep_r$r.txt" 2>&1
  echo "fusion sweep r$r rc=$?" | tee -a "$OUT/steps.log"
done

for spec in "default:-1" "g32:32" "g64:64" "g128:128" "default2:-1" "default3:-1"; do
  tag="${spec%%:*}"; g="${spec##*:}"
  step "B3 probe $tag"
  python3 benchmarks/scripts/run_bench.py gather --tag "probe-$tag" -- \
    --table-mib 64 --lookups 4194304 --dists uniform,zipf1.0 \
    --variants gather,gather_pair64,gather_pair128 --reps 20 --warmup 3 --steady-iters 5 \
    --l2-fetch-granularity "$g" > "$OUT/gather_probe_$tag.txt" 2>&1
  echo "gather probe $tag rc=$?" | tee -a "$OUT/steps.log"
done

step "B1 smoke (identity / drift)"
python3 benchmarks/scripts/run_copy_stride.py --smoke --tag followup > "$OUT/copy_stride.txt" 2>&1
echo "copy_stride rc=$?" | tee -a "$OUT/steps.log"

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > "$OUT/gpu_after.txt" 2>&1
step "done"
tar czf /tmp/dmkl_followup.tgz benchmarks/results 2>/dev/null
echo "archive: /tmp/dmkl_followup.tgz" | tee -a "$OUT/steps.log"
