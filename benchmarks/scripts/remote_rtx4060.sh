#!/usr/bin/env bash
# Binding measurement protocol for the RTX 4060 test of H2 (pre-registered 2026-11-04; see
# research/HYPOTHESES.md "RTX 4060 pre-registration"). NOT yet run; needs PI approval of a rental.
#   bash benchmarks/scripts/remote_rtx4060.sh
# Steps (no free choices after the fact):
#   1. environment + unit tests, build for sm_89 (Ada)
#   2. B1 smoke at 256 MiB footprint (>= 10x L2): memcpy BW for calibration + H1 secondary check
#   3. B4 sweep, K=1, 3 independent runs, sizes 0.0625 ... 4 x L2 (4 = calibration point)
#   4. exploratory (not scored): B3 fetch-granularity probe, default and limit=64, 256 MiB table
# Scoring happens off-host (see HYPOTHESES.md): calibrate tau on run 1 at l2x=4, BW from step 2,
# regenerate predictions deterministically, score the median of the 3 runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="benchmarks/results/rtx4060_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
step() { echo "== $(date -u +%H:%M:%S) $*" | tee -a "$OUT/steps.log"; }
export CUDA_ARCH="${CUDA_ARCH:-sm_89}"

step "python3"
command -v python3 >/dev/null 2>&1 || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive \
  apt-get install -y -qq --no-install-recommends python3) > "$OUT/apt_python3.txt" 2>&1

step "environment"
{
  date -u; uname -a; head -4 /etc/os-release; nproc; free -h; grep -m1 "model name" /proc/cpuinfo
  nvidia-smi
  nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,clocks.max.sm,clocks.max.mem,temperature.gpu,power.limit,pstate,pcie.link.gen.max,pcie.link.width.max --format=csv
  nvcc --version; python3 --version; gcc --version | head -1
  grep -i profil /proc/driver/nvidia/params 2>&1
} > "$OUT/environment.txt" 2>&1

step "unit tests"
python3 -m unittest discover -s benchmarks/scripts -p 'test_*.py' > "$OUT/unittest.txt" 2>&1
echo "unittest rc=$?" | tee -a "$OUT/steps.log"
find benchmarks -name __pycache__ -prune -exec rm -rf {} +

step "build $CUDA_ARCH"
bash benchmarks/scripts/build.sh > "$OUT/build.txt" 2>&1
echo "build rc=$?" | tee -a "$OUT/steps.log"
cp benchmarks/build/*.log benchmarks/build/*.txt "$OUT/" 2>/dev/null

step "B1 smoke, 256 MiB footprint"
python3 benchmarks/scripts/run_copy_stride.py --smoke --footprint-mib 256 --tag rtx4060 > "$OUT/b1.txt" 2>&1
echo "B1 rc=$?" | tee -a "$OUT/steps.log"

for r in 1 2 3; do
  step "B4 sweep rep $r"
  python3 benchmarks/scripts/run_bench.py fusion --tag "rtx4060-sweep-r$r" -- \
    --sizes-l2x 0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,2,4 --ks 1 \
    --reps 30 --warmup 5 --steady-iters 10 > "$OUT/b4_sweep_r$r.txt" 2>&1
  echo "B4 sweep r$r rc=$?" | tee -a "$OUT/steps.log"
done

for spec in "default:-1" "g64:64"; do
  tag="${spec%%:*}"; g="${spec##*:}"
  step "B3 probe (exploratory) $tag"
  python3 benchmarks/scripts/run_bench.py gather --tag "rtx4060-probe-$tag" -- \
    --table-mib 256 --lookups 4194304 --dists uniform,zipf1.0 \
    --variants gather,gather_pair64,gather_pair128 --reps 20 --warmup 3 --steady-iters 5 \
    --l2-fetch-granularity "$g" > "$OUT/b3_probe_$tag.txt" 2>&1
  echo "B3 probe $tag rc=$?" | tee -a "$OUT/steps.log"
done

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > "$OUT/gpu_after.txt" 2>&1
step "done"
tar czf /tmp/dmkl_rtx4060.tgz benchmarks/results 2>/dev/null
echo "archive: /tmp/dmkl_rtx4060.tgz" | tee -a "$OUT/steps.log"
