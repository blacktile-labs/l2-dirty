#!/usr/bin/env bash
# Binding measurement protocol for model v3 + B5b ablation (pre-registered 2026-11-04; see
# research/HYPOTHESES.md "Model v3 + B5b pre-registration"). NOT yet run; needs PI approval.
#   CUDA_ARCH=sm_86 bash benchmarks/scripts/remote_v3.sh      # RTX 3060
#   CUDA_ARCH=sm_89 bash benchmarks/scripts/remote_v3.sh      # RTX 4060
# Steps (no free choices after the fact):
#   1. environment + unit tests + C++ host tests; build copy_stride, fusion, stencil
#   2. B1 smoke at 256 MiB (identity / drift check only; v3 fits its own bandwidths)
#   3. B5 stencil (mat4, fused2, fused2c, sweep warm/cold): 3 runs at NEW sizes + calibration 2, 4
#   4. B4 fusion (K=1): 3 runs at the same sizes
# Scoring off-host: model_v3.py predict --calib-run <r1> (l2x 2 and 4), then score r1..r3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_ARCH:?set CUDA_ARCH (sm_86 for RTX 3060, sm_89 for RTX 4060)}"
export CUDA_ARCH
OUT="benchmarks/results/v3_${CUDA_ARCH}_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
step() { echo "== $(date -u +%H:%M:%S) $*" | tee -a "$OUT/steps.log"; }
SIZES="0.125,0.5625,0.6875,0.8125,0.9375,1.125,1.375,1.75,2,2.5,3,4"

step "python3"
command -v python3 >/dev/null 2>&1 || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive \
  apt-get install -y -qq --no-install-recommends python3) > "$OUT/apt_python3.txt" 2>&1

step "environment"
{
  date -u; uname -a; head -4 /etc/os-release; nproc; free -h; grep -m1 "model name" /proc/cpuinfo
  nvidia-smi
  nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,clocks.max.sm,clocks.max.mem,temperature.gpu,power.limit,pstate --format=csv
  nvcc --version; python3 --version; gcc --version | head -1
} > "$OUT/environment.txt" 2>&1

step "tests"
python3 -m unittest discover -s benchmarks/scripts -p 'test_*.py' > "$OUT/unittest.txt" 2>&1
echo "unittest rc=$?" | tee -a "$OUT/steps.log"
bash benchmarks/scripts/test_host_cpp.sh > "$OUT/test_host_cpp.txt" 2>&1
echo "test_host_cpp rc=$?" | tee -a "$OUT/steps.log"
find benchmarks -name __pycache__ -prune -exec rm -rf {} +

step "build $CUDA_ARCH"
for b in copy_stride fusion stencil; do
  ONLY=$b bash benchmarks/scripts/build.sh > "$OUT/build_$b.txt" 2>&1
  echo "build $b rc=$?" | tee -a "$OUT/steps.log"
done
cp benchmarks/build/ptxas_*.log benchmarks/build/nvcc_version.txt benchmarks/build/build_flags.txt "$OUT/" 2>/dev/null

step "B1 smoke, 256 MiB"
python3 benchmarks/scripts/run_copy_stride.py --smoke --footprint-mib 256 --tag "v3-$CUDA_ARCH" > "$OUT/b1.txt" 2>&1
echo "B1 rc=$?" | tee -a "$OUT/steps.log"

for r in 1 2 3; do
  step "B5 stencil rep $r"
  python3 benchmarks/scripts/run_bench.py stencil --tag "v3-$CUDA_ARCH-r$r" -- \
    --sizes-l2x "$SIZES" --reps 30 --warmup 5 --steady-iters 10 > "$OUT/b5_r$r.txt" 2>&1
  echo "B5 r$r rc=$?" | tee -a "$OUT/steps.log"
done
for r in 1 2 3; do
  step "B4 fusion rep $r"
  python3 benchmarks/scripts/run_bench.py fusion --tag "v3-$CUDA_ARCH-r$r" -- \
    --sizes-l2x "$SIZES" --ks 1 --reps 30 --warmup 5 --steady-iters 10 > "$OUT/b4_r$r.txt" 2>&1
  echo "B4 r$r rc=$?" | tee -a "$OUT/steps.log"
done

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > "$OUT/gpu_after.txt" 2>&1
step "done"
tar czf /tmp/dmkl_v3.tgz benchmarks/results 2>/dev/null
echo "archive: /tmp/dmkl_v3.tgz" | tee -a "$OUT/steps.log"
