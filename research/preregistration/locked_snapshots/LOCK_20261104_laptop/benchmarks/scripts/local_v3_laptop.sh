#!/usr/bin/env bash
# Binding measurement protocol for the RTX 3060 Laptop test of H2 / model v3 (pre-registered
# 2026-11-04 before any B4/B5 run on this GPU; research/HYPOTHESES.md "RTX 3060 Laptop
# pre-registration"). Local GPU, no rental. Identical to remote_v3.sh except:
#   * one extra size, l2x 0.46875 = 1.40625 MiB per array = the byte size at which the rented
#     desktop RTX 3060 (L2 2.25 MiB) sits at l2x 0.625, i.e. on its cliff (byte control L-3)
#   * no apt step; nvcc from /usr/local/cuda-12.5 if not on PATH
#   * a 1 Hz nvidia-smi log (clocks, power, temperature, throttle reasons) for the whole run
# Steps (no free choices after the fact):
#   1. environment + unit tests + C++ host tests; build copy_stride, fusion, stencil (sm_86)
#   2. B1 smoke at 256 MiB (identity / drift check only)
#   3. B5 stencil (mat4, fused2, fused2c, sweep warm/cold): 3 runs
#   4. B4 fusion (K=1): 3 runs at the same sizes
# Scoring: model_v3.py predict --calib-run <B5/B4 r1> (fits at l2x 2 and 4), then score r1..r3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_ARCH=sm_86
command -v nvcc >/dev/null 2>&1 || export PATH=/usr/local/cuda-12.5/bin:$PATH
OUT="benchmarks/results/laptop_v3_${CUDA_ARCH}_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
step() { echo "== $(date -u +%H:%M:%S) $*" | tee -a "$OUT/steps.log"; }
SIZES="0.125,0.46875,0.5625,0.6875,0.8125,0.9375,1.125,1.375,1.75,2,2.5,3,4"

nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_event_reasons.active \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

step "environment"
{
  date -u; uname -a; head -4 /etc/os-release; nproc; free -h; grep -m1 "model name" /proc/cpuinfo
  nvidia-smi
  nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,clocks.max.sm,clocks.max.mem,temperature.gpu,power.limit,pstate --format=csv
  nvcc --version; python3 --version; gcc --version | head -1
  echo "--- Windows host power source (BatteryStatus 2 = on AC), best effort"
  timeout 20 powershell.exe -NoProfile -Command "Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | Format-List" 2>&1 | tr -d '\r'
  echo "--- sha256 of protocol inputs"
  sha256sum benchmarks/scripts/local_v3_laptop.sh benchmarks/scripts/model_v3.py benchmarks/scripts/cache_sim.py \
    benchmarks/scripts/cache_sim_b5.py benchmarks/src/fusion.cu benchmarks/src/stencil.cu benchmarks/src/common.cuh \
    benchmarks/src/stencil_host.hpp
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
python3 benchmarks/scripts/run_copy_stride.py --smoke --footprint-mib 256 --tag "laptop-v3" > "$OUT/b1.txt" 2>&1
echo "B1 rc=$?" | tee -a "$OUT/steps.log"

for r in 1 2 3; do
  step "B5 stencil rep $r"
  python3 benchmarks/scripts/run_bench.py stencil --tag "laptop-v3-r$r" -- \
    --sizes-l2x "$SIZES" --reps 30 --warmup 5 --steady-iters 10 > "$OUT/b5_r$r.txt" 2>&1
  echo "B5 r$r rc=$?" | tee -a "$OUT/steps.log"
done
for r in 1 2 3; do
  step "B4 fusion rep $r"
  python3 benchmarks/scripts/run_bench.py fusion --tag "laptop-v3-r$r" -- \
    --sizes-l2x "$SIZES" --ks 1 --reps 30 --warmup 5 --steady-iters 10 > "$OUT/b4_r$r.txt" 2>&1
  echo "B4 r$r rc=$?" | tee -a "$OUT/steps.log"
done

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > "$OUT/gpu_after.txt" 2>&1
step "done"
