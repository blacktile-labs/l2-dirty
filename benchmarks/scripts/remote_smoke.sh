#!/usr/bin/env bash
# Bounded B1-B4 smoke on a freshly provisioned NVIDIA GPU host (e.g. a rented container).
# Run from the project root:  bash benchmarks/scripts/remote_smoke.sh
# Everything it records lands in benchmarks/results/ ; nothing else on the host is modified
# except installing python3 when the image lacks it (CUDA images usually do).
# It runs exactly the smoke configuration: no sweeps, no extra experiments.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="benchmarks/results/remote_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
step() { echo "== $(date -u +%H:%M:%S) $*" | tee -a "$OUT/steps.log"; }

step "python3"
if ! command -v python3 >/dev/null 2>&1; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3) \
    > "$OUT/apt_python3.txt" 2>&1
fi

step "environment"
{
  echo "### date";            date -u
  echo "### uname";           uname -a
  echo "### os-release";      head -4 /etc/os-release
  echo "### cpu/mem";         nproc; free -h; grep -m1 "model name" /proc/cpuinfo
  echo "### nvidia-smi";      nvidia-smi
  echo "### query";           nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw,power.limit,pstate,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
  echo "### clocks/perf";     nvidia-smi -q -d CLOCK,PERFORMANCE
  echo "### nvcc";            nvcc --version
  echo "### tools";           for t in ncu nsys compute-sanitizer cuobjdump; do printf '%s: ' "$t"; command -v "$t" || echo "not found"; done
  echo "### ncu version";     ncu --version 2>&1 | head -3
  echo "### profiling param"; grep -i profil /proc/driver/nvidia/params 2>&1
  echo "### python";          python3 --version
  echo "### gcc";             gcc --version | head -1
} > "$OUT/environment.txt" 2>&1

step "unit tests (CPU: Python + C++ host)"
python3 -m unittest discover -s benchmarks/scripts -p 'test_*.py' -v > "$OUT/unittest.txt" 2>&1
echo "unittest rc=$?" | tee -a "$OUT/steps.log"
find benchmarks -name __pycache__ -prune -exec rm -rf {} +
bash benchmarks/scripts/test_host_cpp.sh > "$OUT/test_host_cpp.txt" 2>&1
echo "test_host_cpp rc=$?" | tee -a "$OUT/steps.log"

step "build sm_86"
CUDA_ARCH="${CUDA_ARCH:-sm_86}" bash benchmarks/scripts/build.sh > "$OUT/build.txt" 2>&1
echo "build rc=$?" | tee -a "$OUT/steps.log"
cp benchmarks/build/ptxas_*.log benchmarks/build/build_flags.txt "$OUT/" 2>/dev/null
cp benchmarks/build/cublas_attempt_transpose.log "$OUT/" 2>/dev/null

step "smoke B1 copy_stride"
python3 benchmarks/scripts/run_copy_stride.py --smoke --tag remote > "$OUT/smoke_copy_stride.txt" 2>&1
echo "smoke copy_stride rc=$?" | tee -a "$OUT/steps.log"
for b in transpose gather fusion; do
  step "smoke $b"
  python3 benchmarks/scripts/run_bench.py "$b" --smoke --tag remote > "$OUT/smoke_$b.txt" 2>&1
  echo "smoke $b rc=$?" | tee -a "$OUT/steps.log"
done

step "compute-sanitizer memcheck (reduced sizes, 1 rep)"
if command -v compute-sanitizer >/dev/null 2>&1; then
  B=benchmarks/build
  declare -A MC_ARGS=(
    [copy_stride]="--smoke --reps 1 --warmup 0"
    [transpose]="--shapes 1000x777,64x96 --reps 1 --warmup 0 --steady-iters 1"
    [gather]="--table-mib 4 --lookups 262144 --dists seq,uniform,zipf1.0 --reps 1 --warmup 0 --steady-iters 1"
    [fusion]="--sizes-l2x 0.25,1 --ks 1,8 --reps 1 --warmup 0 --steady-iters 1"
  )
  for b in copy_stride transpose gather fusion; do
    # shellcheck disable=SC2086
    compute-sanitizer --tool memcheck --leak-check full "$B/$b" ${MC_ARGS[$b]} \
      --csv "$OUT/memcheck_$b.csv" --meta "$OUT/memcheck_$b.json" > "$OUT/memcheck_$b.txt" 2>&1
    echo "memcheck $b rc=$? $(grep -h 'ERROR SUMMARY' "$OUT/memcheck_$b.txt")" | tee -a "$OUT/steps.log"
  done
else
  echo "compute-sanitizer: unavailable" | tee -a "$OUT/steps.log"
fi

step "ncu counters (may be permission-blocked)"
bash benchmarks/scripts/profile_ncu.sh > "$OUT/ncu.txt" 2>&1
echo "ncu rc=$?" | tee -a "$OUT/steps.log"

step "post-run nvidia-smi"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > "$OUT/nvidia_smi_after.txt" 2>&1

step "done"
tar czf /tmp/dmkl_results.tgz benchmarks/results benchmarks/build/*.log benchmarks/build/*.txt 2>/dev/null
echo "archive: /tmp/dmkl_results.tgz" | tee -a "$OUT/steps.log"
