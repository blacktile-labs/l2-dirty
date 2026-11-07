#!/usr/bin/env bash
# Build the CUDA baselines into benchmarks/build/.
#   CUDA_ARCH=sm_86 bash benchmarks/scripts/build.sh     # explicit (RTX 30xx = sm_86)
#   bash benchmarks/scripts/build.sh                      # -arch=native (needs a visible GPU, nvcc >= 11.6)
#   NVCC=/path/to/nvcc CUDA_ARCH=sm_86 bash benchmarks/scripts/build.sh
#   ONLY=transpose bash benchmarks/scripts/build.sh       # build a single benchmark
# ptxas -v output (registers, spills, smem per kernel) is saved per binary: it is a Gate C
# input (register pressure / occupancy), not a performance result.
# transpose is first tried with cuBLAS (-DWITH_CUBLAS -lcublas, vendor baseline); if that
# fails to compile or link, it is rebuilt without cuBLAS and the fallback is recorded.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(dirname "$HERE")"
NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-native}"
OUT="$BENCH/build"
mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas -v)
"$NVCC" --version | tee "$OUT/nvcc_version.txt"
echo "flags: ${FLAGS[*]}" | tee "$OUT/build_flags.txt"

build() {  # name [extra flags...]
  local name="$1"; shift
  "$NVCC" "${FLAGS[@]}" "$@" -o "$OUT/$name" "$BENCH/src/$name.cu" 2> "$OUT/ptxas_$name.log"
}

for b in copy_stride transpose gather fusion stencil wb_probe; do
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$b" ]; then continue; fi
  if [ "$b" = transpose ] && [ "${NO_CUBLAS:-0}" != 1 ]; then
    if build transpose -DWITH_CUBLAS -lcublas; then
      echo "transpose: built WITH_CUBLAS" | tee -a "$OUT/build_flags.txt"
    else
      echo "transpose: cuBLAS build failed (see ptxas_transpose.log); rebuilding without cuBLAS" \
        | tee -a "$OUT/build_flags.txt"
      cp "$OUT/ptxas_transpose.log" "$OUT/cublas_attempt_transpose.log"
      build transpose
    fi
  else
    build "$b"
  fi
  if grep -qE "warning|error" "$OUT/ptxas_$b.log"; then
    echo "---- $b: compiler warnings/errors ----"; grep -E "warning|error" "$OUT/ptxas_$b.log" | head -20
  fi
  echo "built: $OUT/$b"
done
