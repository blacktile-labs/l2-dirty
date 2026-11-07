#!/usr/bin/env bash
# Build and run the CPU-only C++ tests for benchmarks/src/dmk_host.hpp (no GPU needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(dirname "$HERE")"
mkdir -p "$BENCH/build"
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra -o "$BENCH/build/test_host" "$BENCH/src/test_host.cpp"
"$BENCH/build/test_host"
