#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-1}"
EXPONENT="${2:-175000039}"
cd "$ROOT"
make -j"$(nproc)" engine-lib
mkdir -p build-tests
${CXX:-c++} -O2 -std=c++20 tests/pow2_type4_lead_cache_compare.cpp -ldl \
  -o build-tests/pow2-type4-lead-cache-compare
./build-tests/pow2-type4-lead-cache-compare \
  "$ROOT/build-engine/libaevum_engine.so" "$DEVICE" "$EXPONENT"
