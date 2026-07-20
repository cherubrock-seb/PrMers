#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-1}"
EXPONENT="${2:-175000039}"
cd "$ROOT"
make -j"$(nproc)" engine-lib
mkdir -p build-tests
${CXX:-c++} -O2 -std=c++20 tests/pfa9_lead_bridge_compare.cpp -ldl \
  -o build-tests/pfa9-lead-bridge-compare
./build-tests/pfa9-lead-bridge-compare \
  "$ROOT/build-engine/libaevum_engine.so" "$DEVICE" "$EXPONENT"
