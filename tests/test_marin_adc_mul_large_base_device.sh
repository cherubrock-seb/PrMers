#!/usr/bin/env bash
# Build and run the Marin large multiplier (adc_mul) device test (needs an OpenCL device, libgmp).
# usage: tests/test_marin_adc_mul_large_base_device.sh [device-index] [q ...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-adc-mul-base"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DGPU \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_adc_mul_large_base_test.cpp" \
  -o "$BUILD/marin-adc-mul-large-base-device-test" \
  -lOpenCL -lgmp

"$BUILD/marin-adc-mul-large-base-device-test" "$@"
