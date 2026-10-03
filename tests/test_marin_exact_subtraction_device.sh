#!/usr/bin/env bash
# Build and run the Marin exact subtraction device test (needs an OpenCL device, libgmp).
# usage: tests/test_marin_exact_subtraction_device.sh [device-index] [q ...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-exact-sub"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DGPU \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_exact_subtraction_device_test.cpp" \
  -o "$BUILD/marin-exact-subtraction-device-test" \
  -lOpenCL -lgmp

"$BUILD/marin-exact-subtraction-device-test" "$@"
