#!/usr/bin/env bash
# Build and run the legacy check_equal device test (needs an OpenCL device).
# usage: tests/test_legacy_check_equal_device.sh [device-index]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-legacy-check-equal"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DPRMERS_KERNEL_DIR=\"$ROOT/kernels\" \
  -I"$ROOT/include" \
  "$ROOT/tests/legacy_check_equal_device_test.cpp" \
  "$ROOT/src/opencl/Kernels.cpp" \
  "$ROOT/src/util/OpenCLError.cpp" \
  -o "$BUILD/legacy-check-equal-device-test" \
  -lOpenCL

"$BUILD/legacy-check-equal-device-test" "$@"
