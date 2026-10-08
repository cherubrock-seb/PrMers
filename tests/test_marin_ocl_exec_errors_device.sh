#!/usr/bin/env bash
# Build and run the Marin ocl::device execution error test (needs an OpenCL device).
# usage: tests/test_marin_ocl_exec_errors_device.sh [device-index]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-ocl-exec-errors"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DGPU \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_ocl_exec_errors_device_test.cpp" \
  -o "$BUILD/marin-ocl-exec-errors-device-test" \
  -lOpenCL

"$BUILD/marin-ocl-exec-errors-device-test" "$@"
