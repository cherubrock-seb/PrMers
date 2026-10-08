#!/usr/bin/env bash
# Build and run the legacy kernel_carry_mul_3 device test (needs an OpenCL GPU device, libgmp).
# usage: tests/test_legacy_carry_mul3_device.sh [device-index] [p ...]
# The legacy Context enumerates GPU devices only; to run on a CPU OpenCL device (e.g. PoCL) preload a
# clGetDeviceIDs wrapper that maps CL_DEVICE_TYPE_GPU to CL_DEVICE_TYPE_ALL via PRMERS_TEST_PRELOAD=<lib.so>.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-legacy-carry-mul3"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DPRMERS_KERNEL_DIR=\""$ROOT/kernels"\" \
  -I"$ROOT/include" \
  "$ROOT/tests/legacy_carry_mul3_device_test.cpp" \
  "$ROOT/src/math/Precompute.cpp" "$ROOT/src/math/Mod64.cpp" "$ROOT/src/math/Carry.cpp" \
  "$ROOT/src/opencl/Context.cpp" "$ROOT/src/opencl/Program.cpp" "$ROOT/src/opencl/Buffers.cpp" \
  "$ROOT/src/util/OpenCLError.cpp" \
  -o "$BUILD/legacy-carry-mul3-device-test" \
  -lOpenCL -lgmp

LD_PRELOAD="${PRMERS_TEST_PRELOAD:-}" "$BUILD/legacy-carry-mul3-device-test" "$@"
