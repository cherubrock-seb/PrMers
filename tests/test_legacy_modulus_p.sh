#!/usr/bin/env bash
# Build and run the legacy MODULUS_P test. The host part needs no OpenCL device; the kernel part uses a CPU
# OpenCL device (for example PoCL) only and skips itself when there is none.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-legacy-modulus-p"

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
  "$ROOT/tests/legacy_modulus_p_test.cpp" \
  "$ROOT/src/math/Precompute.cpp" "$ROOT/src/math/Mod64.cpp" \
  "$ROOT/src/opencl/Context.cpp" "$ROOT/src/opencl/Program.cpp" \
  "$ROOT/src/util/OpenCLError.cpp" \
  -o "$BUILD/legacy-modulus-p-test" \
  -lOpenCL -lgmp

"$BUILD/legacy-modulus-p-test"
