#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-gaussian-tf-preparser"

rm -rf "$BUILD"
mkdir -p "$BUILD"

declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gmpxx gmp; then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"
fi

declare -a OCL_LIBS=(-lOpenCL)
declare -a OCL_FLAGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  OCL_LIBS=(-framework OpenCL)
  OCL_FLAGS=(-I/System/Library/Frameworks/OpenCL.framework/Headers)
fi

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  -DGPU \
  -DKERNEL_PATH=\"$ROOT/kernels/\" \
  "${OCL_FLAGS[@]}" \
  "${GMP_CFLAGS[@]}" \
  "$ROOT/tests/gaussian_tf_preparser_test.cpp" \
  "$ROOT/src/modes/RunGaussianTrialFactor.cpp" \
  "$ROOT/src/opencl/Context.cpp" \
  "$ROOT/src/io/WorktodoParser.cpp" \
  "$ROOT/src/util/StringUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/math/Pm1Bounds.cpp" \
  "$ROOT/src/ui/WebGuiServer.cpp" \
  "${OCL_LIBS[@]}" \
  "${GMP_LIBS[@]}" \
  -o "$BUILD/gaussian-tf-preparser-test"

"$BUILD/gaussian-tf-preparser-test"
rm -rf "$BUILD"
