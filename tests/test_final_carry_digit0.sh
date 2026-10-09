#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-final-carry-digit0"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/final_carry_digit0_test.cpp" \
  "$ROOT/src/math/Carry.cpp" \
  "$ROOT/src/util/OpenCLError.cpp" \
  -o "$BUILD/final-carry-digit0-test" \
  -lOpenCL -lgmpxx -lgmp

"$BUILD/final-carry-digit0-test"
