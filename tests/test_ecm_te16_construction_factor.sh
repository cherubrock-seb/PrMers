#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-ecm-te16-construction-factor"

rm -rf "$BUILD"
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  ${GMP_PREFIX:+-I"$GMP_PREFIX/include" -L"$GMP_PREFIX/lib"} \
  "$ROOT/tests/ecm_te16_construction_factor_test.cpp" \
  -o "$BUILD/ecm-te16-construction-factor-test" \
  -lgmpxx -lgmp

"$BUILD/ecm-te16-construction-factor-test"
rm -rf "$BUILD"
