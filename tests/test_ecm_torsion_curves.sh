#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-ecm-torsion"

rm -rf "$BUILD"
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  ${GMP_PREFIX:+-I"$GMP_PREFIX/include" -L"$GMP_PREFIX/lib"} \
  "$ROOT/tests/ecm_torsion_curves_test.cpp" \
  -o "$BUILD/ecm-torsion-curves-test" \
  -lgmpxx -lgmp

"$BUILD/ecm-torsion-curves-test"
