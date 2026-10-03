#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-pm1-bounds"
mkdir -p "$BUILD"
"${CXX:-g++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/test_pm1_bounds.cpp" \
  "$ROOT/src/math/Pm1Bounds.cpp" \
  -o "$BUILD/test_pm1_bounds"
"$BUILD/test_pm1_bounds"
