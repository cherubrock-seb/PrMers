#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-quick-checker"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/quick_checker_test.cpp" \
  "$ROOT/src/core/QuickChecker.cpp" \
  -o "$BUILD/quick-checker-test"

"$BUILD/quick-checker-test"
