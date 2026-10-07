#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-error-check-retry"

rm -rf "$BUILD"
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/error_check_retry_test.cpp" \
  -o "$BUILD/error-check-retry-test"

"$BUILD/error-check-retry-test" "$ROOT"
rm -rf "$BUILD"
