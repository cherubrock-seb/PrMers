#!/usr/bin/env bash
# Build and run the worktodo read-error test (Linux: it fails reads through /proc/self/fd; no OpenCL device).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-worktodo-read-error"

if [ ! -d /proc/self/fd ]; then echo "SKIP: worktodo read error test needs /proc/self/fd"; exit 0; fi

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  $(pkg-config --cflags gmpxx gmp 2>/dev/null || true) \
  "$ROOT/tests/worktodo_read_error_test.cpp" \
  "$ROOT/src/io/WorktodoParser.cpp" \
  "$ROOT/src/util/StringUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/math/Pm1Bounds.cpp" \
  "$ROOT/src/core/Logger.cpp" \
  $(pkg-config --libs gmpxx gmp 2>/dev/null || echo "-lgmpxx -lgmp") \
  -ldl \
  -o "$BUILD/worktodo-read-error-test"

"$BUILD/worktodo-read-error-test" "$BUILD/scratch"
