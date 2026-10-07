#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-compact-bits-wrap"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/compact_bits_wrap_test.cpp" \
  "$ROOT/src/io/JsonBuilder.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  "$ROOT/src/util/Crc32.cpp" \
  "$ROOT/src/io/md5.cpp" \
  -o "$BUILD/compact-bits-wrap-test" \
  -lgmpxx -lgmp -lpthread

"$BUILD/compact-bits-wrap-test"
