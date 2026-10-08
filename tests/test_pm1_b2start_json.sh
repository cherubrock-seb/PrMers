#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-pm1-b2start-json"

trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O1 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/pm1_b2start_json_test.cpp" \
  "$ROOT/src/io/JsonBuilder.cpp" \
  "$ROOT/src/io/md5.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/util/Crc32.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  -lgmpxx -lgmp \
  -o "$BUILD/pm1-b2start-json-test"

"$BUILD/pm1-b2start-json-test"
