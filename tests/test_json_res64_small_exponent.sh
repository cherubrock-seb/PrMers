#!/usr/bin/env bash
# Host test: Res64 / Res2048 helpers for exponents of at most 32 bits (one 32-bit word).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-json-res64-small-exponent"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gmpxx gmp; then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"
fi

"${CXX:-c++}" \
  -std=c++20 \
  -O1 \
  -g \
  -fsanitize=address \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "${GMP_CFLAGS[@]}" \
  "$ROOT/tests/json_res64_small_exponent_test.cpp" \
  "$ROOT/src/io/JsonBuilder.cpp" \
  "$ROOT/src/io/md5.cpp" \
  "$ROOT/src/util/Crc32.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "${GMP_LIBS[@]}" \
  -o "$BUILD/json-res64-small-exponent-test"

"$BUILD/json-res64-small-exponent-test"
