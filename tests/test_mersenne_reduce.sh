#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-mersenne-reduce"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/mersenne_reduce_test.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  -o "$BUILD/mersenne-reduce-test" \
  -lgmpxx -lgmp -lpthread

"$BUILD/mersenne-reduce-test"
