#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-ibdwt-size-bound"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_ibdwt_size_bound_test.cpp" \
  -o "$BUILD/marin-ibdwt-size-bound-test"

"$BUILD/marin-ibdwt-size-bound-test"
