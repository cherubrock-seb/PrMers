#!/usr/bin/env bash
# Host test: include/marin/arith.h helpers and their range checks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-arith-host"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_arith_host_test.cpp" \
  -o "$BUILD/marin-arith-host-test"

"$BUILD/marin-arith-host-test"
