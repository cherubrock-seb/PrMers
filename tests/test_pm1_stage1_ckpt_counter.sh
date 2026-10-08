#!/usr/bin/env bash
# Build and run the P-1 stage-1 checkpoint counter host test (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-pm1-stage1-ckpt-counter"

mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/pm1_stage1_ckpt_counter_test.cpp" \
  -o "$BUILD/pm1-stage1-ckpt-counter-test"

"$BUILD/pm1-stage1-ckpt-counter-test" "$BUILD"
