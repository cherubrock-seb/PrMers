#!/usr/bin/env bash
# Build and run the legacy transform-size / checkpoint-size host test (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-legacy-transform-size"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/legacy_transform_size_test.cpp" \
  "$ROOT/src/math/Precompute.cpp" \
  "$ROOT/src/math/Mod64.cpp" \
  "$ROOT/src/core/BackupManager.cpp" \
  -o "$BUILD/legacy-transform-size-test" \
  -lOpenCL -lgmpxx -lgmp

"$BUILD/legacy-transform-size-test"
