#!/usr/bin/env bash
# Build and run the legacy checkpoint-set host test (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-legacy-checkpoint-set"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/legacy_checkpoint_set_test.cpp" \
  "$ROOT/src/core/BackupManager.cpp" \
  -o "$BUILD/legacy-checkpoint-set-test" \
  -lOpenCL -lgmpxx -lgmp

"$BUILD/legacy-checkpoint-set-test" > "$BUILD/out.log" 2>&1 || { cat "$BUILD/out.log"; exit 1; }
tail -n 1 "$BUILD/out.log"
