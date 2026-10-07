#!/usr/bin/env bash
# Build and run the legacy sizing helper host test (header-only, no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-legacy-small-items"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/legacy_small_items_test.cpp" \
  -o "$BUILD/legacy-small-items-test"

"$BUILD/legacy-small-items-test"
rm -rf "$BUILD"
