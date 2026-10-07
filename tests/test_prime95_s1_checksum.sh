#!/usr/bin/env bash
# Build and run the .p95 stage-1 checksum host test (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-prime95-s1-checksum"

mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  -DGPU \
  "$ROOT/tests/prime95_s1_checksum_test.cpp" \
  -o "$BUILD/prime95-s1-checksum-test" \
  -lOpenCL -lgmpxx -lgmp

"$BUILD/prime95-s1-checksum-test" "$BUILD"
