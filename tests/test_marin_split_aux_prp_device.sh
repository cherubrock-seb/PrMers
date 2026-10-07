#!/usr/bin/env bash
# Build and run the forced split-aux Marin PRP device test (needs an OpenCL device, libgmp).
# usage: tests/test_marin_split_aux_prp_device.sh [device-index] [q ...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-split-aux-prp"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DGPU \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_split_aux_prp_device_test.cpp" \
  -o "$BUILD/marin-split-aux-prp-device-test" \
  -lOpenCL -lgmp

"$BUILD/marin-split-aux-prp-device-test" "$@"
