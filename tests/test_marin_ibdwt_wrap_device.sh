#!/usr/bin/env bash
# Build and run the Marin IBDWT wrap device test (needs an OpenCL device, libgmp).
# usage: tests/test_marin_ibdwt_wrap_device.sh [device-index] [q ...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-ibdwt-wrap"

mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DGPU \
  -I"${MARIN_INCLUDE:-$ROOT/include}" \
  -I"${MARIN_INCLUDE:-$ROOT/include}/marin" \
  "$ROOT/tests/marin_ibdwt_wrap_device_test.cpp" \
  -o "$BUILD/marin-ibdwt-wrap-device-test" \
  -lOpenCL -lgmp

"$BUILD/marin-ibdwt-wrap-device-test" "$@"
