#!/usr/bin/env bash
# Build and run the Marin engine destructor lost-device test (needs an OpenCL device; PoCL's CPU device is enough).
# usage: tests/test_marin_destructor_lost_device.sh [device-index] [p]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-destructor-lost-device"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -DGPU \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_destructor_lost_device_test.cpp" \
  -o "$BUILD/marin-destructor-lost-device-test" \
  -lOpenCL -lgmp -ldl

"$BUILD/marin-destructor-lost-device-test" "$@"
