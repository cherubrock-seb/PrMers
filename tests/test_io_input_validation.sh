#!/usr/bin/env bash
# Host test: -filemers file name and interactive exponent validation (header-only helpers).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-io-input-validation"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/io_input_validation_test.cpp" \
  -o "$BUILD/io-input-validation-test"

"$BUILD/io-input-validation-test"
