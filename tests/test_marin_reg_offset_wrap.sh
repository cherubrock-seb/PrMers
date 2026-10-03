#!/usr/bin/env bash
# Opt-in: needs an OpenCL device that accepts a 32 GiB single allocation
# (for example PoCL on a 64 GiB host). Not part of any default test target.
#   OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd make test-marin-reg-offset-wrap
# Extra arguments are passed through: <device index> <register count>.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-marin-reg-offset-wrap"

rm -rf "$BUILD"
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -I"$ROOT/include" \
  -I"$ROOT/include/marin" \
  "$ROOT/tests/marin_reg_offset_wrap_test.cpp" \
  -o "$BUILD/marin-reg-offset-wrap-test" \
  -lOpenCL -lgmp

"$BUILD/marin-reg-offset-wrap-test" "$@"
