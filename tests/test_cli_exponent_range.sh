#!/usr/bin/env bash
# Host test (no GPU): the command line must reject exponents above 2^32 - 1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-cli-exponent-range"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 -O1 -Wall -Wextra \
  -I"$ROOT/include" -I"$ROOT/include/marin" -DGPU \
  '-DKERNEL_PATH="./kernels/"' '-DAEVUM_ENGINE_DEFAULT_LIB="x"' '-DAEVUM_ENGINE_DEFAULT_TUNE_DIR="x"' \
  "$ROOT/tests/cli_exponent_range_test.cpp" \
  "$ROOT/src/io/CliParser.cpp" \
  "$ROOT/src/util/PathUtils.cpp" \
  "$ROOT/src/util/StringUtils.cpp" \
  "$ROOT/src/util/Fs.cpp" \
  -o "$BUILD/cli-exponent-range-test" -lgmpxx -lgmp -ldl

T="$BUILD/cli-exponent-range-test"
fail=0

expect_accept() {
  if ! out=$("$T" accepted "$@" 2>&1); then echo "FAIL: '$*' should be accepted: $out"; fail=1; else echo "ok   accepted: $*"; fi
}
expect_reject() {
  local needle=$1; shift
  if out=$("$T" rejected "$@" 2>&1); then echo "FAIL: '$*' should be rejected, got: $out"; fail=1
  elif [[ "$out" != *"$needle"* ]]; then echo "FAIL: '$*' rejected without '$needle': $out"; fail=1
  else echo "ok   rejected: $*"; fi
}

expect_accept 86243 -prp --noask
expect_accept 4294967295 -prp --noask
expect_reject "Exponent must be <= 4294967295" 4294967296 -prp --noask
expect_reject "Exponent must be <= 4294967295" 4294967357 -prp --noask
expect_reject "Exponent must be <= 4294967295" 5650242869 -prp --noask
expect_reject "Exponent must be <= 5650242869" 5650242870 -prp --noask
expect_reject "Exponent must be <=" 99999999999999999999999 -prp --noask
# Wagstaff doubles the exponent before the engines see it.
expect_accept 2147483647 -wagstaff --noask
expect_reject "twice the requested Wagstaff exponent" 2147483648 -wagstaff --noask

exit $fail
