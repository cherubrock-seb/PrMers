#!/usr/bin/env bash
# Host test: the GUI access token is redacted in prmers.log (header-only helper).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-log-redact"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/log_redact_test.cpp" \
  -o "$BUILD/log-redact-test"

"$BUILD/log-redact-test"
