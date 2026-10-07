#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-pm1-p95-checksum"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD"
"${CXX:-g++}" -std=c++20 -O1 -I"$ROOT/include" -I"$ROOT/include/marin" \
  "$ROOT/tests/test_pm1_p95_checksum.cpp" -o "$BUILD/test_pm1_p95_checksum" -lgmpxx -lgmp
"$BUILD/test_pm1_p95_checksum"
