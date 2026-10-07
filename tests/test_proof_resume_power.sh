#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-proof-resume-power"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/proof_resume_power_test.cpp" \
  "$ROOT/src/core/ProofSetMarin.cpp" \
  "$ROOT/src/core/ProofMarin.cpp" \
  "$ROOT/src/io/sha3.cpp" \
  "$ROOT/src/util/Crc32.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  "$ROOT/src/util/Timer.cpp" \
  -o "$BUILD/proof-resume-power-test" \
  -lgmpxx -lgmp -lpthread

"$BUILD/proof-resume-power-test"
