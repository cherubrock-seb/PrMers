#!/usr/bin/env bash
# Build and run the CPU proof verification host test (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-proof-verify-cpu"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/proof_verify_cpu_test.cpp" \
  "$ROOT/src/core/ProofVerifyCpu.cpp" \
  "$ROOT/src/core/ProofSetMarin.cpp" \
  "$ROOT/src/core/ProofMarin.cpp" \
  "$ROOT/src/io/sha3.cpp" \
  "$ROOT/src/util/Crc32.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  "$ROOT/src/util/Timer.cpp" \
  -o "$BUILD/proof-verify-cpu-test" \
  -lgmpxx -lgmp -lpthread

"$BUILD/proof-verify-cpu-test" > "$BUILD/out.log" 2>&1 || { grep -v '^proof \[' "$BUILD/out.log"; exit 1; }
tail -n 1 "$BUILD/out.log"
