#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-proof-marin"

rm -rf "$BUILD"
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/proof_marin_padding_test.cpp" \
  "$ROOT/src/core/ProofMarin.cpp" \
  "$ROOT/src/io/sha3.cpp" \
  -o "$BUILD/proof-marin-padding-test"

"$BUILD/proof-marin-padding-test"
