#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-aevum-default"
mkdir -p "$BUILD"
"${CXX:-g++}" -std=c++20 -O2 -I"$ROOT/include" \
  "$ROOT/tests/test_aevum_default_backend.cpp" \
  -o "$BUILD/test_aevum_default_backend"
"$BUILD/test_aevum_default_backend"
