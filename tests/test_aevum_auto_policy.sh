#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-aevum-auto"
mkdir -p "$BUILD"
CXX="${CXX:-c++}"
COMMON=(
  -std=c++20 -O2
  -I"$ROOT/include" -I"$ROOT/include/marin"
  "$ROOT/tests/test_aevum_auto_policy.cpp"
  "$ROOT/src/aevum/AutoPolicy.cpp"
  "$ROOT/src/aevum/EngineAevum.cpp"
)
if [[ "$(uname -s)" == Darwin ]]; then
  "$CXX" "${COMMON[@]}" -lgmpxx -lgmp -o "$BUILD/test_aevum_auto_policy"
else
  "$CXX" "${COMMON[@]}" -ldl -lgmpxx -lgmp -o "$BUILD/test_aevum_auto_policy"
fi
AEVUM_ENGINE_LIB="$ROOT/third_party/aevum/build-engine/libaevum_engine.so" \
  "$BUILD/test_aevum_auto_policy"
