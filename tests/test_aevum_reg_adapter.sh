#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-aevum-reg"
mkdir -p "$BUILD"

CXX="${CXX:-g++}"
"$CXX" -std=c++20 -O2 -fPIC -shared \
    "$ROOT/tests/aevum_fake_engine.cpp" \
    -o "$BUILD/libaevum_engine_fake.so"

"$CXX" -std=c++20 -O2 \
    -I"$ROOT/include" -I"$ROOT/include/marin" \
    "$ROOT/tests/test_aevum_reg_adapter.cpp" \
    "$ROOT/src/aevum/EngineAevum.cpp" \
    "$ROOT/src/aevum/AutoPolicy.cpp" \
    "$ROOT/src/marin/gpu.cpp" \
    "$ROOT/src/ui/WebGuiServer.cpp" \
    -DAEVUM_ENGINE_DEFAULT_LIB=\"/nonexistent/libaevum_engine.so\" \
    -pthread -ldl -lOpenCL -lgmpxx -lgmp \
    -o "$BUILD/test_aevum_reg_adapter"

AEVUM_ENGINE_LIB="$BUILD/libaevum_engine_fake.so" "$BUILD/test_aevum_reg_adapter"
