#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-aevum-reg"
mkdir -p "$BUILD"

CXX="${CXX:-c++}"
SHARED_FLAGS=(-shared)
DL_LIBS=(-ldl)
OPENCL_LIBS=(-lOpenCL)
if [[ "$(uname -s)" == "Darwin" ]]; then
    SHARED_FLAGS=(-dynamiclib)
    DL_LIBS=()
    OPENCL_LIBS=(-framework OpenCL)
fi
"$CXX" -std=c++20 -O2 -fPIC "${SHARED_FLAGS[@]}" \
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
    -pthread "${DL_LIBS[@]}" "${OPENCL_LIBS[@]}" -lgmpxx -lgmp \
    -o "$BUILD/test_aevum_reg_adapter"

AEVUM_ENGINE_LIB="$BUILD/libaevum_engine_fake.so" "$BUILD/test_aevum_reg_adapter"
