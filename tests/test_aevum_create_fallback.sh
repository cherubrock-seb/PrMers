#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${AEVUM_CREATE_FALLBACK_BUILD:-$ROOT/tests/build-aevum-create-fallback}"
mkdir -p "$BUILD"
CXX="${CXX:-c++}"
SOURCES=(
  "$ROOT/tests/test_aevum_create_fallback.cpp"
  "$ROOT/src/aevum/EngineAevum.cpp"
  "$ROOT/src/aevum/AutoPolicy.cpp"
  "$ROOT/src/marin/gpu.cpp"
  "$ROOT/src/ui/WebGuiServer.cpp"
)
if [[ "$(uname -s)" == Darwin ]]; then
  "$CXX" -std=c++20 -O2 -fPIC -dynamiclib \
    "$ROOT/tests/aevum_plan_report_engine.cpp" \
    -o "$BUILD/libaevum_engine_plan_report.so"
  "$CXX" -std=c++20 -O2 \
    -I"$ROOT/include" -I"$ROOT/include/marin" \
    "${SOURCES[@]}" \
    -DAEVUM_ENGINE_DEFAULT_LIB=\"/nonexistent/libaevum_engine.so\" \
    -pthread -framework OpenCL -lgmpxx -lgmp \
    -o "$BUILD/test_aevum_create_fallback"
else
  "$CXX" -std=c++20 -O2 -fPIC -shared \
    "$ROOT/tests/aevum_plan_report_engine.cpp" \
    -o "$BUILD/libaevum_engine_plan_report.so"
  "$CXX" -std=c++20 -O2 \
    -I"$ROOT/include" -I"$ROOT/include/marin" \
    "${SOURCES[@]}" \
    -DAEVUM_ENGINE_DEFAULT_LIB=\"/nonexistent/libaevum_engine.so\" \
    -pthread -ldl -lOpenCL -lgmpxx -lgmp \
    -o "$BUILD/test_aevum_create_fallback"
fi
# The AUTO fallback builds a Marin engine on device 0.  Hide every OpenCL driver so that it fails fast and the
# result does not depend on, or run anything on, the machine's devices.
mkdir -p "$BUILD/no-icd"
OCL_ICD_VENDORS="$BUILD/no-icd" OCL_ICD_FILENAMES="" \
  AEVUM_ENGINE_LIB="$BUILD/libaevum_engine_plan_report.so" "$BUILD/test_aevum_create_fallback"
