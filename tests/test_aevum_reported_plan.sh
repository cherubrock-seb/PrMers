#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${AEVUM_REPORTED_PLAN_BUILD:-$ROOT/tests/build-aevum-reported-plan}"
mkdir -p "$BUILD"
CXX="${CXX:-c++}"
SOURCES=(
  "$ROOT/tests/test_aevum_reported_plan.cpp"
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
    -o "$BUILD/test_aevum_reported_plan"
else
  "$CXX" -std=c++20 -O2 -fPIC -shared \
    "$ROOT/tests/aevum_plan_report_engine.cpp" \
    -o "$BUILD/libaevum_engine_plan_report.so"
  "$CXX" -std=c++20 -O2 \
    -I"$ROOT/include" -I"$ROOT/include/marin" \
    "${SOURCES[@]}" \
    -DAEVUM_ENGINE_DEFAULT_LIB=\"/nonexistent/libaevum_engine.so\" \
    -pthread -ldl -lOpenCL -lgmpxx -lgmp \
    -o "$BUILD/test_aevum_reported_plan"
fi
AEVUM_ENGINE_LIB="$BUILD/libaevum_engine_plan_report.so" "$BUILD/test_aevum_reported_plan"
