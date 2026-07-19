#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OS="$(uname -s)"
if command -v nproc >/dev/null 2>&1; then JOBS="$(nproc)";
elif [[ "$OS" == Darwin ]]; then JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)";
else JOBS=4; fi
CXX_BIN="${CXX:-c++}"
make -j"$JOBS" engine-lib CXX="$CXX_BIN"
mkdir -p build-tests
if [[ "$OS" == Darwin ]]; then
  "$CXX_BIN" -O2 -std=c++20 -Wall -Wextra \
    tests/native_pfa_engine_compare.cpp \
    -o build-tests/native-pfa-engine-compare
else
  "$CXX_BIN" -O2 -std=c++20 -Wall -Wextra \
    tests/native_pfa_engine_compare.cpp -ldl \
    -o build-tests/native-pfa-engine-compare
fi
python3 tools/native_pfa_reference_test.py
python3 tools/native_pfa_source_audit.py
python3 tools/native_pfa_plan_test.py build-engine/libaevum_engine.so
printf '\nBuilt: %s\n' "$ROOT/build-engine/libaevum_engine.so"
