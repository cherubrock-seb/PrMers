#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OS="$(uname -s)"
if command -v nproc >/dev/null 2>&1; then JOBS="$(nproc)";
elif [[ "$OS" == Darwin ]]; then JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)";
else JOBS=4; fi
CXX_BIN="${CXX:-c++}"
make -j"$JOBS" all KERNEL_PATH=./kernels/ CXX="$CXX_BIN"
python3 tests/native_pfa_cli_source_test.py
python3 third_party/aevum/tools/native_pfa_reference_test.py
python3 third_party/aevum/tools/native_pfa_source_audit.py
echo "Built PrMers: $ROOT/prmers"
echo "Built Aevum: $ROOT/third_party/aevum/build-engine/libaevum_engine.so"
