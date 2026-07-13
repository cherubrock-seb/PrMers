#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${JOBS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
    else
        JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
fi
make aevum-engine -j"$JOBS"
make prmers -j"$JOBS"
echo "Built ./prmers and ./third_party/aevum/build-engine/libaevum_engine.so"
