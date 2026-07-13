#!/usr/bin/env bash
set -euo pipefail

JOBS="${JOBS:-$(nproc)}"
make aevum-engine -j"$JOBS"
make prmers -j"$JOBS"
echo "Built ./prmers and ./third_party/aevum/build-engine/libaevum_engine.so"
