#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${1:-0}"
ITERS="${2:-1}"
"$ROOT/scripts/build_native_pfa_ubuntu.sh"
bash "$ROOT/third_party/aevum/tests/native_pfa_opencl_syntax.sh"
exec "$ROOT/scripts/test_native_pfa_gpu.sh" "$DEVICE" "$ITERS"
