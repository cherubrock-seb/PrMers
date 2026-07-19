#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${1:-0}"
ITERS="${2:-1}"
python3 "$ROOT/tools/native_pfa_reference_test.py"
python3 "$ROOT/tools/native_pfa_source_audit.py"
bash "$ROOT/tests/native_pfa_opencl_syntax.sh"
exec "$ROOT/scripts/test_native_pfa_gpu.sh" "$DEVICE" "$ITERS"
