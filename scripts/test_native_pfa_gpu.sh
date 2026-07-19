#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
ITERS="${2:-1}"
"$ROOT/scripts/build_native_pfa_ubuntu.sh"
"$ROOT/third_party/aevum/scripts/test_native_pfa_gpu.sh" "$DEVICE" "$ITERS"
echo
"$ROOT/prmers" --help 2>&1 | grep -E -- '-pfa \[3\|9\]|-pfa3 / -pfa9'
echo 'PrMers CLI and native Aevum PFA differential tests passed'
