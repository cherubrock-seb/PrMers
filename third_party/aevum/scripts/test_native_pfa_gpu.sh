#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
ITERS="${2:-1}"
"$ROOT/scripts/build_native_pfa_ubuntu.sh"
BIN="$ROOT/build-tests/native-pfa-engine-compare"
LIB="$ROOT/build-engine/libaevum_engine.so"
if ! nm -D "$LIB" 2>/dev/null | grep -q "aevum_engine_resolve_fft"; then
  echo "ERROR: libaevum_engine.so does not export aevum_engine_resolve_fft" >&2
  exit 1
fi
echo '=== native Aevum PFA radix-3 exact comparison ==='
"$BIN" "$LIB" "$DEVICE" 100000019 pfa:3 "$ITERS"
echo '=== native Aevum PFA radix-9 exact comparison ==='
"$BIN" "$LIB" "$DEVICE" 175000001 pfa:9 "$ITERS"
echo 'ALL NATIVE AEVUM PFA GPU TESTS PASSED'
