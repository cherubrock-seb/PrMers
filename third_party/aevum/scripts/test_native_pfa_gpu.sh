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
# Dense random residues near each plan's operating point (~34-38 bits/word).
# set_u32(3) has a constant transform spectrum and cannot see a twiddle that
# both transform directions omit; dense input can.
DENSE="$ROOT/build-tests/native-pfa-dense-compare"
DENSE_ITERS="${AEVUM_PFA_DENSE_ITERS:-2}"
echo '=== native Aevum PFA radix-3 dense random comparison (256:3:256, 34.6 bpw) ==='
"$DENSE" "$LIB" "$DEVICE" 13600003 1:256:4:256:202 pfa3:1:256:3:256:202 "$DENSE_ITERS"
echo '=== native Aevum PFA radix-9 dense random comparison (256:9:256, 34.8 bpw) ==='
"$DENSE" "$LIB" "$DEVICE" 41000017 1:256:16:256:202 pfa9:1:256:9:256:202 "$DENSE_ITERS"
echo '=== native Aevum PFA radix-9 dense random comparison (512:9:512, 34.8 bpw) ==='
"$DENSE" "$LIB" "$DEVICE" 164000047 1:512:16:512:202 pfa9:1:512:9:512:202 "$DENSE_ITERS"
echo 'ALL NATIVE AEVUM PFA GPU TESTS PASSED'
