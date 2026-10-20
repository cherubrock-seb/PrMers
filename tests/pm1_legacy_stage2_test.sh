#!/usr/bin/env bash
# The legacy (-marin) P-1 stage 2 must run to the end.  NttEngine::mulInPlace(A, ...) copies A into the input
# buffer and back, stage 2 calls it with A = input, and clEnqueueCopyBuffer rejects a buffer copied onto itself
# (CL_MEM_COPY_OVERLAP); since copy() throws on an error every legacy stage 2 stopped with
# "clEnqueueCopyBuffer failed with error -8".  M269 has the P-1 factor 13822297, found in stage 2 from B1=4.
# usage: pm1_legacy_stage2_test.sh <device>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$ROOT/kernels" "$WORK/kernels"
( cd "$WORK" && timeout -k 10 300 "$ROOT/prmers" 269 -pm1 -b1 4 -b2 2141 -pm1-vtrace-off -marin -d "$DEVICE" --noask \
    > run.log 2>&1 || true )
if ! grep -aq 'Factor P-1 (stage 2) found : 13822297' "$WORK/run.log"; then
    echo "FAIL: legacy P-1 stage 2 of M269 did not find 13822297"; tail -n 5 "$WORK/run.log"; exit 1
fi
if ! grep -aq '"factors":\["13822297"\]' "$WORK/results.txt"; then
    echo "FAIL: no factor result written"; exit 1
fi
echo "Legacy P-1 stage 2: PASS"
