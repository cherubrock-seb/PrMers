#!/usr/bin/env bash
# A P-1 stage-1 checkpoint written for one B1 must not be resumed by a run with
# another B1.  M269 has the P-1 factor 13822297 (k = 25692 is 2141-smooth), so a
# correct B1=2141 run must report it even when a checkpoint from B1=2000000 is
# lying around.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$ROOT/kernels" "$WORK/kernels"
cd "$WORK"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# 1. Start a large-B1 run and interrupt it so that it leaves a checkpoint.
timeout --signal=INT --kill-after=10s 3 "$ROOT/prmers" 269 -pm1 -b1 2000000 -d "$DEVICE" --noask >first.log 2>&1 || true
[ -f pm1_m_269.ckpt ] || { echo "no stage-1 checkpoint was written" >&2; exit 1; }

# 2. Run a small B1 in the same directory.
timeout --signal=INT --kill-after=10s 40 "$ROOT/prmers" 269 -pm1 -b1 2141 -d "$DEVICE" --noask >second.log 2>&1 || true
if ! grep -q 'Ignoring checkpoint' second.log; then
  echo "checkpoint from another B1 was not rejected" >&2; exit 1
fi
if ! grep -q 'bits=3108' second.log; then
  echo "run did not rebuild the exponent for B1=2141" >&2; exit 1
fi
grep -q 'P-1 factor stage 1 found: 13822297' second.log || { echo "factor 13822297 not found" >&2; exit 1; }
echo "pm1 stage-1 checkpoint B1 test passed"
