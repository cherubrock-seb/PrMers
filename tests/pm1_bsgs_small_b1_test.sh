#!/usr/bin/env bash
# Classic (non V-trace) stage-2 BSGS with B1 below the largest prime factor of D
# (D = 630 = 2*3^2*5*7).  The primes 5 and 7 lie in (B1, B2] and divide D, so they
# need their own baby steps; the run used to abort with "residue not found".
#   M269: 13822297 = 2*269*(2^2*3*2141) + 1, found by stage 2 with B1=4, B2=2141.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

( cd "$WORK" && ln -s "$ROOT/kernels" kernels &&
  timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 269 -pm1 -b1 4 -b2 2141 -pm1-vtrace-off \
    -d "$DEVICE" --noask >run.log 2>&1 || true )
if grep -q 'INTERNAL ERROR' "$WORK/run.log"; then
  echo "classic BSGS reported an internal error" >&2; exit 1
fi
grep -q 'P-1 factor stage 2 found: 13822297' "$WORK/run.log" || { echo "factor 13822297 not found" >&2; exit 1; }
echo "pm1 classic BSGS small-B1 test passed"
