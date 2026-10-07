#!/usr/bin/env bash
# V-trace stage 2 with D > 2*B1: the primes q <= D/2 have k = 0 and no lower pair
# member.  Their "lower member" k*D-j used to wrap around to 2^64-q, which the
# trial-division primality test then called prime about 1 time in 8, so the upper
# member (q itself) was skipped as a duplicate and its term was never accumulated.
#   M97: 11447 = 2*59*97 + 1, found by stage 2 with B1=4, B2=100 (needs q = 59).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$ROOT/kernels" "$WORK/kernels"
cd "$WORK"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

run_plan() {
  local name="$1"; shift
  timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 97 -pm1 -b1 4 -b2 100 -pm1-vtrace-d 210 \
    "$@" -d "$DEVICE" --noask >"$name.log" 2>&1 || true
  if grep -q 'INTERNAL ERROR' "$name.log"; then
    echo "$name: internal error" >&2; exit 1
  fi
  grep -q 'P-1 factor stage 2 found: 11447' "$name.log" || { echo "$name: factor 11447 not found" >&2; exit 1; }
  if ! grep -q 'paired upper skips=0' "$name.log"; then
    echo "$name: an upper pair member was skipped although no lower member exists" >&2; exit 1
  fi
}

run_plan linear -pm1-vtrace-pair95-off
run_plan batched -pm1-vtrace-pair95-off -pm1-vtrace-baby-batch 8
run_plan product_tree -pm1-vtrace-pair95-off -pm1-vtrace-product-tree
echo "pm1 V-trace low-prime pair test passed"
