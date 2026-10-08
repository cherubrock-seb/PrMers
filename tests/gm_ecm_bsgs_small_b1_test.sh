#!/usr/bin/env bash
# BSGS Stage 2 with a B1 below the default D/2 must still cover the primes above
# B1. M(17) has the factor 137 whose Stage 2 prime is 3 (B1=2, B2=3, sigma=14):
# the legacy Stage 2 finds it, and BSGS used to skip the prime and report no
# factor. B1=1 cannot be covered by any D and must be rejected.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

run() {
  local name="$1"; shift
  mkdir -p "$WORK/$name"
  ( cd "$WORK/$name" && ln -s "$ROOT/kernels" kernels &&
    timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 17 -gm-ecm "$@" \
      -K 1 -sigma 14 -gm-sieve 0 -d "$DEVICE" --noask >run.log 2>&1 || true )
}

run legacy -b1 2 -b2 3
grep -q "Stage 2 factor: 137" "$WORK/legacy/run.log" || { echo "legacy: factor 137 not found" >&2; exit 1; }

run bsgs -bsgs -b1 2 -b2 3
grep -q "Stage 2 factor: 137" "$WORK/bsgs/run.log" || { echo "bsgs: factor 137 not found in Stage 2" >&2; cat "$WORK/bsgs/run.log" >&2; exit 1; }

run reject -bsgs -b1 1 -b2 3
grep -q "requires B1 >= 2" "$WORK/reject/run.log" || { echo "reject: B1=1 was not rejected" >&2; exit 1; }
echo "gm ecm bsgs small-B1 test passed"
