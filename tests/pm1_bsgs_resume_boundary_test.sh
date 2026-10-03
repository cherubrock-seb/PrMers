#!/usr/bin/env bash
# Interrupt the classic BSGS stage 2 (D=6, so nearly every saved position sits
# on a giant-step boundary) and resume it.  M677 has the factor 1943118631
# = 2*677*45*31891 + 1, which stage 2 only reaches at the prime 31891 (B1=10
# covers 45), long after the interrupt.  The resumed run must still find it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$ROOT/kernels" "$WORK/kernels"
cd "$WORK"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off PRMERS_PM1_CLASSIC_D=6
ARGS=(677 -pm1 -b1 10 -b2 32500 -pm1-vtrace-off -d "$DEVICE" --noask)

# The first run is interrupted inside stage 2 and leaves pm1_s2_m_677.ckpt.
timeout --signal=INT --kill-after=10s 5 "$ROOT/prmers" "${ARGS[@]}" >first.log 2>&1 || true
grep -q 'Stage 2 state saved at prime' first.log || { echo "stage 2 was not interrupted" >&2; exit 1; }

timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" "${ARGS[@]}" >second.log 2>&1 || true
grep -q 'Resuming Stage 2 from checkpoint' second.log || { echo "stage 2 did not resume" >&2; exit 1; }
grep -q 'P-1 factor stage 2 found: 1943118631' second.log || { echo "resumed stage 2 missed the factor" >&2; exit 1; }
echo "pm1 bsgs resume boundary test passed"
