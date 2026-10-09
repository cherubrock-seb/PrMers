#!/usr/bin/env bash
# Interrupt the classic BSGS stage 2 (D=6, so nearly every saved position sits
# on a giant-step boundary) and resume it.  M677 has the factor 1943118631
# = 2*677*45*31891 + 1, which stage 2 only reaches at the prime 31891 (B1=10
# covers 45), long after the interrupt.  The resumed run must still find it.
#
# The interrupt is sent as soon as the stage-2 loop has started, not after a
# fixed time, so it lands at the same point of the run on a slow or a fast
# device.  The saved position must also be one the resume logic can get wrong
# (see below) and lie before the factor's prime; if an attempt lands elsewhere
# it is repeated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off PRMERS_PM1_CLASSIC_D=6
ARGS=(677 -pm1 -b1 10 -b2 32500 -pm1-vtrace-off -d "$DEVICE" --noask)
FACTOR_PRIME=31891
fail() { echo "$1" >&2; exit 1; }

# The checkpoint is written after the loop has advanced to the next prime, while
# the giant step still belongs to the prime processed last.  With D=6 these lie
# in different blocks exactly when the saved prime is 1 mod 6 (the previous
# prime is then 5 mod 6 or lower), which is the case the resume must handle.
ATTEMPTS=20
for attempt in $(seq 1 "$ATTEMPTS"); do
  dir="$WORK/try$attempt"
  mkdir "$dir"
  ln -s "$ROOT/kernels" "$dir/kernels"
  # Start in the background and send SIGINT as soon as the stage-2 loop starts.
  ( cd "$dir" && exec timeout --signal=INT --kill-after=10s 300 "$ROOT/prmers" "${ARGS[@]}" >first.log 2>&1 ) &
  pid=$!
  until grep -aq 'PM1-CLASSIC' "$dir/first.log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.005
  done
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" || true
  saved="$(sed -n 's/.*Stage 2 state saved at prime \([0-9][0-9]*\).*/\1/p' "$dir/first.log" | tail -n 1)"
  idx="$(sed -n 's/.*Stage 2 state saved at prime [0-9]* idx=\([0-9][0-9]*\).*/\1/p' "$dir/first.log" | tail -n 1)"
  if [ -z "$saved" ] || [ ! -e "$dir/pm1_s2_m_677.ckpt" ]; then
    echo "attempt $attempt: stage 2 finished or stopped before the interrupt landed, retrying" >&2
    continue
  fi
  if [ "$saved" -gt "$FACTOR_PRIME" ]; then
    echo "attempt $attempt: interrupt landed at prime $saved, after the factor's prime, retrying" >&2
    continue
  fi
  if [ "${idx:-0}" -eq 0 ] || [ $((saved % 6)) -ne 1 ]; then
    echo "attempt $attempt: saved prime $saved (idx $idx) is not the first prime of a new block, retrying" >&2
    continue
  fi
  echo "interrupted stage 2 at prime $saved"
  break
done
[ -n "${saved:-}" ] && [ -e "$dir/pm1_s2_m_677.ckpt" ] && [ "$saved" -le "$FACTOR_PRIME" ] \
  && [ "${idx:-0}" -gt 0 ] && [ $((saved % 6)) -eq 1 ] \
  || fail "stage 2 was not interrupted at a usable position in $ATTEMPTS attempts"

cd "$dir"
timeout --signal=INT --kill-after=10s 300 "$ROOT/prmers" "${ARGS[@]}" >second.log 2>&1 || true
grep -q 'Resuming Stage 2 from checkpoint' second.log || { echo "stage 2 did not resume" >&2; exit 1; }
grep -q 'P-1 factor stage 2 found: 1943118631' second.log || { echo "resumed stage 2 missed the factor" >&2; exit 1; }
echo "pm1 bsgs resume boundary test passed"
