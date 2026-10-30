#!/usr/bin/env bash
# A factor that shows up when the stage 1 result of a -cmont -notorsion curve is converted to
# affine form (the inverse of Z - Y fails) used to be recorded as already known before it was
# reported: the curve printed "known factor=", the run did not stop, and a result with that
# factor was published again by later curves and at the end.  It is a new factor like any other.
#   M239 = 479 * 1913 * 5737 * ...; with -b1 5, seed 1 exposes 479 this way on curve 1.
# Run on PoCL or any OpenCL device.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

FAILURES=0
fail() { echo "FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
newdir() { rm -rf "$WORK/$1"; mkdir -p "$WORK/$1"; ln -s "$ROOT/kernels" "$WORK/$1/kernels"; cd "$WORK/$1" || exit 1; }
run_fg() { # log args... ; sets RC
  local log=$1; shift
  timeout --signal=INT --kill-after=10s 120 "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1
  RC=$?
}
flat() { tr '\r' '\n' <"$1"; }

# 1. a single curve: the factor is new, the run stops with success and publishes one result
newdir single
run_fg s.log 239 -ecm -cmont -notorsion -b1 5 -b2 0 -seed 1
flat s.log | grep -aq '^\[ECM\] factor=479' || fail "single: seed 1 no longer exposes 479 while converting to affine form (test data out of date?)"
flat s.log | grep -aq 'Curve 1/1 | factor=479' || fail "single: the factor was not reported as new"
flat s.log | grep -aq 'known factor' && fail "single: the new factor was reported as known"
[ "$RC" -eq 0 ] || fail "single: exit code $RC, expected 0"
[ "$(grep -ac '"status":"F"' results.txt)" -eq 1 ] || fail "single: expected one factor result, got $(grep -ac '"status":"F"' results.txt)"
flat s.log | grep -aq 'No factor found' && fail "single: ended with the no-factor result"

# 2. a series: curve 4 meets the same factor again; that is not a result
newdir series
run_fg r.log 239 -ecm -cmont -notorsion -b1 5 -b2 0 -seed 1 -K 4 -ecm-continue-after-factor
flat r.log | grep -aq 'Curve 1/4 | factor=479' || fail "series: curve 1 did not report 479 as new"
flat r.log | grep -aq 'Curve 3/4 | factor=1913' || fail "series: curve 3 did not report 1913 as new (test data out of date?)"
flat r.log | grep -aq 'Curve 4/4: only a known factor, skipping curve' || fail "series: the known factor on curve 4 was not skipped"
# one result per new factor (479, 1913) plus the closing summary of the run
[ "$(grep -ac '"status":"F"' results.txt)" -eq 3 ] || fail "series: expected 3 results, got $(grep -ac '"status":"F"' results.txt)"

[ "$FAILURES" -eq 0 ] || exit 1
echo "ecm stage 1 affine-factor test passed"
