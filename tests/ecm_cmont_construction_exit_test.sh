#!/usr/bin/env bash
# -cmont curve construction can end early: the gcd or inverse needed to build the curve exposes a
# factor of N, or the curve is degenerate modulo N.  Such a curve must be handled like the others:
#   1. a new factor is reported like a stage-1 factor (the result does not claim a stage 2 that
#      never ran) and, with -ecm-continue-after-factor, the later curves still run;
#   2. a degenerate curve is skipped without publishing a result: one "no factor" result for the
#      run, not one per skipped curve plus a final one.
# M239 = 479 * 1913 * 5737 * 176383 * ...; -seed 13 builds a curve whose construction exposes 479.
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

newdir() {
  rm -rf "$WORK/$1"; mkdir -p "$WORK/$1"; ln -s "$ROOT/kernels" "$WORK/$1/kernels"; cd "$WORK/$1" || exit 1
}
run_fg() { # log args... : run to completion (bounded); sets RC
  local log=$1; shift
  timeout --signal=INT --kill-after=10s 120 "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1
  RC=$?
}
results() { grep -ac '^{"status"' results.txt 2>/dev/null || true; }

# ---- 1. a factor exposed while the curve is built (stage 2 requested, never reached)
newdir factor
run_fg f.log 239 -ecm -cmont -notorsion -seed 13 -b1 100 -b2 1000
[ "$RC" -eq 0 ] || fail "factor: exit code $RC, expected 0"
grep -aq '(curve construction)' f.log || fail "factor: seed 13 no longer exposes a factor while building the curve (test data out of date?)"
grep -aq '"factors":\["479"\]' results.txt || fail "factor: result does not carry factor 479"
grep -aq '"b2"' results.txt && fail "factor: the result claims a stage 2 (b2) that never ran"
[ "$(results)" -eq 1 ] || fail "factor: expected exactly one result, got $(results)"

# ---- 2. the same curve inside a series: later curves still run
newdir series
run_fg s.log 239 -ecm -cmont -notorsion -seed 13 -K 3 -ecm-continue-after-factor -b1 100 -b2 0
grep -aq '(curve construction)' s.log || fail "series: curve 1 did not expose a factor while being built"
grep -aq 'Continuing with later curves by -ecm-continue-after-factor' s.log || fail "series: the run did not announce that it continues"
grep -aq 'Curve 3/3' s.log || fail "series: the later curves did not run"

# ---- 3. a degenerate curve (sigma 0) is skipped without publishing its own result
newdir degenerate
run_fg d.log 239 -ecm -cmont -sigma 0 -b1 100 -b2 0
grep -aq 'degenerate curve, skipping curve' d.log || fail "degenerate: the curve was not reported as skipped"
[ "$(results)" -eq 1 ] || fail "degenerate: expected exactly one result for the run, got $(results)"
[ "$(grep -ac 'json for manual submit' d.log)" -eq 1 ] || fail "degenerate: result published more than once"
grep -aq 'No factor found' d.log || fail "degenerate: the run did not end with the usual no-factor result"

[ "$FAILURES" -eq 0 ] || exit 1
echo "ecm cmont construction-exit test passed"
