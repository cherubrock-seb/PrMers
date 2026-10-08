#!/usr/bin/env bash
# End-to-end checks of the ECM resume and known-factor handling:
#   1. Twisted Edwards: a leftover checkpoint made with another -seed must not
#      replace the requested -seed; the same -seed still resumes it.
#   2. -cmont -seed -K n -ecm-continue-after-factor resumes at the interrupted
#      curve instead of curve 1, and ignores a checkpoint of another seed series.
#   3. -ecm-continue-after-factor reports only the part of a gcd that the factors
#      found earlier in the run do not explain (never p1*p2 after p1).
#   4. Garbage and truncated checkpoint files do not crash a run.
# Small exponents only (p = 239); run on PoCL or any OpenCL device.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
PID=""
cleanup() { [ -n "$PID" ] && kill -KILL "$PID" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

FAILURES=0
fail() { # set ECM_TEST_KEEP_GOING=1 to see every failing check instead of stopping at the first
  echo "FAIL: $*" >&2; FAILURES=$((FAILURES + 1))
  [ -n "${ECM_TEST_KEEP_GOING:-}" ] || exit 1
}

newdir() { # fresh working directory with the kernels linked in
  rm -rf "$WORK/$1"; mkdir -p "$WORK/$1"; ln -s "$ROOT/kernels" "$WORK/$1/kernels"; cd "$WORK/$1" || exit 1
}
start_bg() { # log args... : start prmers in the background
  local log=$1; shift
  "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1 &
  PID=$!
}
wait_for() { # log pattern seconds : poll until the pattern shows up
  local i
  for ((i = 0; i < $3 * 10; i++)); do
    grep -aq -- "$2" "$1" 2>/dev/null && return 0
    kill -0 "$PID" 2>/dev/null || { grep -aq -- "$2" "$1" 2>/dev/null; return; }
    sleep 0.1
  done
  return 1
}
stop_bg() { # SIGINT, then wait for the run to save its checkpoint and exit
  kill -INT "$PID" 2>/dev/null
  local i
  for ((i = 0; i < 300; i++)); do kill -0 "$PID" 2>/dev/null || break; sleep 0.1; done
  kill -KILL "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  PID=""
}
run_fg() { # log args... : run to completion (bounded)
  local log=$1; shift
  timeout --signal=INT --kill-after=10s 120 "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1
  return 0
}
flat() { tr '\r' '\n' <"$1"; }

# ---------------------------------------------------------------- 1. TE -seed
newdir te
TE=(239 -ecm -b1 6000 -b2 0)
start_bg r1.log "${TE[@]}" -seed 111
wait_for r1.log 'Stage1 [0-9]*/[0-9]' 90 || fail "TE: run 1 never reported Stage 1 progress"
stop_bg
ls ecm_te_m_239_c0.ckpt* >/dev/null 2>&1 || fail "TE: interrupted run left no Stage 1 checkpoint"

# same seed resumes
start_bg r2.log "${TE[@]}" -seed 111
wait_for r2.log 'Stage1 [0-9]*/[0-9]' 90 || fail "TE: resumed run never reported Stage 1 progress"
stop_bg
grep -aq 'curve_seed=111' r2.log || fail "TE: same -seed did not keep curve_seed=111"
grep -aq 'resumed from Stage1 checkpoint' r2.log || fail "TE: same -seed no longer resumes its checkpoint"

# another seed does not pick the leftover checkpoint's seed
start_bg r3.log "${TE[@]}" -seed 222
wait_for r3.log 'Stage1 [0-9]*/[0-9]' 90 || fail "TE: -seed 222 run never reported Stage 1 progress"
stop_bg
grep -aq 'curve_seed=222' r3.log || fail "TE: -seed 222 did not run curve_seed=222"
grep -aq 'curve_seed=111' r3.log && fail "TE: -seed 222 was replaced by the leftover seed 111"
grep -aq 'Ignoring checkpoint of curve 1 made with seed 111' r3.log || fail "TE: stale checkpoint not reported"

# a forced seed series (-K 3) accepts only the seeds of its own series
newdir te_series
start_bg s1.log "${TE[@]}" -seed 5 -K 3 -ecm-continue-after-factor
wait_for s1.log 'Stage1 [0-9]*/[0-9]' 90 || fail "TE series: no Stage 1 progress"
stop_bg
start_bg s2.log "${TE[@]}" -seed 6 -K 3 -ecm-continue-after-factor
wait_for s2.log 'curve_seed=' 90 || fail "TE series: second seed never started"
sleep 1; stop_bg
grep -aq 'curve_seed=5' s2.log && fail "TE series: seed 6 picked up the seed-5 checkpoint"

# garbage / truncated checkpoint files must not crash the run
newdir te_garbage
head -c 7 /dev/urandom >ecm_te_m_239_c0.ckpt
printf 'PrMers' >ecm2_te_m_239_c0.ckpt
: >ecm_te_m_239_c0.ckpt.old
start_bg g1.log "${TE[@]}" -seed 9
wait_for g1.log 'Stage1 [0-9]*/[0-9]' 90 || fail "TE: garbage checkpoint stopped the run"
stop_bg
grep -aq 'curve_seed=9' g1.log || fail "TE: garbage checkpoint changed the seed"

# ------------------------------------------------- 2. -cmont seed series resume
newdir mont
SER=(239 -ecm -cmont -b1 6000 -b2 0 -seed 7 -K 3 -ecm-continue-after-factor)
start_bg m1.log "${SER[@]}"
wait_for m1.log 'Curve 2/3 | Stage1 start' 120 || fail "cmont series: curve 2 never started"
sleep 1; stop_bg
start_bg m2.log "${SER[@]}"
wait_for m2.log 'Resuming at curve' 60 || fail "cmont series: a restart did not resume at the interrupted curve"
sleep 1; stop_bg
n=$(flat m2.log | sed -n 's/.*Resuming at curve \([0-9]*\)\/3.*/\1/p' | head -1)
[ -n "$n" ] && [ "$n" -ge 2 ] || fail "cmont series: resumed at curve '$n', expected 2 or later"
flat m2.log | grep -aq "Curve 1/3 | Stage1 start" && fail "cmont series: completed curve 1 was redone"

# a checkpoint of another series is ignored
start_bg m3.log 239 -ecm -cmont -b1 6000 -b2 0 -seed 9 -K 3 -ecm-continue-after-factor
wait_for m3.log 'Stage1 start' 60 || fail "cmont series: seed 9 run did not start"
sleep 1; stop_bg
flat m3.log | grep -aq 'Resuming at curve' && fail "cmont series: seed 9 resumed a checkpoint of seed 7"
flat m3.log | grep -aq 'Curve 1/3 | Stage1 start' || fail "cmont series: seed 9 did not start at curve 1"

# a single forced seed never resumes from the probe (K ignored)
newdir mont_single
start_bg o1.log 239 -ecm -cmont -b1 6000 -b2 0 -seed 7
wait_for o1.log 'Stage1 [0-9]*/[0-9]' 60 || wait_for o1.log 'Stage1 start' 5 || fail "cmont single: no start"
stop_bg
flat o1.log | grep -aq 'Resuming at curve' && fail "cmont single: unexpected series resume"

# ------------------------------------------------- 3. factors found earlier
newdir known
run_fg k.log 239 -ecm -cmont -b1 800 -b2 0 -seed 7 -K 3 -ecm-continue-after-factor
flat k.log | grep -aq 'Curve 1/3 | factor=5256967999' || fail "known: curve 1 factor changed (test data out of date?)"
flat k.log | grep -aq 'Curve 2/3 | factor=176383' || fail "known: curve 2 did not report only the new factor 176383"
flat k.log | grep -a 'Curve [0-9]/3 | factor=927239786567617' && fail "known: the product 5256967999*176383 was reported as a new factor"
flat k.log | grep -aq 'Curve 3/3 | known factor=' || fail "known: curve 3 did not report a known factor"

[ "$FAILURES" -eq 0 ] || exit 1
echo "ecm resume/seed/known-factor test passed"
