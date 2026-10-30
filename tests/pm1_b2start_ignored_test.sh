#!/usr/bin/env bash
# -b2start (and the worktodo B2_start field) is read only by the low-memory stage 2 and the
# Prime95 handoff.  The default V-trace stage 2 and the classic BSGS stage 2 search all of
# (B1, B2], so they must say that the option is ignored instead of dropping it silently.
#   M269: 13822297 = 2*269*(2^2*3*2141) + 1, found by stage 2 with B1=4, B2=2141.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

fail() { echo "FAIL: $*" >&2; exit 1; }

# run_case <name> <prmers args...>
run_case() {
  local name="$1"; shift
  mkdir -p "$WORK/$name"; ln -s "$ROOT/kernels" "$WORK/$name/kernels"
  ( cd "$WORK/$name" && timeout --signal=INT --kill-after=10s 100 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >run.log 2>&1 || true )
}
warned() { grep -aq "\[PM1\] Warning: -b2start $2 is ignored by the $3 Stage 2; it searches all of (4, 2141]" "$WORK/$1/run.log"; }
found() { grep -aq 'P-1 factor stage 2 found: 13822297' "$WORK/$1/run.log"; }

run_case vtrace 269 -pm1 -b1 4 -b2 2141 -b2start 100
warned vtrace 100 'V-trace' || fail "vtrace: no warning that -b2start is ignored"
found vtrace || fail "vtrace: the full range was not searched"
grep -aq 'b2-start' "$WORK/vtrace/results.txt" && fail "vtrace: the result claims a start bound that was not used"

run_case classic 269 -pm1 -b1 4 -b2 2141 -b2start 100 -pm1-vtrace-off
warned classic 100 'classic BSGS' || fail "classic: no warning that -b2start is ignored"
found classic || fail "classic: the full range was not searched"

# The worktodo B2_start field reaches the same stage 2.
mkdir -p "$WORK/worktodo"; ln -s "$ROOT/kernels" "$WORK/worktodo/kernels"
echo 'Pminus1=1,2,269,-1,4,2141,0,100' >"$WORK/worktodo/worktodo.txt"
( cd "$WORK/worktodo" && timeout --signal=INT --kill-after=10s 100 "$ROOT/prmers" -pm1 -d "$DEVICE" --noask >run.log 2>&1 || true )
warned worktodo 100 'V-trace' || fail "worktodo: no warning that B2_start is ignored"

# Nothing is ignored, so nothing is said.
run_case plain 269 -pm1 -b1 4 -b2 2141
grep -aq 'is ignored by' "$WORK/plain/run.log" && fail "plain: warning without -b2start"
run_case below 269 -pm1 -b1 4 -b2 2141 -b2start 3
grep -aq 'is ignored by' "$WORK/below/run.log" && fail "below: warning for a start bound not above B1"
run_case lowmem 269 -pm1 -b1 4 -b2 2141 -b2start 100 -pm1-lowmem
grep -aq 'is ignored by' "$WORK/lowmem/run.log" && fail "lowmem: warning although the low-memory stage 2 uses -b2start"
found lowmem || fail "lowmem: factor not found"

echo "pm1 -b2start ignored-warning test passed"
