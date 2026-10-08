#!/usr/bin/env bash
# A P-1 stage 2 that is interrupted or fails has not finished the job: the
# stage-1 checkpoint and the worktodo line must stay, and the run must not move
# on to the next entry.  It used to treat both like a finished stage 2.
#   error:       a stage-2 start bound above B2 makes the low-memory stage 2 fail at once.
#   interrupted: SIGINT while stage 2 of M677 (B1=10, B2=10^7) is running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# setup <name> <worktodo line>: fresh run directory with a worktodo.txt that has
# the entry to test followed by a second entry that must stay untouched.
setup() {
  mkdir -p "$WORK/$1"
  ln -s "$ROOT/kernels" "$WORK/$1/kernels"
  printf '%s\nPminus1=1,2,269,-1,4,2141\n' "$2" >"$WORK/$1/worktodo.txt"
}

# check_kept <name> <exponent>
check_kept() {
  local d="$WORK/$1"
  grep -q "^Pminus1=1,2,$2," "$d/worktodo.txt" || { echo "$1: the worktodo line was removed" >&2; exit 1; }
  if grep -q 'Entry removed from\|Restarting for next entry' "$d/run.log"; then
    echo "$1: the run treated the entry as finished" >&2; exit 1
  fi
  ls "$d"/pm1_m_"$2".ckpt >/dev/null 2>&1 || { echo "$1: the stage-1 checkpoint was deleted" >&2; exit 1; }
  grep -q "Stage 2 $3; keeping the checkpoint" "$d/run.log" || { echo "$1: no 'Stage 2 $3' message" >&2; exit 1; }
}

# 1. Error.
setup err 'Pminus1=1,2,269,-1,100,1000,0,2000'
rc=0
( cd "$WORK/err" && timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" -pm1 -pm1-lowmem -d "$DEVICE" --noask >run.log 2>&1 ) || rc=$?
[ "$rc" -eq 2 ] || { echo "err: exit status $rc, expected 2" >&2; exit 1; }
check_kept err 269 failed

# 2. Interrupt.
setup int 'Pminus1=1,2,677,-1,10,10000000'
cd "$WORK/int"
"$ROOT/prmers" -pm1 -d "$DEVICE" --noask >run.log 2>&1 &
pid=$!
cd "$WORK"
# Wait until the stage-2 loop itself is running (the planning before it takes a
# while on a busy machine and is not interruptible).
marker='executing greedy irregular pair plan'
for _ in $(seq 1 400); do
  grep -q "$marker" "$WORK/int/run.log" 2>/dev/null && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
grep -q "$marker" "$WORK/int/run.log" || { kill -KILL "$pid" 2>/dev/null || true; echo "int: stage 2 did not start" >&2; exit 1; }
sleep 2
kill -INT "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "int: prmers did not stop within 30 s of SIGINT" >&2; exit 1
fi
rc=0
wait "$pid" || rc=$?
[ "$rc" -eq 0 ] || { echo "int: exit status $rc, expected 0" >&2; exit 1; }
check_kept int 677 interrupted
echo "pm1 stage-2 incomplete test passed"
