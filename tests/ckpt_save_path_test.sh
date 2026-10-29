#!/usr/bin/env bash
# The LL-SAFE, LL-SAFE2 and ECM checkpoints live under the save path (-f), with the other
# state of the run, not in the working directory.  Like the PRP/LL checkpoint and the proof
# files before them (run_proof_save_path_regression.sh):
#   1. an interrupted run leaves its checkpoint under -f and none in the working directory;
#   2. the same command resumes it;
#   3. a run that finds none under -f but one in the working directory (left by an older
#      version) resumes from it, says so, and removes it once its own is written or the job
#      is done.
# Small exponents only; run on PoCL or any OpenCL device.
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
fail() { echo "FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }

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
  timeout --signal=INT --kill-after=10s 300 "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1
  return 0
}
flat() { tr '\r' '\n' <"$1"; }
has() { ls $1 >/dev/null 2>&1; } # has <glob>

# check_mode <name> <checkpoint file prefix> <progress pattern> <resume pattern> <args...>
# <args> select the mode and must not contain -f.
check_mode() {
  local name=$1 ck=$2 progress=$3 resumed=$4; shift 4
  newdir "$name"
  local W="$PWD"

  # 1. an interrupted run keeps its checkpoint under -f
  start_bg a.log "$@" -f "$W/save"
  wait_for a.log "$progress" 120 || { fail "$name: run 1 never reached '$progress'"; stop_bg; return; }
  sleep "${SETTLE:-1}" # let the run get past its first iterations
  stop_bg
  has "$W/save/$ck*" || fail "$name: no checkpoint under -f after an interrupt"
  has "$W/$ck*" && fail "$name: checkpoint written to the working directory: $(ls "$W"/$ck*)"

  # 2. the same command resumes it
  start_bg b.log "$@" -f "$W/save"
  wait_for b.log "$resumed" 120 || fail "$name: the run did not resume from the checkpoint under -f"
  stop_bg
  flat b.log | grep -aq 'from the working directory' && fail "$name: resumed from the working directory although -f holds one"

  # 3. a checkpoint left in the working directory by an older version is adopted
  # (interrupt a fresh run, then move its checkpoint to the working directory)
  start_bg c.log "$@" -f "$W/save1"
  wait_for c.log "$progress" 120 || { fail "$name: run 3 never reached '$progress'"; stop_bg; return; }
  sleep "${SETTLE:-1}"
  stop_bg
  has "$W/save1/$ck*" || { fail "$name: run 3 left no checkpoint"; return; }
  mv "$W"/save1/"$ck"* "$W"/
  run_fg d.log "$@" -f "$W/save2"
  flat d.log | grep -aq "No checkpoint under .*save2; using $ck.* from the working directory" || fail "$name: no note about the old checkpoint"
  flat d.log | grep -aq "$resumed" || fail "$name: the old checkpoint was not resumed"
  has "$W/$ck*" && fail "$name: old checkpoint left in the working directory: $(ls "$W"/$ck*)"
  has "$W/save2/$ck*" && fail "$name: checkpoint left under -f after the job finished"
  return 0
}

# ECM stage 1, Montgomery and twisted Edwards curves
check_mode ecm_mont ecm_m_239_c0.ckpt 'Stage1 [0-9]*/[0-9]' 'resumed from Stage1 checkpoint' \
  239 -ecm -cmont -notorsion -b1 6000 -b2 0 -seed 7
check_mode ecm_te ecm_te_m_239_c0.ckpt 'Stage1 [0-9]*/[0-9]' 'resumed from Stage1 checkpoint' \
  239 -ecm -b1 6000 -b2 0 -seed 111
# ECM stage 2 (p = 701 has no small factor, so the curve reaches stage 2)
check_mode ecm_mont_s2 ecm2_m_701_c0.ckpt 'Stage2 [0-9]' 'Resuming Stage2' \
  701 -ecm -cmont -torsion8 -b1 50 -b2 20000 -seed 11
check_mode ecm_te_s2 ecm2_te_m_701_c0.ckpt 'Stage2 [0-9]' 'resumed from Stage2 checkpoint' \
  701 -ecm -b1 50 -b2 20000 -seed 11
# LL-SAFE and LL-SAFE2
SETTLE=3 check_mode llsafe llsafe_m_21701.ckpt 'LL-SAFE on' 'Resuming from a checkpoint' \
  21701 -ll -llsafeb 4000 -engine-marin
check_mode llsafe2 llsafe2_m_11213.ckpt 'Check passed' 'Resuming from a checkpoint' \
  11213 -llsafe2 -llsafeb 2000 -engine-marin

[ "$FAILURES" -eq 0 ] || exit 1
echo "checkpoint save-path test passed"
