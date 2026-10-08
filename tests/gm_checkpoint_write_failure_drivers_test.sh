#!/usr/bin/env bash
# A checkpoint that cannot be written (full or read-only disk) must be a warning in the
# Gaussian-Mersenne PRP/Proth, ECM NAF, optimized ECM and trial-factoring drivers, not the
# end of the run.  The failure is forced by making `<checkpoint>.new` (`.tmp` for TF) a
# symlink to /dev/full, so every write to it fails with ENOSPC.
#   periodic save: the run must go on (and the next save, with the symlink gone, works)
#   interrupt save: the run must exit normally and say the checkpoint was not saved
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
PID=""
cleanup() { [ -n "$PID" ] && kill -KILL "$PID" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off
[ -e /dev/full ] || { echo "no /dev/full: test skipped"; exit 0; }

FAILURES=0
fail() {
  echo "FAIL: $*" >&2; FAILURES=$((FAILURES + 1))
  [ -n "${GM_TEST_KEEP_GOING:-}" ] || exit 1
}
newdir() { rm -rf "$WORK/$1"; mkdir -p "$WORK/$1"; ln -s "$ROOT/kernels" "$WORK/$1/kernels"; cd "$WORK/$1" || exit 1; }
start_bg() { local log=$1; shift; "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1 & PID=$!; }
wait_for() { # log pattern seconds
  local i
  for ((i = 0; i < $3 * 10; i++)); do
    grep -aq -- "$2" "$1" 2>/dev/null && return 0
    kill -0 "$PID" 2>/dev/null || { grep -aq -- "$2" "$1" 2>/dev/null; return; }
    sleep 0.1
  done
  return 1
}
stop_bg() { # SIGINT, wait for a normal exit; sets RC
  kill -INT "$PID" 2>/dev/null
  local i
  for ((i = 0; i < 300; i++)); do kill -0 "$PID" 2>/dev/null || break; sleep 0.1; done
  kill -KILL "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null; RC=$?
  PID=""
}
flat() { tr '\r' '\n' <"$1"; }
dead_end() { # log: the run died from a thrown error instead of warning
  flat "$1" | grep -aiq -e 'cannot write' -e 'Unable to write TF' -e 'terminate called' -e 'Error:' && ! flat "$1" | grep -aq 'was not saved'
}

# name | arguments | checkpoint file | periodic-save wait pattern (empty: no periodic test)
check_driver() {
  local name=$1 ckpt=$2 periodic=$3; shift 3
  # --- periodic save fails ---
  if [ -n "$periodic" ]; then
    newdir "${name}_periodic"
    ln -s /dev/full "$ckpt.new"
    start_bg p.log "$@" -t 1
    wait_for p.log 'was not saved' 60 || fail "$name: a failing periodic checkpoint save was not reported as a warning"
    sleep 2
    kill -0 "$PID" 2>/dev/null || fail "$name: the run ended after a checkpoint write failure"
    stop_bg
    flat p.log | grep -aq 'Interrupted' || fail "$name: the run did not reach its interrupt handler after the failed save"
    dead_end p.log && fail "$name: the failed save ended the run with an error"
    [ -s "$ckpt" ] || fail "$name: no checkpoint was written once the disk worked again"
  fi
  # --- interrupt save fails ---
  newdir "${name}_interrupt"
  start_bg i.log "$@" -t 100000
  sleep 5
  kill -0 "$PID" 2>/dev/null || fail "$name: the run ended before the interrupt (test parameters too small)"
  ln -s /dev/full "$ckpt.new"
  stop_bg
  flat i.log | grep -aq 'could not be saved' || fail "$name: a failing interrupt-time save was not reported"
  # a stopped run exits 1 (core/ExitCodes.hpp) whether or not the save worked; it must not crash or report an error
  [ "$RC" -eq 1 ] || fail "$name: exit code $RC after a failed interrupt-time save"
  dead_end i.log && fail "$name: the failed interrupt-time save ended the run with an error"
}

check_driver prp gm_prp_p100003.ckpt yes 100003 -gm-prp -gm-base 3 -gm-family GM -gm-sieve 0
check_driver naf gm_ecm_p20011_c0_stage1_naf.ckpt yes 20011 -gm-ecm -edwards -b1 300000 -K 1 -gm-family GM -gm-sieve 0
check_driver opt gm_ecm_special32_p20011_c0_stage1_fused.ckpt "" 20011 -gm-ecm-special32 -gm-family GM -b1 20000 -b2 30000 -K 1 -gm-sieve 0

# trial factoring: the temporary checkpoint cannot be written; the search finishes anyway
newdir tf
ln -s /dev/full gm_tf_p20011_30_40_GM.checkpoint.tmp
timeout --signal=INT --kill-after=10s 120 "$BIN" 20011 -gm-tf 30 40 -gm-family GM -d "$DEVICE" --noask >tf.log 2>&1
tf_rc=$?
[ "$tf_rc" -eq 0 ] || fail "tf: exit code $tf_rc after a checkpoint write failure"
flat tf.log | grep -aq 'checkpoint .* was not saved' || fail "tf: the failed checkpoint save was not reported"
[ -s gm_tf_p20011_30_40_GM_result.json ] || fail "tf: no result after a checkpoint write failure"
[ -e gm_tf_p20011_30_40_GM.checkpoint ] && fail "tf: checkpoint not removed at the end"

[ "$FAILURES" -eq 0 ] || exit 1
echo "gm checkpoint write-failure driver test passed"
