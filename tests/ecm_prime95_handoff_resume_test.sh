#!/usr/bin/env bash
# Twisted-Edwards ECM with the Prime95 stage-2 handoff (-p95stage2): a curve
# handed to Prime95 must keep its stage-1 state until Prime95 reports a result
# for it.  A stub "mprime" stands in for Prime95.
#
#  1. SIGINT / SIGTERM while Prime95 runs curve 1 and curve 2 is in stage 1:
#     PrMers stops promptly (and stops Prime95), curve 1 stays pending, and
#     the restart hands curve 1 to Prime95 again before "no factor".
#  2. Prime95 exits without a result: every queued curve stays pending and
#     the restart hands all of them to Prime95.
#  3. a status "F" result without a factor is an error, not "no factor".
#  4. a pending curve with no way to finish it (no -p95stage2, or its resume
#     file is gone) blocks the "no factor" result.
#  5. a garbage marker blocks the "no factor" result.
#  6. a marker that cannot be written falls back to the internal stage 2.
#  7. Ctrl-C (SIGINT to the process group) while waiting for Prime95 at the end.
#
# 2^1279-1 is prime, so no curve can find a factor in stage 1.
# usage: ecm_prime95_handoff_resume_test.sh [device]   (runs prmers on <device>)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
  for f in "$WORK"/*/p95/stub.pid; do [ -f "$f" ] && kill -KILL "$(cat "$f")" 2>/dev/null; done
  rm -rf "$WORK"
}
trap cleanup EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

FAILS=0
check() { # check <description> <command...>
  local d="$1"; shift
  if "$@"; then echo "  ok:   $d"; else echo "  FAIL: $d"; FAILS=$((FAILS + 1)); fi
}

# Stub Prime95.  Every call appends its worktodo line to calls.log.
#   sleep: waits up to 30 s (interruptibly), then exits without a result
#   nf:    writes a "no factor" result
#   fail:  exits 1 without a result
#   fempty: writes status "F" without any factor
make_stub() { # make_stub <dir> <kind>
  local d="$1/p95"
  mkdir -p "$d"
  {
    echo '#!/bin/sh'
    echo 'echo $$ > stub.pid'
    echo 'cat worktodo.txt >> calls.log'
    case "$2" in
      sleep) echo "trap 'kill \$S 2>/dev/null; exit 0' TERM INT"
             echo 'sleep 30 & S=$!'
             echo 'wait $S' ;;
      nf)    echo "echo '{\"status\":\"NF\", \"exponent\":1279, \"worktype\":\"ECM\"}' > results.json.txt" ;;
      fail)  echo 'exit 1' ;;
      fempty) echo "echo '{\"status\":\"F\", \"exponent\":1279, \"worktype\":\"ECM\"}' > results.json.txt" ;;
    esac
  } >"$d/mprime"
  chmod +x "$d/mprime"
}

new_case() { # new_case <name>
  local w="$WORK/$1"
  mkdir -p "$w"
  ln -s "$ROOT/kernels" "$w/kernels"
  echo "$w"
}

run_prmers() { # run_prmers <dir> <logname> <args...>; foreground, 300 s cap
  local w="$1" log="$2"; shift 2
  ( cd "$w" && timeout -k 10 300 "$BIN" 1279 -ecm -ced -notorsion -p95stage2 -p95path ./p95 \
      -d "$DEVICE" --noask "$@" >"$log" 2>&1 )
}

run_plain() { # run_plain <dir> <logname> <args...>; no Prime95 handoff
  local w="$1" log="$2"; shift 2
  ( cd "$w" && timeout -k 10 300 "$BIN" 1279 -ecm -ced -notorsion -d "$DEVICE" --noask "$@" >"$log" 2>&1 )
}

start_bg() { # start_bg <dir> <logname> <args...>; sets PID (timeout, own session) and PRM (prmers)
  local w="$1" log="$2"; shift 2
  ( cd "$w" && exec setsid timeout -k 10 300 "$BIN" 1279 -ecm -ced -notorsion -p95stage2 -p95path ./p95 \
      -d "$DEVICE" --noask "$@" >"$log" 2>&1 ) &
  PID=$!
  PIDS+=("$PID")
  local i
  for ((i = 0; i < 50; i++)); do
    PRM="$(pgrep -P "$PID" | head -1)"
    [ -n "$PRM" ] && return 0
    sleep 0.1
  done
  return 1
}

# stop_and_time <signal-target> <signal> : signal, wait up to 20 s, else SIGTERM and wait
stop_and_time() {
  local T0; T0=$(date +%s)
  kill "-$2" -- "$1"
  if wait_exit "$PID" 20; then
    echo "  ok:   prmers exited $(( $(date +%s) - T0 )) s after SIG$2"
  else
    echo "  FAIL: prmers still running 20 s after SIG$2; sending SIGTERM"
    FAILS=$((FAILS + 1))
    kill -TERM "$PRM" 2>/dev/null
    wait_exit "$PID" 200 || true
    echo "        exited $(( $(date +%s) - T0 )) s after the first signal"
  fi
  wait "$PID" 2>/dev/null || true
}

wait_for() { # wait_for <file> <pattern> <seconds>
  local i
  for ((i = 0; i < $3 * 10; i++)); do
    grep -aq -- "$2" "$1" 2>/dev/null && return 0
    sleep 0.1
  done
  return 1
}

wait_exit() { # wait_exit <pid> <seconds>; 0 if it exited in time
  local i
  for ((i = 0; i < $2 * 10; i++)); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

called() { grep -aq "_c$(printf '%06d' "$2").p95" "$1/p95/calls.log" 2>/dev/null; }
pending() { [ -f "$1/resume_p1279_ECM_TE_B1_$2_c$(printf '%06d' "$3").p95.pending" ]; }
no_pending() { ! ls "$1"/*.pending >/dev/null 2>&1; }
stub_gone() { [ ! -f "$1/p95/stub.pid" ] || ! kill -0 "$(cat "$1/p95/stub.pid")" 2>/dev/null; }
recovered() { grep -aq "Curve $2 | Stage2 still owed by an earlier Prime95 handoff" "$1"; }
no_nf() { ! grep -aq 'No factor found' "$1"; }

# Interrupt while Prime95 runs curve 1 and curve 2 is in stage 1, then restart.
interrupt_case() { # interrupt_case <name> <signal>
  echo "[$1] SIG$2 to prmers while Prime95 runs curve 1 and curve 2 is in stage 1"
  local w; w="$(new_case "$1")"
  make_stub "$w" sleep
  start_bg "$w" run1.log -b1 5000 -b2 10000 -K 3 || { echo "  FAIL: prmers did not start"; exit 1; }
  if ! wait_for "$w/run1.log" 'Curve 2/3 | Stage1' 120; then
    echo "  FAIL: curve 2 stage 1 never started"; tail -20 "$w/run1.log"; exit 1
  fi
  stop_and_time "$PRM" "$2"
  check "Prime95 stub was stopped with prmers" stub_gone "$w"
  check "curve 1 is still pending for Prime95" pending "$w" 5000 1
  check "curve 2 kept its stage-1 checkpoint" test -f "$w/ecm_te_m_1279_c1.ckpt"
  check "no final result was written" no_nf "$w/run1.log"

  echo "[$1] restart with a working Prime95"
  make_stub "$w" nf
  : >"$w/p95/calls.log"
  run_prmers "$w" run2.log -b1 5000 -b2 10000 -K 3
  check "curve 1 handed to Prime95 again (no new stage 1)" recovered "$w/run2.log" 1
  check "curve 1 Prime95 stage 2 ran" called "$w" 1
  check "curve 2 resumed from its checkpoint" grep -aq 'resumed from Stage1 checkpoint, curve index 2' "$w/run2.log"
  check "curve 2 handed to Prime95" called "$w" 2
  check "curve 3 handed to Prime95" called "$w" 3
  check "no pending markers left" no_pending "$w"
  check "final no-factor result" grep -aq 'No factor found' "$w/run2.log"
}

interrupt_case 1-sigint INT
interrupt_case 1-sigterm TERM

# ------------------------------------------------------------------ 2. failure
echo "[2] Prime95 fails while later curves are queued"
W3="$(new_case failure)"
make_stub "$W3" fail
run_prmers "$W3" run1.log -b1 5000 -b2 10000 -K 10
check "the failure is reported" grep -aq 'Prime95 Stage2 background error' "$W3/run1.log"
check "no final result was written" no_nf "$W3/run1.log"
NQ=$(grep -ac 'Stage2 queued for Prime95 background' "$W3/run1.log" || true)
NP=$(ls "$W3"/*.pending 2>/dev/null | wc -l)
echo "        curves handed off: $NQ, pending markers kept: $NP"
check "every handed-off curve is still pending" test "$NQ" -gt 0 -a "$NP" -eq "$NQ"
make_stub "$W3" nf
: >"$W3/p95/calls.log"
run_prmers "$W3" run2.log -b1 5000 -b2 10000 -K 10
for c in $(seq 1 "$NQ"); do check "restart: curve $c handed to Prime95 again" recovered "$W3/run2.log" "$c"; done
for c in $(seq 1 10); do check "restart: curve $c Prime95 stage 2 ran" called "$W3" "$c"; done
check "restart: no pending markers left" no_pending "$W3"
check "restart: final no-factor result" grep -aq 'No factor found' "$W3/run2.log"

# ------------------------------------------------------------ 3. F, no factor
echo "[3] Prime95 status F without a factor"
W4="$(new_case fempty)"
make_stub "$W4" fempty
run_prmers "$W4" run1.log -b1 3 -b2 2000 -K 1
check "not reported as no factor" no_nf "$W4/run1.log"
check "reported as an error" grep -aq 'status F' "$W4/run1.log"
check "curve 1 is still pending" pending "$W4" 3 1

# --------------------------------------- 4. pending curve, handoff unavailable
echo "[4] restart without -p95stage2 while a curve is pending"
run_plain "$W4" run2.log -b1 3 -b2 2000 -K 1
check "refuses to claim no factor" no_nf "$W4/run2.log"
check "names the pending curve" grep -aq 'pending: .*c000001.p95.pending' "$W4/run2.log"
check "curve 1 is still pending" pending "$W4" 3 1
echo "[4] restart with -p95stage2 but the resume file is gone"
mv "$W4/resume_p1279_ECM_TE_B1_3_c000001.p95" "$W4/saved.p95"
make_stub "$W4" nf
run_prmers "$W4" run3.log -b1 3 -b2 2000 -K 1
check "refuses to claim no factor" no_nf "$W4/run3.log"
check "reports the missing resume file" grep -aq 'is missing' "$W4/run3.log"
mv "$W4/saved.p95" "$W4/resume_p1279_ECM_TE_B1_3_c000001.p95"
run_prmers "$W4" run4.log -b1 3 -b2 2000 -K 1
check "finishes once the resume file is back" grep -aq 'No factor found' "$W4/run4.log"
check "no pending markers left" no_pending "$W4"

# ------------------------------------------------- 5. garbage pending marker
echo "[5] garbage pending marker"
W5="$(new_case garbage)"
make_stub "$W5" nf
printf 'p=1279\nB1=3\ncurve=-1\n' >"$W5/resume_p1279_ECM_TE_B1_3_c000001.p95.pending"
run_prmers "$W5" run1.log -b1 3 -b2 2000 -K 1
check "refuses to claim no factor" no_nf "$W5/run1.log"
check "reports the marker" grep -aq 'unreadable or inconsistent' "$W5/run1.log"

# ------------------------------------------ 6. marker cannot be written
echo "[6] the pending marker cannot be written: internal stage 2 instead"
W6="$(new_case nomarker)"
make_stub "$W6" nf
mkdir -p "$W6/resume_p1279_ECM_TE_B1_3_c000001.p95.pending.new"
run_prmers "$W6" run1.log -b1 3 -b2 2000 -K 1
check "falls back to the internal stage 2" grep -aq 'falling back to internal Stage2' "$W6/run1.log"
check "Prime95 not used for the curve" bash -c "! grep -aq c000001 '$W6/p95/calls.log' 2>/dev/null"
check "final no-factor result" grep -aq 'No factor found' "$W6/run1.log"

# ----------------------------- 7. Ctrl-C to the process group while draining
echo "[7] Ctrl-C (SIGINT to the process group) while waiting for Prime95 at the end"
W7="$(new_case drain)"
make_stub "$W7" sleep
start_bg "$W7" run1.log -b1 3 -b2 2000 -K 2 || { echo "  FAIL: prmers did not start"; exit 1; }
if ! wait_for "$W7/run1.log" 'Waiting for Prime95 Stage2 background jobs' 120; then
  echo "  FAIL: never reached the final wait"; tail -20 "$W7/run1.log"; exit 1
fi
stop_and_time "-$PID" INT
check "Prime95 stub was stopped" stub_gone "$W7"
check "curve 1 is still pending" pending "$W7" 3 1
check "curve 2 is still pending" pending "$W7" 3 2
check "no final result was written" no_nf "$W7/run1.log"

if [ "$FAILS" -ne 0 ]; then
  echo "ecm Prime95 handoff resume test: $FAILS check(s) failed" >&2
  exit 1
fi
echo "ecm Prime95 handoff resume test passed"
