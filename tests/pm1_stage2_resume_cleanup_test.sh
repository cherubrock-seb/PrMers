#!/usr/bin/env bash
# Resuming an interrupted classic (-pm1-vtrace-off) stage 2 must finish the job:
# remove the stage-1 checkpoint and the worktodo line and report the stage-2
# result once.  It used to call stage 2 directly when pm1_s2_m_<p>.ckpt existed,
# which left the stage-1 checkpoint and the worktodo line behind, so the next
# run did the whole stage 2 again and appended a second result.
#   M9941, B1=1000, B2=4000000.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

mkdir -p "$WORK/run"
ln -s "$ROOT/kernels" "$WORK/run/kernels"
printf 'Pminus1=1,2,9941,-1,1000,4000000\n' >"$WORK/run/worktodo.txt"
cd "$WORK/run"

# 1. Interrupt the stage-2 loop.
"$ROOT/prmers" -pm1 -pm1-vtrace-off -d "$DEVICE" --noask >first.log 2>&1 &
pid=$!
marker='canonical giant/baby multiplication enabled'
for _ in $(seq 1 1200); do
  grep -q "$marker" first.log 2>/dev/null && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
grep -q "$marker" first.log || { kill -KILL "$pid" 2>/dev/null || true; echo "first run: stage 2 did not start" >&2; exit 1; }
sleep 2
kill -INT "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "first run: prmers did not stop within 30 s of SIGINT" >&2; exit 1
fi
wait "$pid" 2>/dev/null || true
ls pm1_s2_m_9941.ckpt >/dev/null 2>&1 || { echo "first run: no stage-2 checkpoint (stage 2 finished before the interrupt?)" >&2; exit 1; }
ls pm1_m_9941.ckpt >/dev/null 2>&1 || { echo "first run: the stage-1 checkpoint was deleted" >&2; exit 1; }
grep -q '^Pminus1=1,2,9941,' worktodo.txt || { echo "first run: the worktodo line was removed" >&2; exit 1; }

# 2. Rerun: resumes stage 2 and completes the entry.
timeout --signal=INT --kill-after=10s 600 "$ROOT/prmers" -pm1 -pm1-vtrace-off -d "$DEVICE" --noask >second.log 2>&1 || true
grep -q 'Resuming Stage 2 from checkpoint' second.log || { echo "rerun did not resume stage 2" >&2; exit 1; }
grep -q 'Entry removed from' second.log || { echo "rerun did not finish the entry" >&2; exit 1; }
for f in pm1_m_9941.ckpt pm1_m_9941.ckpt.old pm1_m_9941.ckpt.backend pm1_s2_m_9941.ckpt pm1_s2_m_9941.ckpt.old pm1_s2_m_9941.ckpt.new; do
  [ ! -e "$f" ] || { echo "rerun left $f behind" >&2; exit 1; }
done
if grep -q '^Pminus1=1,2,9941,' worktodo.txt 2>/dev/null; then echo "rerun left the worktodo line" >&2; exit 1; fi
n=$(grep -c '"b2":4000000' results.txt || true)
[ "$n" -eq 1 ] || { echo "expected one stage-2 result line, found $n" >&2; exit 1; }
echo "pm1 stage-2 resume cleanup test passed"
