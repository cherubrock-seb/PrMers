#!/usr/bin/env bash
# -pm1-ultralowmem keeps the stage-1 checkpoint when the run dies during stage 2
# (the one-register stage 2 cannot checkpoint).  That checkpoint has i = 0 and
# no chunk size, so the rerun must treat stage 1 as complete instead of
# rebuilding E and failing with "-pm1-ultralowmem requires the fast3 single-chunk
# path" on every rerun.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

mkdir -p "$WORK/run"
ln -s "$ROOT/kernels" "$WORK/run/kernels"
cd "$WORK/run"

# First run: kill it hard while the one-register stage 2 is running.
"$ROOT/prmers" 9941 -pm1 -b1 1000 -b2 20000000 -pm1-ultralowmem -d "$DEVICE" --noask >first.log 2>&1 &
pid=$!
marker='Ultra-low-memory Stage 2 legacy one-register mode'
for _ in $(seq 1 400); do
  grep -q "$marker" first.log 2>/dev/null && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
grep -q "$marker" first.log || { kill -KILL "$pid" 2>/dev/null || true; echo "first run: stage 2 did not start" >&2; exit 1; }
sleep 1
kill -KILL "$pid"
wait "$pid" 2>/dev/null || true
ls pm1_m_9941.ckpt >/dev/null 2>&1 || { echo "the stage-1 checkpoint was not left behind" >&2; exit 1; }

# Rerun: it must get through to stage 2 again (then we stop it).
timeout --signal=INT --kill-after=10s 20 "$ROOT/prmers" 9941 -pm1 -b1 1000 -b2 20000000 -pm1-ultralowmem -d "$DEVICE" --noask >second.log 2>&1 || true
if grep -q 'requires the fast3 single-chunk path' second.log; then
  echo "rerun rejected the stage-1 checkpoint of a finished stage 1" >&2; exit 1
fi
grep -q 'Stage 1 checkpoint is complete' second.log || { echo "rerun did not recognise the finished stage 1" >&2; exit 1; }
grep -q "$marker" second.log || { echo "rerun did not reach stage 2" >&2; exit 1; }
echo "pm1 ultralowmem resume test passed"
