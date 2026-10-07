#!/usr/bin/env bash
# A -b1old extension that was stopped during stage 2 must not be redone: the
# stage-1 checkpoint written when the extension finished already holds the
# extended residue (i = 0, no chunk size, B1 = the new B1).  It used to be
# ignored in extend mode, so the identical rerun built E_diff and exponentiated
# again before returning to stage 2.
#   M9941: -b1 1000, then -b1old 1000 -b1 2000 with a stage 2 that is killed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

mkdir -p "$WORK/run"
ln -s "$ROOT/kernels" "$WORK/run/kernels"
cd "$WORK/run"

timeout --signal=INT --kill-after=10s 100 "$ROOT/prmers" 9941 -pm1 -b1 1000 -d "$DEVICE" --noask >step1.log 2>&1 || true
ls resume_p9941_B1_1000.save >/dev/null 2>&1 || { echo "step 1: no stage-1 resume file" >&2; exit 1; }

# 2. Extend and kill the run as soon as stage 2 starts.
"$ROOT/prmers" 9941 -pm1 -b1old 1000 -b1 2000 -b2 100000000 -d "$DEVICE" --noask >step2.log 2>&1 &
pid=$!
marker='Start a P-1 factoring : Stage 2'
for _ in $(seq 1 400); do
  grep -q "$marker" step2.log 2>/dev/null && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
grep -q "$marker" step2.log || { kill -KILL "$pid" 2>/dev/null || true; echo "step 2: stage 2 did not start" >&2; exit 1; }
kill -KILL "$pid"
wait "$pid" 2>/dev/null || true
grep -q 'Extension exponentiation done' step2.log || { echo "step 2: the extension did not run" >&2; exit 1; }
ls pm1_m_9941.ckpt >/dev/null 2>&1 || { echo "step 2: the stage-1 checkpoint was not left behind" >&2; exit 1; }
cp resume_p9941_B1_2000.save step2.save

# 3. Identical rerun: the extension must be skipped and the residue unchanged.
timeout --signal=INT --kill-after=10s 25 "$ROOT/prmers" 9941 -pm1 -b1old 1000 -b1 2000 -b2 100000000 -d "$DEVICE" --noask >step3.log 2>&1 || true
grep -q 'already holds the extended residue' step3.log || { echo "rerun did not recognise the finished extension" >&2; exit 1; }
if grep -q 'Building E_diff\|Extension exponentiation done' step3.log; then
  echo "rerun redid the extension" >&2; exit 1
fi
grep -q "$marker" step3.log || { echo "rerun did not reach stage 2" >&2; exit 1; }
cmp -s step2.save resume_p9941_B1_2000.save || { echo "the rerun changed the extended residue" >&2; exit 1; }
echo "pm1 extension stage-2 rerun test passed"
