#!/usr/bin/env bash
# LL-SAFE end of job: result first, then the worktodo entry, then the checkpoint.
# The worktodo update is made to fail (worktodo_save.txt is a directory), after the
# result has been saved. The checkpoint must survive, so that the next start, which
# runs the entry again, resumes at the end instead of from iteration 0. Once the
# update can succeed, the entry is retired and the checkpoint removed.
# usage: run_llsafe_worktodo_retire_failure.sh <device> [exponent]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-11213}"
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

CKPT="llsafe_m_$P.ckpt"
echo "DoubleCheck=$P,70,1" > worktodo.txt
mkdir worktodo_save.txt

# -t 1 writes a checkpoint every second, so one exists when the test ends.
set +e
$RUN timeout 300 "$ROOT/prmers" -worktodo worktodo.txt -t 1 -d "$DEVICE" -noask -f "$WORK" > fail.log 2>&1
rc=$?
set -e
echo "RETIRE_FAILURE_RUN_RC=$rc"

grep -q 'Failed to update worktodo.txt' fail.log || { echo "FAIL: worktodo update did not fail"; cat fail.log; exit 1; }
grep -q "\"exponent\":$P" results.txt || { echo "FAIL: result not saved"; exit 1; }
grep -q "DoubleCheck=$P" worktodo.txt || { echo "FAIL: entry no longer in worktodo.txt"; exit 1; }
[ -f "$CKPT" ] || { echo "FAIL: checkpoint deleted although the entry was not retired"; cat fail.log; exit 1; }
grep -q 'keeping the checkpoint' fail.log || { echo "FAIL: no notice that the checkpoint is kept"; cat fail.log; exit 1; }
echo "RETIRE_FAILURE_KEEPS_CHECKPOINT=PASS"

rmdir worktodo_save.txt
set +e
$RUN timeout 300 "$ROOT/prmers" -worktodo worktodo.txt -t 1 -d "$DEVICE" -noask -f "$WORK" > ok.log 2>&1
rc=$?
set -e
echo "RETRY_RUN_RC=$rc"

grep -q 'Resuming from a checkpoint' ok.log || { echo "FAIL: retry did not resume from the kept checkpoint"; cat ok.log; exit 1; }
grep -q 'Entry removed from worktodo.txt' ok.log || { echo "FAIL: entry not retired on retry"; cat ok.log; exit 1; }
if grep -q "DoubleCheck=$P" worktodo.txt; then echo "FAIL: entry still pending"; exit 1; fi
grep -q "DoubleCheck=$P" worktodo_save.txt
if find . -maxdepth 1 -name "$CKPT*" | grep -q .; then echo "FAIL: checkpoint left after the entry was retired"; exit 1; fi
echo "RETIRED_THEN_CLEANED=PASS"
