#!/usr/bin/env bash
# PRP end of job, both drivers: result first, then the worktodo entry, then the
# checkpoint / loop state and the proof residues. The worktodo update is made to
# fail (worktodo_save.txt is a directory) after the result has been saved. The
# saved state and the residues must survive, so that the next start, which runs
# the entry again, resumes at the end. Once the update can succeed, the entry is
# retired and everything is removed.
# usage: run_prp_worktodo_retire_failure.sh <device> [exponent]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-11213}"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

check_driver() {
    local name="$1" state="$2"; shift 2
    local work rc=0
    work="$(mktemp -d)"
    (
        cd "$work"
        ln -s "$ROOT/kernels" kernels
        echo "PRP=1,2,$P,-1" > worktodo.txt
        mkdir worktodo_save.txt

        # -t 1 saves the state every second, so some exists when the test ends.
        set +e
        $RUN timeout 300 "$ROOT/prmers" -worktodo worktodo.txt -prp -proof 2 -t 1 "$@" -d "$DEVICE" -noask -f "$work" > fail.log 2>&1
        rc=$?
        set -e
        echo "$name: RETIRE_FAILURE_RUN_RC=$rc"

        grep -q 'Failed to update worktodo.txt' fail.log || { echo "FAIL($name): worktodo update did not fail"; cat fail.log; exit 1; }
        grep -q "\"exponent\":$P" results.txt || { echo "FAIL($name): result not saved"; exit 1; }
        grep -q "PRP=1,2,$P,-1" worktodo.txt || { echo "FAIL($name): entry no longer in worktodo.txt"; exit 1; }
        [ -f "$state" ] || { echo "FAIL($name): $state deleted although the entry was not retired"; cat fail.log; exit 1; }
        [ -n "$(ls -A "$P/proof" 2>/dev/null)" ] || { echo "FAIL($name): proof residues deleted although the entry was not retired"; cat fail.log; exit 1; }
        grep -q 'worktodo entry could not be removed' fail.log || { echo "FAIL($name): no notice that the residues are kept"; cat fail.log; exit 1; }
        echo "$name: RETIRE_FAILURE_KEEPS_STATE=PASS"

        rmdir worktodo_save.txt
        set +e
        $RUN timeout 300 "$ROOT/prmers" -worktodo worktodo.txt -prp -proof 2 -t 1 "$@" -d "$DEVICE" -noask -f "$work" > ok.log 2>&1
        rc=$?
        set -e
        echo "$name: RETRY_RUN_RC=$rc"

        grep -q 'Entry removed from worktodo.txt' ok.log || { echo "FAIL($name): entry not retired on retry"; cat ok.log; exit 1; }
        if grep -q "PRP=1,2,$P,-1" worktodo.txt; then echo "FAIL($name): entry still pending"; exit 1; fi
        [ ! -e "$state" ] || { echo "FAIL($name): $state left after the entry was retired"; exit 1; }
        [ ! -e "$P/proof" ] || { echo "FAIL($name): proof residues left after the entry was retired"; exit 1; }
        echo "$name: RETIRED_THEN_CLEANED=PASS"
    ) || rc=$?
    rm -rf "$work"
    return "$rc"
}

fail=0
check_driver marin "m_$P.ckpt" || fail=1
check_driver legacy "${P}prp.loop" -marin || fail=1
exit "$fail"
