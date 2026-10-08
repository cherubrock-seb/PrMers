#!/usr/bin/env bash
# End of job for LL-UNSAFE and Wagstaff (Marin path) and legacy Wagstaff (-marin):
# result first, then the worktodo entry, then the checkpoint / saved state. First
# the result write and then the worktodo update are made to fail; each time the
# saved state must survive, so that the next start, which runs the entry again,
# can resume. Once both succeed, the entry is retired and the state is removed.
# usage: run_worktodo_retire_failure_cleanup.sh <device> [ll-exponent] [wagstaff-q]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-11213}"
Q="${3:-42737}"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# check_case <name> <worktodo line> <state file> <prmers args...>
check_case() {
    local name="$1" entry="$2" state="$3"; shift 3
    local work rc=0
    work="$(mktemp -d)"
    (
        cd "$work"
        ln -s "$ROOT/kernels" kernels
        echo "$entry" > worktodo.txt

        # 1. The result cannot be saved (results.txt is a directory): the entry and
        # the state stay. -t 1 saves the state every second, so some exists at the end.
        mkdir results.txt
        $RUN timeout 600 "$ROOT/prmers" -worktodo worktodo.txt -t 1 "$@" -d "$DEVICE" -noask -f "$work" > nosave.log 2>&1
        echo "$name: SAVE_FAILURE_RUN_RC=$?"
        grep -qF "$entry" worktodo.txt || { echo "FAIL($name): entry removed although the result was not saved"; exit 1; }
        [ -f "$state" ] || { echo "FAIL($name): $state deleted although the result was not saved"; ls; exit 1; }
        grep -q 'Result could not be saved; keeping' nosave.log || { echo "FAIL($name): no notice that the state is kept"; cat nosave.log; exit 1; }
        echo "$name: SAVE_FAILURE_KEEPS_STATE=PASS"
        rmdir results.txt

        # 2. The result is saved but the worktodo update fails (worktodo_save.txt
        # is a directory): the entry and the state stay.
        mkdir worktodo_save.txt
        $RUN timeout 600 "$ROOT/prmers" -worktodo worktodo.txt -t 1 "$@" -d "$DEVICE" -noask -f "$work" > fail.log 2>&1
        echo "$name: RETIRE_FAILURE_RUN_RC=$?"

        [ -s results.txt ] || { echo "FAIL($name): result not saved"; exit 1; }
        grep -qF "$entry" worktodo.txt || { echo "FAIL($name): entry no longer in worktodo.txt"; exit 1; }
        [ -f "$state" ] || { echo "FAIL($name): $state deleted although the entry was not retired"; ls; exit 1; }
        grep -q 'Failed to update worktodo.txt; keeping' fail.log || { echo "FAIL($name): no worktodo failure notice"; cat fail.log; exit 1; }
        echo "$name: RETIRE_FAILURE_KEEPS_STATE=PASS"

        # 3. Both succeed: the entry is retired and the state removed.
        rmdir worktodo_save.txt
        $RUN timeout 600 "$ROOT/prmers" -worktodo worktodo.txt -t 1 "$@" -d "$DEVICE" -noask -f "$work" > ok.log 2>&1
        echo "$name: RETRY_RUN_RC=$?"

        grep -q 'Entry removed from worktodo.txt' ok.log || { echo "FAIL($name): entry not retired on retry"; cat ok.log; exit 1; }
        if grep -qF "$entry" worktodo.txt; then echo "FAIL($name): entry still pending"; exit 1; fi
        [ ! -e "$state" ] || { echo "FAIL($name): $state left after the entry was retired"; exit 1; }
        echo "$name: RETIRED_THEN_CLEANED=PASS"
    ) || rc=$?
    rm -rf "$work"
    return "$rc"
}

fail=0
check_case llunsafe "Test=$P,70,1" "llunsafe_m_$P.ckpt" -engine-marin || fail=1
check_case wagstaff "PRP=1,2,$((2 * Q)),-1" "wagstaff_m_$((2 * Q)).ckpt" -wagstaff -engine-marin || fail=1
check_case legacy-wagstaff "PRP=1,2,$((2 * Q)),-1" "$((2 * Q))prp_wagstaff.loop" -wagstaff -marin || fail=1
exit "$fail"
