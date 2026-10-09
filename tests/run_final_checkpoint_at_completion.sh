#!/usr/bin/env bash
# When a test completes but its saved state is kept (the worktodo entry could not be retired, or the
# result could not be saved), the rerun must resume at the END of the test and only redo the
# bookkeeping, not repeat the iterations since the last periodic backup. So every mode writes a final
# checkpoint at completion, before the result is saved.
#
# Each case runs to completion with the periodic backup effectively off (-t 100000), so the only
# state that can exist after the run is the final one. The entry cannot be retired (worktodo_save.txt
# is a directory) or, for the modes that do not use the worktodo, the result cannot be saved
# (results.txt is a directory). The rerun must then:
#   - resume at the final iteration (and run no iteration: with -t 0 any iteration would back up),
#   - report the same verdict and residue as the first run,
#   - retire the entry / save the result, and remove all of the state.
# usage: run_final_checkpoint_at_completion.sh <device> [p] [wagstaff-q]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
P="${2:-4423}"
Q="${3:-5807}"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# verdict <file>: the verdict and residue of the last result in a results file or a log.
verdict() {
    grep -oE '"(status|res64)":"[^"]*"|is (prime|composite|a probable prime)|Wagstaff PRP confirmed|Not a Wagstaff PRP' "$1" | tail -n 3 | tr '\n' ' '
}

# check_case <name> <kind> <worktodo line or -> <total iterations> <state glob> <prmers args...>
# kind: retire (the worktodo entry cannot be retired) or save (the result cannot be saved).
check_case() {
    local name="$1" kind="$2" entry="$3" total="$4" state="$5"; shift 5
    local work rc=0
    work="$(mktemp -d)"
    (
        cd "$work"
        ln -s "$ROOT/kernels" kernels
        if [ "$entry" = "-" ]; then cmd=("$BIN" "$P"); else echo "$entry" > worktodo.txt; cmd=("$BIN" -worktodo worktodo.txt); fi
        if [ "$kind" = retire ]; then mkdir worktodo_save.txt; else mkdir results.txt; fi

        # 1. Run to completion; the final state must be kept.
        $RUN timeout 900 "${cmd[@]}" -t 100000 "$@" -d "$DEVICE" -noask -f "$work" > first.log 2>&1 || true
        grep -qE 'Failed to update worktodo.txt; keeping|Result could not be saved; keeping|Result persistence failed; keeping' first.log ||
            { echo "FAIL($name): the first run did not keep its state"; tail -n 20 first.log; exit 1; }
        # shellcheck disable=SC2086
        ls $state > /dev/null 2>&1 || { echo "FAIL($name): no state kept after completion ($state)"; ls; exit 1; }
        first="$(verdict first.log)"
        loopf="$(ls $state 2>/dev/null | grep '\.loop$' || true)"
        if [ -n "$loopf" ]; then
            # The final state records the end, and a rerun that is again unable to finish keeps it
            # there instead of moving one past it.
            [ "$(cat "$loopf")" = "$total" ] || { echo "FAIL($name): $loopf holds $(cat "$loopf"), not $total"; exit 1; }
            $RUN timeout 900 "${cmd[@]}" -t 0 "$@" -d "$DEVICE" -noask -f "$work" > again.log 2>&1 || true
            [ "$(cat "$loopf")" = "$total" ] || { echo "FAIL($name): $loopf holds $(cat "$loopf") after a rerun, not $total"; exit 1; }
        fi
        if [ "$kind" = retire ]; then
            [ -s results.txt ] || { echo "FAIL($name): result not saved"; exit 1; }
            grep -qF "$entry" worktodo.txt || { echo "FAIL($name): entry retired"; exit 1; }
            rmdir worktodo_save.txt
        else
            rmdir results.txt
        fi
        echo "$name: FINAL_STATE_KEPT=PASS"

        # 2. Rerun: resume at the end, run no iteration, same result, clean up.
        $RUN timeout 900 "${cmd[@]}" -t 0 "$@" -d "$DEVICE" -noask -f "$work" > second.log 2>&1 || true
        if grep -q 'Resuming from iteration' second.log; then
            grep -qE "Resuming from iteration $total( |$)" second.log ||
                { echo "FAIL($name): did not resume at the final iteration $total"; grep 'Resuming from' second.log; exit 1; }
            # Legacy backend: with -t 0 an iteration would write the loop file; at most the final one.
            n="$(grep -c 'Loop iteration saved' second.log || true)"
            [ "$n" -le 1 ] || { echo "FAIL($name): $n loop saves on the rerun, an iteration ran"; exit 1; }
        else
            grep -q 'Resuming from a checkpoint' second.log ||
                { echo "FAIL($name): the rerun did not resume"; tail -n 20 second.log; exit 1; }
            if grep -q 'Backup point done' second.log; then
                echo "FAIL($name): the rerun ran iterations after resuming"; grep 'Backup point done' second.log | head -n 3; exit 1
            fi
        fi
        echo "$name: RESUMED_AT_END=PASS"
        second="$(verdict second.log)"
        [ "$second" = "$first" ] || { echo "FAIL($name): result differs: '$first' vs '$second'"; exit 1; }
        if [ "$kind" = retire ]; then
            grep -q 'Entry removed from worktodo.txt' second.log || { echo "FAIL($name): entry not retired"; tail -n 20 second.log; exit 1; }
            if grep -qF "$entry" worktodo.txt; then echo "FAIL($name): entry still pending"; exit 1; fi
        else
            [ -s results.txt ] || { echo "FAIL($name): result not saved on the rerun"; exit 1; }
        fi
        # shellcheck disable=SC2086
        if ls $state > /dev/null 2>&1; then echo "FAIL($name): state left after completion"; ls; exit 1; fi
        # A proof file is the product of a PRP with a proof; its residues must be gone.
        if [ -d proof-tmp ] && [ -n "$(ls -A proof-tmp)" ]; then echo "FAIL($name): proof residues left"; exit 1; fi
        case "$name" in *-proof) ls proof/*.proof > /dev/null 2>&1 || { echo "FAIL($name): no proof file"; exit 1; } ;; esac
        left="$(ls | grep -vE '^(proof|proof-tmp|kernels|worktodo(_save)?\.txt|results\.txt|prmers\.log|first\.log|again\.log|second\.log|.*\.json)$' || true)"
        [ -z "$left" ] || { echo "FAIL($name): leftovers: $left"; exit 1; }
        echo "$name: SAME_RESULT_AND_CLEANED=PASS"
    ) || rc=$?
    rm -rf "$work"
    return "$rc"
}

fail=0
# A -wagstaff worktodo entry holds q, like the command line; the state is named after 2q.
W=$((2 * Q))
check_case marin-prp           retire "PRP=1,2,$P,-1"  "$P"         "m_$P.ckpt*"           -prp -proof 0 -engine-marin || fail=1
check_case marin-llunsafe      retire "Test=$P,70,1"   "$((P - 2))" "llunsafe_m_$P.ckpt*"  -engine-marin || fail=1
check_case marin-wagstaff      retire "PRP=1,2,$Q,-1"  "$Q"         "wagstaff_m_$W.ckpt*"  -wagstaff -engine-marin || fail=1
check_case legacy-prp          retire "PRP=1,2,$P,-1"  "$P"         "${P}prp.*"            -prp -proof 0 -marin || fail=1
check_case legacy-prp-proof    retire "PRP=1,2,$P,-1"  "$P"         "${P}prp.*"            -prp -proof 2 -marin || fail=1
check_case legacy-ll           retire "Test=$P,70,1"   "$((P - 2))" "${P}ll.*"             -marin -allow-unvalidated-legacy-ll || fail=1
check_case legacy-wagstaff     retire "PRP=1,2,$Q,-1"  "$Q"         "${W}prp_wagstaff.*"   -wagstaff -marin || fail=1
check_case llsafe2             save   "-"              "$((P - 2))" "llsafe2_m_$P.ckpt*"   -llsafe2 -engine-marin || fail=1
check_case llsafe              save   "-"              "$((P - 1))" "llsafe_m_$P.ckpt*"    -ll -engine-marin || fail=1
check_case llsafe-worktodo     retire "DoubleCheck=$P,70,1" "$((P - 1))" "llsafe_m_$P.ckpt*"    || fail=1
exit "$fail"
