#!/usr/bin/env bash
# Legacy (-marin) checkpoint consistency on a device.
#
# The legacy state is a set of files (.mers residue, .loop iteration, ...). A kill between writing the
# .mers and the .loop of a save used to leave the .mers of one save with the .loop of the previous one,
# and a resume continued from that mixed state and gave a wrong result without a word. Here such a set
# is built from two real saves (the .loop and the rest of save 1, the .mers of the later save 2) and
# resumed: the result must be the one of an uninterrupted run, and the mismatch must be reported.
# PRP runs with Gerbicz-Li checking off (-gerbiczli toggles it), which would otherwise repair the
# residue by a rollback; legacy LL (-llunsafe) has no such check.
# usage: run_legacy_checkpoint_mismatch.sh <device> [prime p] [composite p]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
PPRIME="${2:-21701}"
PCOMP="${3:-21713}"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# verdict <dir>: status and res64 of the last result
verdict() { { grep -oE '"(status|res64)":"[^"]*"' "$1/results.txt" 2>/dev/null || true; } | tail -n 2 | tr '\n' ' '; }

fail=0
# check_case <name> <p> <mode: state file name part> <args...>
check_case() {
    local name="$1" p="$2" mode="$3"; shift 3
    local w="$WORK/$name" base="${p}${mode}" want got loop1 loop2 d
    mkdir -p "$w"
    run() { local dir="$1"; shift; (cd "$dir" && $RUN "$@" "$BIN" "$p" -marin -proof 0 "${ARGS[@]}" -d "$DEVICE" -noask -f "$dir" > "$dir/run.log" 2>&1) || true; }
    ARGS=("$@")
    fresh() { rm -rf "$1"; mkdir -p "$1"; ln -s "$ROOT/kernels" "$1/kernels"; }

    # The result of an uninterrupted run.
    fresh "$w/base"; run "$w/base" timeout 900
    want="$(verdict "$w/base")"
    [ -n "$want" ] || { echo "FAIL($name): no baseline result"; tail -n 5 "$w/base/run.log"; fail=1; return; }

    # Save 1: an interrupt part way.
    loop1=""
    for d in 1.5 2 2.5 3 4; do
        fresh "$w/s1"; run "$w/s1" timeout -s INT "$d"
        if [ -f "$w/s1/$base.loop" ] && [ "$(cat "$w/s1/$base.loop")" -gt 1 ]; then loop1="$(cat "$w/s1/$base.loop")"; break; fi
    done
    [ -n "$loop1" ] || { echo "FAIL($name): no first save"; tail -n 5 "$w/s1/run.log"; fail=1; return; }

    # Save 2: resume from save 1 and interrupt again further on.
    loop2=""
    for d in 2 2.5 3 4 5; do
        fresh "$w/s2"; cp -p "$w/s1/$base".* "$w/s2/"; run "$w/s2" timeout -s INT "$d"
        if [ -f "$w/s2/$base.loop" ] && [ "$(cat "$w/s2/$base.loop")" -gt "$loop1" ]; then loop2="$(cat "$w/s2/$base.loop")"; break; fi
    done
    [ -n "$loop2" ] || { echo "FAIL($name): no second save past $loop1"; tail -n 5 "$w/s2/run.log"; fail=1; return; }

    # Save 1 with the .mers of save 2: what a kill between the two files of save 2 used to leave.
    fresh "$w/mix"; cp -p "$w/s1/$base".* "$w/mix/"; cp -p "$w/s2/$base.mers" "$w/mix/$base.mers"
    run "$w/mix" timeout 900
    got="$(verdict "$w/mix")"
    if [ "$got" != "$want" ]; then
        echo "FAIL($name): resume from the .mers of iteration $loop2 with the .loop of iteration $loop1 gave $got, an uninterrupted run $want"
        fail=1; return
    fi
    if ! grep -qE 'saved state .* is not usable|no complete saved state' "$w/mix/run.log"; then
        echo "FAIL($name): the mismatched state files were used without a warning"
        fail=1; return
    fi
    echo "$name: loops $loop1/$loop2, result $got: MISMATCH_REFUSED=PASS"
}

check_case prp-prime "$PPRIME" prp -prp -gerbiczli
check_case prp-composite "$PCOMP" prp -prp -gerbiczli
check_case ll-prime "$PPRIME" ll -llunsafe -allow-unvalidated-legacy-ll
check_case ll-composite "$PCOMP" ll -llunsafe -allow-unvalidated-legacy-ll
[ "$fail" = 0 ] && echo "LEGACY_CHECKPOINT_MISMATCH=PASS"
exit "$fail"
