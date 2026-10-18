#!/usr/bin/env bash
# Legacy (-marin) PRP: a GPU read that fails must stop the run, never be taken as the GPU's answer.
#
# A preloaded shim (tests/legacy_cl_read_fault_shim.c) makes one clEnqueueReadBuffer return
# CL_INVALID_COMMAND_QUEUE, what a reset or lost GPU returns.  Before the fix the status was ignored: a failed
# Gerbicz-Li flag read printed "Check passed!" and saved the state as verified, a failed final residue read
# reported the prime M21701 as composite with res64 0000000000000000, and a failed checkpoint read saved zeros.
# Legacy LL (opt-in) reported a composite as prime when its result read failed.
# Each case must now exit with an error that says the GPU was reset or lost, and write no result.
#
# The reads of a short run are counted first (with the shim logging only), so the failing read is picked by
# position: the one 4-byte read is the Gerbicz-Li flag of the final check; of the full residue reads, the first
# is the final residue, the last is the result, and the ones in between are the final checkpoint save.
# usage: run_legacy_read_fault.sh <device> [prime p] [composite p]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
P="${2:-21701}"
PCOMP="${3:-21713}"
EXP="$P"
MODE=()

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra "$ROOT/tests/legacy_cl_read_fault_shim.c" -o "$WORK/shim.so" -ldl

fail=0
# run <dir> [VAR=value ...]: one legacy PRP of P in <dir>, the shim preloaded with the given settings
run() {
    local dir="$1"; shift
    rm -rf "$dir"; mkdir -p "$dir"; ln -s "$ROOT/kernels" "$dir/kernels"
    (cd "$dir" && env LD_PRELOAD="$WORK/shim.so${PRMERS_TEST_PRELOAD:+:$PRMERS_TEST_PRELOAD}" "$@" \
        timeout 900 "$BIN" "$EXP" "${MODE[@]}" -marin -proof 0 -d "$DEVICE" -noask -f "$dir" > "$dir/run.log" 2>&1) && echo 0 || echo $?
}

# The run with nothing failing: a result, and the read counts.
rc="$(run "$WORK/base" PRMERS_FAULT_READ_LOG="$WORK/base.reads")"
if [ "$rc" != 0 ] || ! grep -q '"status":"P"' "$WORK/base/results.txt" 2>/dev/null; then
    echo "FAIL(base): M$P without a fault did not give a prime result (exit $rc)"; tail -n 5 "$WORK/base/run.log"; exit 1
fi
full="$(sort -n "$WORK/base.reads" | tail -n 1)"
nfull="$(grep -cx "$full" "$WORK/base.reads")"
nflag="$(grep -cx 4 "$WORK/base.reads" || true)"
if [ "$nflag" -lt 1 ] || [ "$nfull" -lt 3 ]; then
    echo "FAIL(base): unexpected reads: $nflag flag reads, $nfull residue reads of $full bytes"; exit 1
fi

# check <name> <bytes> <nth> [code] [expected message]: fail that read, expect a stop with the message and no result
check() {
    local name="$1" bytes="$2" nth="$3" code="${4:--36}" want="${5:-GPU reset or lost}" dir="$WORK/$1" rc
    rc="$(run "$dir" PRMERS_FAULT_READ_BYTES="$bytes" PRMERS_FAULT_READ_NTH="$nth" PRMERS_FAULT_READ_CODE="$code")"
    if ! grep -q 'failing clEnqueueReadBuffer' "$dir/run.log"; then
        echo "FAIL($name): the read to fail was not reached"; fail=1; return
    fi
    if [ "$rc" = 0 ]; then echo "FAIL($name): the run exited 0 after a failed read"; fail=1; fi
    if ! grep -qF "$want" "$dir/run.log"; then
        echo "FAIL($name): no '$want' message"; tail -n 5 "$dir/run.log"; fail=1
    fi
    if sed -n '/failing clEnqueueReadBuffer/,$p' "$dir/run.log" | grep -q 'Check passed'; then
        echo "FAIL($name): the Gerbicz-Li check passed on a failed read"; fail=1
    fi
    if [ -s "$dir/results.txt" ] || ls "$dir"/*_result.json > /dev/null 2>&1; then
        echo "FAIL($name): a result was written: $(head -c 200 "$dir/results.txt" 2>/dev/null)"; fail=1
    fi
    echo "case $name: exit $rc"
}

check gerbicz-flag 4 "$nflag"
check final-residue "$full" 1
check checkpoint "$full" 2
check result "$full" "$nfull"
# A code that does not mean a lost GPU (-14, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) stops the run too.
check flag-other-error 4 "$nflag" -14 "clEnqueueReadBuffer (Gerbicz-Li flag) failed"
check result-other-error "$full" "$nfull" -14 "clEnqueueReadBuffer (result) failed"

# Legacy LL of a composite: its result read is the last full residue read.
EXP="$PCOMP"
MODE=(-ll -llunsafe -allow-unvalidated-legacy-ll)
rc="$(run "$WORK/llbase" PRMERS_FAULT_READ_LOG="$WORK/llbase.reads")"
# (a composite result exits 1)
if [ "$rc" != 1 ] || ! grep -q '"status":"C"' "$WORK/llbase/results.txt" 2>/dev/null; then
    echo "FAIL(llbase): legacy LL of M$PCOMP without a fault did not give a composite result (exit $rc)"
    tail -n 5 "$WORK/llbase/run.log"; exit 1
fi
llfull="$(sort -n "$WORK/llbase.reads" | tail -n 1)"
check ll-result "$llfull" "$(grep -cx "$llfull" "$WORK/llbase.reads")"

if [ "$fail" -ne 0 ]; then echo "Legacy read fault regression: FAIL"; exit 1; fi
echo "Legacy read fault regression: PASS"
