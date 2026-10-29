#!/usr/bin/env bash
# Legacy (-marin) PRP: a clFinish that fails in the IPS benchmark (the sample before the test, and -tune) must stop
# the run, never be timed as if the GPU had done the work.
#
# A preloaded shim (tests/legacy_cl_read_fault_shim.c) makes one clFinish return CL_INVALID_COMMAND_QUEUE, what a
# reset or lost GPU returns.  Before the fix the status was ignored: the benchmark timed a GPU that had stopped
# (far too fast), printed that rate, and the run went on (with -tune, the iterforce search was sized from it).
# Each case must now exit with an error that says the GPU was reset or lost, and write no result.
#
# The legacy PRP calls clFinish first in the benchmark: with the default iterforce (500) the 100-iteration sample
# waits once, at its end; with -iterforce 10 it also waits every 10 iterations.  -tune then samples 64 iterations
# at iterforce 10 (6 waits in the loop, one at the end).
# usage: run_ips_finish_fault.sh <device> [prime p]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
P="${2:-9941}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra "$ROOT/tests/legacy_cl_read_fault_shim.c" -o "$WORK/shim.so" -ldl

fail=0
# run <dir> <prmers options> [VAR=value ...]: one legacy PRP of P in <dir>, the shim preloaded with the given settings
run() {
    local dir="$1" opts="$2"; shift 2
    local -a extra
    read -r -a extra <<< "$opts"
    rm -rf "$dir"; mkdir -p "$dir"; ln -s "$ROOT/kernels" "$dir/kernels"
    (cd "$dir" && env LD_PRELOAD="$WORK/shim.so${PRMERS_TEST_PRELOAD:+:$PRMERS_TEST_PRELOAD}" "$@" \
        timeout 900 "$BIN" "$P" "${extra[@]}" -marin -proof 0 -d "$DEVICE" -noask -f "$dir" > "$dir/run.log" 2>&1) \
        && echo 0 || echo $?
}

# The run with nothing failing: a prime result.
rc="$(run "$WORK/base" "")"
if [ "$rc" != 0 ] || ! grep -q '"status":"P"' "$WORK/base/results.txt" 2>/dev/null; then
    echo "FAIL(base): M$P without a fault did not give a prime result (exit $rc)"; tail -n 5 "$WORK/base/run.log"; exit 1
fi

# check <name> <prmers options> <nth> [code] [expected message]: fail that clFinish, expect a stop with the message
check() {
    local name="$1" opts="$2" nth="$3" code="${4:--36}" want="${5:-GPU reset or lost}" dir="$WORK/$1" rc
    rc="$(run "$dir" "$opts" PRMERS_FAULT_FINISH_NTH="$nth" PRMERS_FAULT_READ_CODE="$code")"
    if ! grep -q 'failing clFinish' "$dir/run.log"; then
        echo "FAIL($name): the clFinish to fail was not reached"; fail=1; return
    fi
    if [ "$rc" = 0 ]; then echo "FAIL($name): the run exited 0 after a failed clFinish"; fail=1; fi
    if ! grep -qF "$want" "$dir/run.log"; then
        echo "FAIL($name): no '$want' message"; tail -n 5 "$dir/run.log"; fail=1
    fi
    if sed -n '/failing clFinish/,$p' "$dir/run.log" | grep -qE 'Estimated IPS|Optimal iterforce'; then
        echo "FAIL($name): a rate was reported from the failed benchmark"; fail=1
    fi
    if [ -s "$dir/results.txt" ] || ls "$dir"/*_result.json > /dev/null 2>&1; then
        echo "FAIL($name): a result was written: $(head -c 200 "$dir/results.txt" 2>/dev/null)"; fail=1
    fi
    echo "case $name: exit $rc"
}

# The wait at the end of the sample, and one in its loop.
check sample-end "" 1
check sample-loop "-iterforce 10" 5
check sample-last "-iterforce 10" 11
# A code that does not mean a lost GPU (-14, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) stops the run too.
check sample-other-error "" 1 -14 "clFinish (IPS benchmark) failed: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST (-14)"
# -tune: the first wait of its own sample, and the one at the end of it.
check tune-sample-loop "-tune" 2
check tune-sample-end "-tune" 8

if [ "$fail" -ne 0 ]; then echo "IPS benchmark clFinish fault regression: FAIL"; exit 1; fi
echo "IPS benchmark clFinish fault regression: PASS"
