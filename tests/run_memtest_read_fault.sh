#!/usr/bin/env bash
# -memtest: a GPU read or wait that fails must stop the test with an error, never let it report a pass.
#
# A preloaded shim (tests/memtest_cl_fault_shim.c) makes one clEnqueueReadBuffer or clFinish return
# CL_INVALID_COMMAND_QUEUE, what a reset or lost GPU returns.  Before the fix the status was ignored: a failed error
# count read left the count at 0, so the test reported "Errors 0" and exited 0 for memory it never checked, and a
# failed kernel or wait was not noticed at all.  Each case must now exit with an error that names the failed call,
# and the report must say the memory was not fully tested.
#
# The shim also lets -memtest use PoCL's CPU device (it looks for GPU-type devices itself) and reports a 40 MB
# device, so the test covers one 8 MB buffer.  The calls of a run with nothing failing are counted first, so the
# failing call is picked by position: the 4-byte reads are the error counts (address read, the inversion
# read-writes, the inversion read, the modulo reads, in that order), the 512-byte reads are the error report.
# usage: run_memtest_read_fault.sh <device>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra "$ROOT/tests/memtest_cl_fault_shim.c" -o "$WORK/shim.so" -ldl

fail=0
# run <dir> [VAR=value ...]: one memtest in <dir>, the shim preloaded with the given settings
run() {
    local dir="$1"; shift
    rm -rf "$dir"; mkdir -p "$dir"
    (cd "$dir" && env LD_PRELOAD="$WORK/shim.so${PRMERS_TEST_PRELOAD:+:$PRMERS_TEST_PRELOAD}" \
        PRMERS_MEMTEST_ANY_DEVICE=1 PRMERS_MEMTEST_VRAM_MB=40 "$@" \
        timeout 900 "$BIN" -memtest -d "$DEVICE" -noask > "$dir/run.log" 2>&1) && echo 0 || echo $?
}

# The run with nothing failing: a pass, and the call counts.
rc="$(run "$WORK/base" PRMERS_FAULT_LOG="$WORK/base.calls")"
if [ "$rc" != 0 ] || ! grep -q 'Totals:  Errors 0 ' "$WORK/base/run.log"; then
    echo "FAIL(base): memtest without a fault did not pass (exit $rc)"; tail -n 20 "$WORK/base/run.log"; exit 1
fi
ncount="$(grep -cx 'R 4' "$WORK/base.calls" || true)"
nfinish="$(grep -cx 'F' "$WORK/base.calls" || true)"
# one buffer: 1 address + 128 inversion read-write + 1 inversion + 20 modulo error counts
if [ "$ncount" != 150 ] || [ "$nfinish" -lt 150 ]; then
    echo "FAIL(base): unexpected calls: $ncount error count reads, $nfinish clFinish"; exit 1
fi

# check <name> <expected message> [VAR=value ...]: expect a stop with the message, a failing exit, and no pass
check() {
    local name="$1" want="$2" dir="$WORK/$1" rc
    shift 2
    rc="$(run "$dir" "$@")"
    if ! grep -q '\[fault shim\] failing' "$dir/run.log"; then
        echo "FAIL($name): the call to fail was not reached"; fail=1; return
    fi
    if [ "$rc" = 0 ]; then echo "FAIL($name): the memtest exited 0 after a failed call"; fail=1; fi
    if [ "$rc" = 1 ]; then echo "FAIL($name): exit 1 is the interrupted code"; fail=1; fi
    if ! grep -qF "$want" "$dir/run.log"; then
        echo "FAIL($name): no '$want' message"; tail -n 8 "$dir/run.log"; fail=1
    fi
    if ! grep -q 'Status: FAILED, stopped by an OpenCL error' "$dir/run.log"; then
        echo "FAIL($name): the report does not say the test failed"; tail -n 8 "$dir/run.log"; fail=1
    fi
    echo "case $name: exit $rc"
}

lost="GPU reset or lost"
# The error count reads of each pattern.
check count-address "$lost" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=1
check count-invert-rw-first "$lost" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=2
check count-invert-rw-last "$lost" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=129
check count-invert "$lost" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=130
check count-modulo-first "$lost" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=131
check count-modulo-last "$lost" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=150
# A code that does not mean a lost GPU (-14, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) stops the test too.
check count-other-error "clEnqueueReadBuffer (error count) failed" PRMERS_FAULT_READ_BYTES=4 PRMERS_FAULT_READ_NTH=1 \
    PRMERS_FAULT_CODE=-14
# The error report reads (address, expected, current, second read), after a count that says an error was found.
for nth in 1 2 3 4; do
    check "report-address-$nth" "$lost" PRMERS_FAKE_COUNT_NTH=1 PRMERS_FAULT_READ_BYTES=512 PRMERS_FAULT_READ_NTH="$nth"
done
check report-invert-rw "$lost" PRMERS_FAKE_COUNT_NTH=2 PRMERS_FAULT_READ_BYTES=512 PRMERS_FAULT_READ_NTH=1
check report-invert "$lost" PRMERS_FAKE_COUNT_NTH=130 PRMERS_FAULT_READ_BYTES=512 PRMERS_FAULT_READ_NTH=4
check report-modulo "$lost" PRMERS_FAKE_COUNT_NTH=150 PRMERS_FAULT_READ_BYTES=512 PRMERS_FAULT_READ_NTH=2
# The waits after the kernels: the address write and read, the first inversion read-write, the last modulo read.
check finish-address-write "$lost" PRMERS_FAULT_FINISH_NTH=1
check finish-address-read "$lost" PRMERS_FAULT_FINISH_NTH=2
check finish-invert-rw "$lost" PRMERS_FAULT_FINISH_NTH=4
check finish-last "$lost" PRMERS_FAULT_FINISH_NTH="$nfinish"
check finish-other-error "failed: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" PRMERS_FAULT_FINISH_NTH=3 \
    PRMERS_FAULT_CODE=-14
# The error count resets (the first one, and the one after a pass that found errors), and the first sector.
check write-first "$lost" PRMERS_FAULT_WRITE_NTH=1
check write-after-errors "$lost" PRMERS_FAKE_COUNT_NTH=2 PRMERS_FAULT_WRITE_NTH=3
check subbuffer "Out of GPU memory" PRMERS_FAULT_SUBBUF_NTH=1 PRMERS_FAULT_CODE=-4
check subbuffer-other-error "clCreateSubBuffer failed: CL_INVALID_VALUE (-30)" PRMERS_FAULT_SUBBUF_NTH=1 PRMERS_FAULT_CODE=-30

if [ "$fail" -ne 0 ]; then echo "Memtest read fault regression: FAIL"; exit 1; fi
echo "Memtest read fault regression: PASS"
