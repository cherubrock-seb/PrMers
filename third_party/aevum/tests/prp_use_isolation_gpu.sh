#!/usr/bin/env bash
# The PRP -use tuner must run its probes in an OpenCL context of its own, so a probe that faults cannot take the
# production engine down with it, and the fault must be reported as a failure, not as an ordinary rejection.
#
# A preloaded shim (tests/opencl_context_fault_shim.c) kills the first OpenCL context that compiles a kernel: its
# first kernel launch fails with CL_OUT_OF_RESOURCES and every later call on it fails too, as after a GPU fault or a
# driver reset of a hung context.  With shape autotune off and the -use tuner forced, the first kernels compiled are
# the tuner's.  Before the fix they ran on the production context, so the engine could not be created (or computed
# on a dead context) and the fault was logged as AEVUM_USE_REJECT.  Now the engine must give the same residue as an
# engine created with the tuner off, and the log must say AEVUM_USE_FAIL.
#
# usage: prp_use_isolation_gpu.sh [libaevum_engine.so] [device]     (OpenCL only; runs on PoCL)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="$(realpath "${1:-$ROOT/build-engine/libaevum_engine.so}")"
DEVICE="${2:-${AEVUM_TEST_DEVICE:-0}}"
EXPONENT=1362763
SQUARINGS=20

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CXX:-g++}" -O2 -std=c++20 "$ROOT/tests/engine_prp_residue_driver.cpp" -ldl -o "$WORK/driver"
"${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra "$ROOT/tests/opencl_context_fault_shim.c" -o "$WORK/shim.so" -ldl -lOpenCL

# run <name> [VAR=value ...]: one engine in its own directory and caches; prints the exit status
run() {
    local name="$1"; shift
    local dir="$WORK/$name"
    mkdir -p "$dir"
    (cd "$dir" && env AEVUM_AUTOTUNE=off AEVUM_AUTOTUNE_CACHE="$dir/cache/autotune.tsv" "$@" \
        timeout 1800 "$WORK/driver" "$LIB" "$DEVICE" "$EXPONENT" "" "$dir" "$SQUARINGS" > "$dir/run.log" 2>&1) \
        && echo 0 || echo $?
}

rc="$(run ref AEVUM_PRP_USE_TUNE=off)"
ref="$(grep -o 'RESIDUE [0-9a-f]*' "$WORK/ref/run.log" || true)"
if [ "$rc" != 0 ] || [ -z "$ref" ]; then
    echo "FAIL(ref): engine with the tuner off failed (exit $rc)"; tail -n 20 "$WORK/ref/run.log"; exit 1
fi

rc="$(run fault AEVUM_PRP_USE_TUNE=retune AEVUM_FAULT_BUILD_OPT=-DEXP= AEVUM_FAULT_LOG="$WORK/fault.calls" \
      LD_PRELOAD="$WORK/shim.so${AEVUM_TEST_PRELOAD:+:$AEVUM_TEST_PRELOAD}")"
log="$WORK/fault/run.log"
fail=0
if ! grep -q '^KILL$' "$WORK/fault.calls" 2>/dev/null; then
    echo "FAIL(fault): the shim never killed a context; the test did not inject anything"; fail=1
fi
if [ "$rc" != 0 ]; then
    echo "FAIL(fault): the production engine failed after a tuner probe faulted (exit $rc)"; fail=1
elif ! grep -q "^$ref\$" "$log"; then
    echo "FAIL(fault): residue differs from the reference ($ref): $(grep RESIDUE "$log" || true)"; fail=1
fi
if ! grep -q 'AEVUM_USE_FAIL ' "$log"; then
    echo "FAIL(fault): the faulting probe was not reported as AEVUM_USE_FAIL"; fail=1
fi
if grep -q 'AEVUM_USE_REJECT' "$log"; then
    echo "FAIL(fault): the fault was logged as an ordinary rejection"; fail=1
fi
if [ "$fail" != 0 ]; then
    grep -v 'saving binary\|Loaded ' "$log" | tail -n 30
    exit 1
fi
echo "prp_use_isolation_gpu: OK ($ref)"
