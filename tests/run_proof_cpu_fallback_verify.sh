#!/usr/bin/env bash
# The CPU fallback proof is verified before it is reported. PRMERS_TEST_FORCE_CPU_PROOF makes the
# proof with the CPU fallback (as when the GPU proof backend is unavailable):
#   - by default the proof is verified on the CPU, kept and reported;
#   - with -noverify it is kept and reported without the check, as on the GPU path.
# usage: run_proof_cpu_fallback_verify.sh <device> [p]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
P="${2:-11213}"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0
check() {
    local name="$1"; shift
    local w="$WORK/$name"
    mkdir -p "$w"; ln -s "$ROOT/kernels" "$w/kernels"
    (cd "$w" && PRMERS_TEST_FORCE_CPU_PROOF=1 $RUN timeout 900 "$BIN" "$P" -prp -proof 3 "$@" -d "$DEVICE" -noask -f "$w" > run.log 2>&1) || true
    grep -q 'falling back to CPU GMP' "$w/run.log" || { echo "FAIL($name): the CPU fallback did not run"; tail -n 20 "$w/run.log"; fail=1; return; }
    ls "$w/proof/$P-3.proof" > /dev/null 2>&1 || { echo "FAIL($name): no proof file"; ls -R "$w"; fail=1; return; }
    grep -q '"proof":{' "$w/results.txt" || { echo "FAIL($name): proof not reported in results.txt"; fail=1; return; }
    if [ "$name" = verify ]; then
        grep -q 'CPU proof verification: SUCCESS' "$w/run.log" || { echo "FAIL($name): the CPU proof was not verified"; grep -i 'verif' "$w/run.log"; fail=1; return; }
    else
        if grep -q 'CPU proof verification' "$w/run.log"; then echo "FAIL($name): verified despite -noverify"; fail=1; return; fi
    fi
    echo "$name: CPU_FALLBACK_PROOF=PASS"
}

check verify
check noverify -noverify
[ "$fail" = 0 ] && echo "PROOF_CPU_FALLBACK_VERIFY=PASS"
exit "$fail"
