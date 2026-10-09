#!/usr/bin/env bash
# How the CPU fallback proof is verified. PRMERS_TEST_FORCE_CPU_PROOF makes the proof with the CPU
# fallback (as when the GPU proof backend is unavailable at that point), and
# PRMERS_TEST_GPU_VERIFY_UNUSABLE makes the GPU unusable for verifying it:
#   - gpu:      verified with the GPU, as the normal proof path verifies it; no CPU verification;
#   - cpu:      GPU unusable and the CPU check cheap: verified on the CPU;
#   - skip:     GPU unusable and the CPU check over the cap (PRMERS_CPU_PROOF_VERIFY_MAX_SECONDS=0):
#               not verified, with a warning; the proof is kept and reported;
#   - noverify: -noverify: kept and reported without any check, as on the GPU path.
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
# check <name> <env assignments, space separated or "-"> [prmers options...]
check() {
    local name="$1" envs="$2"; shift 2
    local w="$WORK/$name"
    local -a extra=()
    [ "$envs" = "-" ] || read -r -a extra <<< "$envs"
    mkdir -p "$w"; ln -s "$ROOT/kernels" "$w/kernels"
    (cd "$w" && env PRMERS_TEST_FORCE_CPU_PROOF=1 "${extra[@]}" $RUN timeout 900 "$BIN" "$P" -prp -proof 3 "$@" -d "$DEVICE" -noask -f "$w" > run.log 2>&1) || true
    local log="$w/run.log"
    grep -q 'falling back to CPU GMP' "$log" || { echo "FAIL($name): the CPU fallback did not run"; tail -n 20 "$log"; fail=1; return; }
    ls "$w/proof/$P-3.proof" > /dev/null 2>&1 || { echo "FAIL($name): no proof file"; ls -R "$w"; fail=1; return; }
    grep -q '"proof":{' "$w/results.txt" || { echo "FAIL($name): proof not reported in results.txt"; fail=1; return; }
    case "$name" in
    gpu)
        grep -q 'Verification result: SUCCESS' "$log" || { echo "FAIL($name): the CPU proof was not verified with the GPU"; grep -i 'verif' "$log"; fail=1; return; }
        if grep -q 'CPU proof verification' "$log"; then echo "FAIL($name): verified on the CPU although the GPU works"; fail=1; return; fi ;;
    cpu)
        grep -q 'CPU proof verification: SUCCESS' "$log" || { echo "FAIL($name): the CPU proof was not verified on the CPU"; grep -i 'verif' "$log"; fail=1; return; }
        if grep -q 'Verification result' "$log"; then echo "FAIL($name): the GPU verified although it is unusable"; fail=1; return; fi ;;
    skip)
        grep -q 'CPU fallback proof for M'"$P"' was not verified' "$log" || { echo "FAIL($name): no warning that the proof was not verified"; grep -i 'verif' "$log"; fail=1; return; }
        if grep -q 'CPU proof verification\|Verification result' "$log"; then echo "FAIL($name): verified although it should be skipped"; fail=1; return; fi ;;
    noverify)
        if grep -q 'CPU proof verification\|Verification result\|was not verified' "$log"; then echo "FAIL($name): verification or warning despite -noverify"; fail=1; return; fi ;;
    esac
    echo "$name: CPU_FALLBACK_PROOF=PASS"
}

check gpu -
check cpu PRMERS_TEST_GPU_VERIFY_UNUSABLE=1
check skip "PRMERS_TEST_GPU_VERIFY_UNUSABLE=1 PRMERS_CPU_PROOF_VERIFY_MAX_SECONDS=0"
check noverify - -noverify
[ "$fail" = 0 ] && echo "PROOF_CPU_FALLBACK_VERIFY=PASS"
exit "$fail"
