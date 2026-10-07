#!/usr/bin/env bash
# LL-SAFE with a Gerbicz-Li check at iteration 0 (p = 227: p - 2 is a multiple of
# floor(sqrt(p)) and -checklevel 1 checks at every block boundary). An error injected
# at iteration 4 must roll back to the state after iteration 0, not repeat iteration 0,
# so the run recovers and ends with the same residue as an error-free run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P=227
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# M227 is composite, so the exit status is 1 and is not checked.
set +e
$RUN timeout 50 "$ROOT/prmers" "$P" -ll -engine-marin -checklevel 1 -d "$DEVICE" -noask -f "$WORK" > clean.log 2>&1
set -e
clean_res="$(sed -n 's/.*res64=0x\([0-9A-Fa-f]*\).*/\1/p' clean.log | head -n 1)"
[ -n "$clean_res" ] || { echo "FAIL: no residue in error-free run"; cat clean.log; exit 1; }

set +e
$RUN timeout 50 "$ROOT/prmers" "$P" -ll -engine-marin -checklevel 1 -erroriter 4 -d "$DEVICE" -noask -f "$WORK" > injected.log 2>&1
rc=$?
set -e
echo "INJECTED_RC=$rc"
grep -q 'Injected error at iteration 4' injected.log
grep -q 'Check FAILED' injected.log
[ "$rc" -ne 124 ] || { echo "FAIL: run did not terminate after the injected error"; exit 1; }
inj_res="$(sed -n 's/.*res64=0x\([0-9A-Fa-f]*\).*/\1/p' injected.log | head -n 1)"
[ "$inj_res" = "$clean_res" ] || { echo "FAIL: residue $inj_res != $clean_res after recovery"; exit 1; }

echo "LLSAFE_ERROR_RECOVERY=PASS"
