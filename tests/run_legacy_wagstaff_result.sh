#!/usr/bin/env bash
# A -wagstaff run on the legacy internal-NTT backend (-marin) must record its verdict and retire the
# worktodo entry, like the Marin/Aevum path does. It used to print the verdict and return, leaving no
# result and the entry in the worktodo (so the next start ran it again).
#
# Q = 127: (2^127 + 1)/3 is a Wagstaff prime, so the run is a PRP of 2^254 - 1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
Q="${2:-127}"
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# 1. Command line exponent.
set +e
$RUN timeout 120 "$ROOT/prmers" "$Q" -wagstaff -marin -d "$DEVICE" -noask -f "$WORK" > cli.log 2>&1
rc=$?
set -e
echo "CLI_RC=$rc"
grep -q 'Wagstaff PRP confirmed' cli.log
[ -s "${Q}_wagstaff_result.json" ] || { echo "FAIL: no individual Wagstaff result file"; exit 1; }
grep -q '"worktype":"Wagstaff-PRP"' results.txt
grep -q '"status":"P"' results.txt
echo "LEGACY_WAGSTAFF_CLI=PASS"

# 2. Worktodo entry (it holds the exponent of the PRP, 2 * Q): retired after the verdict.
rm -f results.txt "${Q}_wagstaff_result.json" worktodo_save.txt
printf 'PRP=1,2,%s,-1\n' "$((2 * Q))" > worktodo.txt
set +e
$RUN timeout 120 "$ROOT/prmers" -wagstaff -marin -d "$DEVICE" -noask -f "$WORK" -worktodo worktodo.txt > wt.log 2>&1
rc=$?
set -e
echo "WORKTODO_RC=$rc"
grep -q 'Wagstaff PRP confirmed' wt.log
grep -q "PRP=1,2,$((2 * Q)),-1" worktodo_save.txt || { echo "FAIL: entry not archived"; cat wt.log; exit 1; }
if grep -q 'PRP=' worktodo.txt; then
    echo "FAIL: entry still in worktodo.txt"
    exit 1
fi
grep -q '"worktype":"Wagstaff-PRP"' results.txt
echo "LEGACY_WAGSTAFF_WORKTODO=PASS"
