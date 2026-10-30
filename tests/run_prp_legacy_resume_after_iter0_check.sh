#!/usr/bin/env bash
# Legacy (-marin) PRP resumed from a checkpoint written after a Gerbicz-Li check at iteration 0.
# p = K*K + 1 makes p - 1 a multiple of the block size floor(sqrt(p)) = K, and -checklevel 1
# checks at every block boundary, so the first check is at iteration 0. A checkpoint saved in
# the first block holds the state after iteration 0 as its last verified state. An error injected
# at the end of the first block of the resumed run must roll back to the true start of the run:
# the run recovers and ends with the residue of an error-free run. Reading the checkpoint's
# verified iteration 0 as "state after iteration 0 is not verified yet" while rolling back to a
# state that already did iteration 0 repeats that iteration, every later check fails, and the
# run stops after three failed checks.
# Usage: run_prp_legacy_resume_after_iter0_check.sh [device]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
K=36
P=$((K * K + 1))   # 1297
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

ARGS=(-prp -marin -checklevel 1 -d "$DEVICE" -noask -f "$WORK")
prmers() { $RUN timeout --kill-after=10s 300 "$ROOT/prmers" "$P" "${ARGS[@]}" "$@"; }
residue() { sed -n 's/.*"res64":"\([0-9A-Fa-f]*\)".*/\1/p' "$1" | head -n 1; }

cd "$WORK"
ln -s "$ROOT/kernels" kernels

set +e
prmers > clean.log 2>&1
set -e
clean_res="$(residue clean.log)"
[ -n "$clean_res" ] || { echo "FAIL: no residue in error-free run"; cat clean.log; exit 1; }
rm -rf "$WORK"/*prp* "$WORK"/proof* "$WORK"/results.txt

# Interrupt the first run once the check at iteration 0 has passed. A backup is written at
# every iteration (-t 0), which slows the run down so the interrupt lands inside the first block.
# The run reports the index of the last iteration it did ("state saved at iteration N"), so N = 0 is
# the checkpoint right after the check at iteration 0 and N < K is still before the next check, at
# iteration K. A run that is interrupted later, or finishes first, is repeated.
saved=""
for try in $(seq 1 20); do
    rm -rf "$WORK"/*prp* "$WORK"/proof* first.log
    ( exec $RUN timeout --signal=INT --kill-after=10s 300 "$ROOT/prmers" "$P" "${ARGS[@]}" -t 0 > first.log 2>&1 ) &
    pid=$!
    until grep -aq 'Check passed! iter=0$' first.log 2>/dev/null; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.001
    done
    kill -INT "$pid" 2>/dev/null || true
    wait "$pid" || true
    saved="$(sed -n 's/.*state saved at iteration \([0-9]*\).*/\1/p' first.log | tail -n 1)"
    if [ -n "$saved" ] && [ "$saved" -ge 0 ] && [ "$saved" -lt "$K" ]; then
        echo "interrupted at iteration $saved (block size $K)"
        break
    fi
    echo "try $try: interrupt did not land in the first block (saved=${saved:-none}), retrying"
    saved=""
done
[ -n "$saved" ] || { echo "FAIL: could not interrupt the run inside the first block"; exit 1; }

set +e
prmers -erroriter $((K + 1)) > second.log 2>&1
rc=$?
set -e
echo "RESUMED_RC=$rc"
grep -q "Injected error at iteration $((K + 1))" second.log || { cat second.log; echo "FAIL: error was not injected"; exit 1; }
grep -q 'Check FAILED' second.log || { cat second.log; echo "FAIL: the injected error was not detected"; exit 1; }
[ "$rc" -ne 124 ] || { echo "FAIL: resumed run did not terminate"; exit 1; }
if grep -q 'times in a row' second.log; then
    cat second.log
    echo "FAIL: resumed run gave up after repeated failed checks"
    exit 1
fi
second_res="$(residue second.log)"
[ "$second_res" = "$clean_res" ] || { cat second.log; echo "FAIL: residue ${second_res:-none} != $clean_res after resume and recovery"; exit 1; }

echo "PRP_LEGACY_RESUME_AFTER_ITER0_CHECK=PASS"
