#!/usr/bin/env bash
# GPU regression: an LL-SAFE2 run that is interrupted between two verified block
# boundaries and then resumed must still report a Mersenne prime as prime.
# Uses a short run of M11213 with a block size that does not divide the
# interruption point. Usage: run_llsafe2_resume_regression.sh [device] [exponent]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-11213}"
B=101
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
ln -s "$ROOT/kernels" kernels

for attempt in 1 2 3 4 5; do
    rm -f llsafe2_m_*.ckpt*
    set +e
    timeout -s INT "${attempt}" "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" -engine-marin -d "$DEVICE" -noask -f "$WORK" >first.log 2>&1
    set -e
    saved="$(sed -n 's/.*state saved at iteration \([0-9]*\).*/\1/p' first.log | tail -n 1)"
    if [ -z "$saved" ]; then
        echo "run finished before it could be interrupted, retrying"
        continue
    fi
    if [ $((saved % B)) -eq 0 ]; then
        echo "interrupted at a block boundary ($saved), retrying"
        continue
    fi
    echo "interrupted at iteration $saved (block size $B)"
    timeout -s INT 50 "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" -engine-marin -d "$DEVICE" -noask -f "$WORK" >second.log 2>&1 || true
    grep -q "Resuming from a checkpoint" second.log
    if grep -q "is prime" second.log && ! grep -q "composite" second.log; then
        echo "PASS: 2^$P - 1 reported prime after resume"
        exit 0
    fi
    cat second.log
    echo "FAIL: resumed LL-SAFE2 run did not report 2^$P - 1 as prime"
    exit 1
done
echo "could not interrupt the run between block boundaries"
exit 1
