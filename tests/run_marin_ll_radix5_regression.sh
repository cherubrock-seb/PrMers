#!/usr/bin/env bash
# GPU regression: the legacy (-marin) Lucas-Lehmer path must report Mersenne
# primes as prime on radix-5 transform sizes (N = 5 * 2^k). M2203 and M4423 use
# N = 80 and N = 160. The LL test is selected through a worktodo Test= line. Legacy LL is not
# validated and is rejected by default, so the run opts in with -allow-unvalidated-legacy-ll.
# Usage: run_marin_ll_radix5_regression.sh [device]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
ln -s "$ROOT/kernels" kernels

status=0
for p in 2203 4423; do
    printf 'Test=N/A,%s,60,1\n' "$p" > worktodo.txt
    out="$(timeout -s INT 50 "$ROOT/prmers" -marin -allow-unvalidated-legacy-ll -d "$DEVICE" -worktodo worktodo.txt -noask -f "$WORK" 2>&1 || true)"
    if echo "$out" | grep -q "M$p is prime"; then
        echo "PASS: M$p reported prime"
    else
        echo "FAIL: M$p not reported prime"
        echo "$out" | grep -E "is prime|composite|rror" || true
        status=1
    fi
    rm -f ./*.mers ./*.loop ./*.bufd ./*.lbufd ./*.gli ./*.isav ./*.jsav ./*.ckpt*
done
exit "$status"
