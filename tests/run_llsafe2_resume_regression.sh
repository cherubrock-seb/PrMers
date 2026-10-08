#!/usr/bin/env bash
# GPU regression: an LL-SAFE2 run that is interrupted between two verified block
# boundaries and then resumed must still report a Mersenne prime as prime.
# Uses a short run of M11213 with a block size that does not divide the
# interruption point. Usage: run_llsafe2_resume_regression.sh [device] [exponent]
#
# The interrupt is sent as soon as the run reports its first verified block, not
# after a fixed time, so it lands shortly after a boundary whatever the device
# speed.  If it lands exactly on a boundary, or the run finishes first, the
# attempt is repeated, and if that keeps happening a larger exponent and block
# size are used.  An exponent given on the command line is used on its own.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
# Mersenne exponent / block size pairs: about five blocks per run.
if [ -n "${2:-}" ]; then
    RUNGS=("$2:$(( $2 / 5 ))")
else
    RUNGS=(11213:2000 23209:4600 44497:8800 110503:22000 216091:43000)
fi
TRIES=4
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

n=0
for rung in "${RUNGS[@]}"; do
    P="${rung%%:*}"
    B="${rung##*:}"
    for try in $(seq 1 "$TRIES"); do
        n=$((n + 1))
        dir="$WORK/run$n"
        mkdir "$dir"
        ln -s "$ROOT/kernels" "$dir/kernels"
        ( cd "$dir" && exec timeout --signal=INT --kill-after=10s 900 "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" \
            -engine-marin -d "$DEVICE" -noask -f "$dir" >first.log 2>&1 ) &
        pid=$!
        until grep -aq 'Check passed' "$dir/first.log" 2>/dev/null; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.005
        done
        kill -INT "$pid" 2>/dev/null || true
        wait "$pid" || true
        saved="$(sed -n 's/.*state saved at iteration \([0-9]*\).*/\1/p' "$dir/first.log" | tail -n 1)"
        if [ -z "$saved" ]; then
            echo "M$P: run finished before it could be interrupted, retrying"
            continue
        fi
        if [ "$saved" -le "$B" ] || [ $((saved % B)) -eq 0 ]; then
            echo "M$P: interrupted at a block boundary or in the first block ($saved), retrying"
            continue
        fi
        echo "interrupted M$P at iteration $saved (block size $B)"
        cd "$dir"
        timeout -s INT --kill-after=10s 900 "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" -engine-marin -d "$DEVICE" -noask -f "$dir" >second.log 2>&1 || true
        grep -q "Resuming from a checkpoint" second.log || { cat second.log; echo "FAIL: the second run did not resume from the checkpoint"; exit 1; }
        if grep -q "is prime" second.log && ! grep -q "composite" second.log; then
            echo "PASS: 2^$P - 1 reported prime after resume"
            exit 0
        fi
        cat second.log
        echo "FAIL: resumed LL-SAFE2 run did not report 2^$P - 1 as prime"
        exit 1
    done
done
echo "could not interrupt the run between block boundaries"
exit 1
