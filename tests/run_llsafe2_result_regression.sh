#!/usr/bin/env bash
# GPU regression: a finished LL-SAFE2 run must write its result (results.txt and
# the per-exponent JSON, with status/res64) and delete its own checkpoint.
# The run is interrupted once so that a checkpoint exists, then resumed.
# Short run of M11213 (a Mersenne prime).
# Usage: run_llsafe2_result_regression.sh [device] [exponent]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-11213}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
ln -s "$ROOT/kernels" kernels

for attempt in 1 2 3 4 5; do
    rm -f llsafe2_m_*.ckpt* results.txt ./*_result.json
    set +e
    timeout -s INT "${attempt}" "$ROOT/prmers" "$P" -llsafe2 -engine-marin -d "$DEVICE" -noask >first.log 2>&1
    set -e
    if ! grep -q "state saved at iteration" first.log; then
        echo "run finished before it could be interrupted, retrying"
        continue
    fi
    [ -f "llsafe2_m_$P.ckpt" ] || { echo "FAIL: no checkpoint written on interrupt"; exit 1; }
    timeout -s INT 50 "$ROOT/prmers" "$P" -llsafe2 -engine-marin -d "$DEVICE" -noask >second.log 2>&1 || true
    grep -q "Resuming from a checkpoint" second.log
    fail=0
    [ -s results.txt ] || { echo "FAIL: results.txt not written"; fail=1; }
    [ -s "${P}_llsafe2_result.json" ] || { echo "FAIL: ${P}_llsafe2_result.json not written"; fail=1; }
    grep -q '"status":"P"' results.txt || { echo "FAIL: result is not status P"; fail=1; }
    grep -q '"worktype":"LL"' results.txt || { echo "FAIL: worktype is not LL"; fail=1; }
    grep -q '"res64":"0000000000000000"' results.txt || { echo "FAIL: res64 missing or wrong"; fail=1; }
    ls llsafe2_m_*.ckpt* >/dev/null 2>&1 && { echo "FAIL: LL-SAFE2 checkpoint not deleted"; fail=1; }
    if [ "$fail" -ne 0 ]; then
        cat results.txt 2>/dev/null || true
        exit 1
    fi
    echo "PASS: LL-SAFE2 result saved and checkpoint removed for 2^$P - 1"
    exit 0
done
echo "could not interrupt the run"
exit 1
