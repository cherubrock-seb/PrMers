#!/usr/bin/env bash
# GPU regression: tiny exponents must not hang while the OpenCL sizes are
# computed. With 1-bit digits (p = 2, 3) the carry-propagation depth search in
# Context::computeOptimalSizes never terminated, so `prmers 2 -gm` stopped
# after "Transform Size = 2" and had to be killed.
# Usage: run_tiny_exponent_regression.sh [device]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
ln -s "$ROOT/kernels" kernels

status=0
check() {
    local name="$1" expect="$2"; shift 2
    local out rc=0
    out="$(timeout -s INT 15 "$ROOT/prmers" "$@" -d "$DEVICE" -noask -f "$WORK" 2>&1)" || rc=$?
    if [[ "$rc" -eq 124 ]]; then
        echo "FAIL: $name timed out (hang)"
        status=1
    elif echo "$out" | grep -q "$expect"; then
        echo "PASS: $name"
    else
        echo "FAIL: $name: expected '$expect' (rc=$rc)"
        echo "$out" | tail -5
        status=1
    fi
}

check "prmers 2 -gm" "GM_2 special value 5 is prime" 2 -gm
check "prmers 2 -prp" "M2 is prime" 2 -prp
exit "$status"
