#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
OUT="${2:-$ROOT/tests/aevum-gpu-smoke}"
SECONDS_PER_TEST="${AEVUM_SMOKE_SECONDS:-45}"
P="${AEVUM_SMOKE_EXPONENT:-1362763}"
mkdir -p "$OUT"

run_and_check() {
    local name="$1"
    shift
    rm -rf "$OUT/$name"
    mkdir -p "$OUT/$name"
    set +e
    timeout "$SECONDS_PER_TEST" "$ROOT/prmers" "$@" -aevum -d "$DEVICE" --noask -f "$OUT/$name" >"$OUT/$name.log" 2>&1
    local rc=$?
    set -e
    grep -q '\[Backend Aevum\] engine::Reg adapter active' "$OUT/$name.log"
    if grep -Eq 'Memory access fault|Aevum .* failed|Abandon|core dumped' "$OUT/$name.log"; then
        cat "$OUT/$name.log"
        return 1
    fi
    echo "$name stayed in Aevum for ${SECONDS_PER_TEST}s, exit=$rc"
}

run_and_check prp "$P" -prp -proof 0
run_and_check ll "$P" -llunsafe
run_and_check pm1 "$P" -pm1 -b1 29 -b2 2000
run_and_check ecm "$P" -ecm -b1 50 -b2 2000 -K 1
