#!/usr/bin/env bash
set -euo pipefail

P="${1:-1362763}"
DEVICE="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/benchmark-aevum-$P"
mkdir -p "$OUT/marin" "$OUT/aevum"

run_one() {
    local name="$1"
    shift
    echo "=== $name ==="
    /usr/bin/time -f "$name elapsed=%e maxrss_kb=%M" \
        "$ROOT/prmers" "$P" -prp -proof 0 -d "$DEVICE" --noask \
        -f "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}

run_one marin
run_one aevum -aevum
