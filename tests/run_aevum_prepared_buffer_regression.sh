#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
OUT="${2:-$ROOT/tests/aevum-prepared-buffer-regression}"
SECONDS_TO_RUN="${AEVUM_REGRESSION_SECONDS:-60}"
rm -rf "$OUT"
mkdir -p "$OUT"
set +e
timeout "$SECONDS_TO_RUN" "$ROOT/prmers" 1362763 \
    -pm1 -b1 29 -b2 6910159 -aevum -d "$DEVICE" --noask -f "$OUT" \
    >"$OUT/run.log" 2>&1
rc=$?
set -e
grep -q '\[Backend Aevum\] engine::Reg adapter active' "$OUT/run.log"
if grep -Eq 'Memory access fault|Aevum .* failed|Abandon|core dumped' "$OUT/run.log"; then
    cat "$OUT/run.log"
    exit 1
fi
echo "Prepared multiplicand regression passed for ${SECONDS_TO_RUN}s, exit=$rc"
