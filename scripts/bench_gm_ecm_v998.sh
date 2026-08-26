#!/usr/bin/env bash
set -euo pipefail
PR="${1:-./prmers}"
DEV="${2:-0}"
P="${3:-15750019}"
SEED="${4:-1262417601919}"
SECONDS="${5:-180}"
ROOT="${6:-$HOME/gm-ecm-v998-bench}"

rm -rf "$ROOT"
mkdir -p "$ROOT/legacy" "$ROOT/optimized"

echo "=== LEGACY ${SECONDS}s ==="
timeout -s INT "$SECONDS" "$PR" "$P" \
  -gm-ecm -gm-family GM \
  -b1 50000 -b2 50000 -K 1 \
  -seed "$SEED" -gm-sieve 0 \
  -aevum-auto -d "$DEV" -t 120 \
  -f "$ROOT/legacy" \
  2>&1 | tee "$ROOT/legacy.log" || true

echo
echo "=== FUSED v99.98 ${SECONDS}s ==="
timeout -s INT "$SECONDS" "$PR" "$P" \
  -gm-ecm -gm-family GM -bsgs \
  -b1 50000 -b2 50000 -K 1 \
  -seed "$SEED" -gm-sieve 0 \
  -aevum-auto -d "$DEV" -t 120 \
  -f "$ROOT/optimized" \
  2>&1 | tee "$ROOT/optimized.log" || true

echo
echo "=== SAME SIGMA CHECK ==="
echo "legacy:"
grep -m1 '\[GM ECM\] curve' "$ROOT/legacy.log" || true
echo "optimized:"
grep -m1 '\[GM ECM\] curve' "$ROOT/optimized.log" || true

echo
echo "=== LAST PROGRESS ==="
echo "legacy:"
tr '\r' '\n' < "$ROOT/legacy.log" |
  grep -E 'GM ECM Stage 1 curve|ladder bits' | tail -3 || true
echo "optimized:"
tr '\r' '\n' < "$ROOT/optimized.log" |
  grep -E 'GM ECM Stage 1 fused curve|fused ladder bits' | tail -3 || true
