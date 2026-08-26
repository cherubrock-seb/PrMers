#!/usr/bin/env bash
set -euo pipefail
PR="${1:-./prmers}"
DEV="${2:-0}"
P="${3:-15000017}"
SEED="${4:-99820015000017}"
SECONDS="${5:-600}"
B1="${6:-2000}"
B2="${7:-100000}"
ROOT="${8:-$HOME/gm-ecm-v998-stage2-bench}"

rm -rf "$ROOT"
mkdir -p "$ROOT/legacy" "$ROOT/optimized"

echo "=== LEGACY Stage1+Stage2 ${SECONDS}s ==="
timeout -s INT "$SECONDS" "$PR" "$P" \
  -gm-ecm -gm-family GM \
  -b1 "$B1" -b2 "$B2" -K 1 \
  -seed "$SEED" -gm-sieve 0 \
  -aevum-auto -d "$DEV" -t 120 \
  -f "$ROOT/legacy" \
  2>&1 | tee "$ROOT/legacy.log" || true

echo
echo "=== v99.98 FUSED+BSGS Stage1+Stage2 ${SECONDS}s ==="
timeout -s INT "$SECONDS" "$PR" "$P" \
  -gm-ecm -gm-family GM -bsgs \
  -b1 "$B1" -b2 "$B2" -K 1 \
  -seed "$SEED" -gm-sieve 0 \
  -aevum-auto -d "$DEV" -t 120 \
  -f "$ROOT/optimized" \
  2>&1 | tee "$ROOT/optimized.log" || true

echo
echo "=== SAME CURVE ==="
grep -m1 '\[GM ECM\] curve' "$ROOT/legacy.log" || true
grep -m1 '\[GM ECM\] curve' "$ROOT/optimized.log" || true

echo
echo "=== LEGACY LAST ==="
tr '\r' '\n' < "$ROOT/legacy.log" | \
  grep -E 'Stage 1|Stage 2|factor:' | tail -20 || true

echo
echo "=== OPTIMIZED LAST ==="
tr '\r' '\n' < "$ROOT/optimized.log" | \
  grep -E 'Stage 1|Stage 2|BSGS|factor:' | tail -25 || true
