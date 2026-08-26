#!/usr/bin/env bash
set -euo pipefail

PR="${1:-./prmers}"
DEV="${2:-0}"
ROOT="${3:-$HOME/gm-ecm-v998-validation}"

rm -rf "$ROOT"
mkdir -p "$ROOT"

echo "============================================================"
echo "A - CURRENT NORMAL GM ECM / LEGACY STAGE 1 GOLDEN"
echo "============================================================"

mkdir -p "$ROOT/legacy-s1"

"$PR" 21403643 \
  -gm-ecm \
  -gm-family GM \
  -b1 2000 \
  -b2 2000 \
  -K 1 \
  -sigma 3059155915320676093 \
  -gm-sieve 0 \
  -aevum-auto \
  -d "$DEV" \
  -t 120 \
  -f "$ROOT/legacy-s1" \
  2>&1 | tee "$ROOT/legacy-s1.log"

grep -F \
  '>>> Gaussian pair ECM Stage 1 factor: 482978801775374901713' \
  "$ROOT/legacy-s1.log"

echo
echo "============================================================"
echo "B - v99.98 FUSED STAGE 1 / SAME EXACT CURVE"
echo "============================================================"

mkdir -p "$ROOT/opt-s1"

"$PR" 21403643 \
  -gm-ecm \
  -gm-family GM \
  -bsgs \
  -b1 2000 \
  -b2 2000 \
  -K 1 \
  -sigma 3059155915320676093 \
  -gm-sieve 0 \
  -aevum-auto \
  -d "$DEV" \
  -t 120 \
  -f "$ROOT/opt-s1" \
  2>&1 | tee "$ROOT/opt-s1.log"

grep -F \
  '>>> Gaussian pair ECM Stage 1 factor: 482978801775374901713' \
  "$ROOT/opt-s1.log"

echo
echo "============================================================"
echo "C - CURRENT NORMAL GM ECM / LEGACY STAGE 2 GOLDEN"
echo "============================================================"

mkdir -p "$ROOT/legacy-s2"

"$PR" 89 \
  -gm-ecm \
  -gm-family GM \
  -b1 20 \
  -b2 50 \
  -K 1 \
  -sigma 6 \
  -gm-sieve 0 \
  -aevum-auto \
  -d "$DEV" \
  -t 120 \
  -f "$ROOT/legacy-s2" \
  2>&1 | tee "$ROOT/legacy-s2.log"

grep -F \
  '>>> Gaussian pair ECM Stage 2 factor: 1069' \
  "$ROOT/legacy-s2.log"

echo
echo "============================================================"
echo "D - v99.98 BSGS STAGE 2 / SAME EXACT CURVE"
echo "============================================================"

mkdir -p "$ROOT/opt-s2"

"$PR" 89 \
  -gm-ecm \
  -gm-family GM \
  -bsgs \
  -b1 20 \
  -b2 50 \
  -K 1 \
  -sigma 6 \
  -gm-sieve 0 \
  -aevum-auto \
  -d "$DEV" \
  -t 120 \
  -f "$ROOT/opt-s2" \
  2>&1 | tee "$ROOT/opt-s2.log"

grep -F \
  '>>> Gaussian pair ECM Stage 2 factor: 1069' \
  "$ROOT/opt-s2.log"

echo
echo "============================================================"
echo "ALL FOUR GPU GOLDENS PASSED"
echo "============================================================"

echo
echo "=== Stage1 timing lines ==="
grep -E 'Stage 1.*factor|elapsed=' "$ROOT/legacy-s1.log" | tail -8 || true
grep -E 'Stage 1.*factor|elapsed=' "$ROOT/opt-s1.log" | tail -8 || true

echo
echo "=== Stage2 timing lines ==="
grep -E 'Stage 2|Stage2' "$ROOT/legacy-s2.log" | tail -12 || true
grep -E 'Stage 2|Stage2' "$ROOT/opt-s2.log" | tail -12 || true
