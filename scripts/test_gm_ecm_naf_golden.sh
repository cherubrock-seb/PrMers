#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./prmers}"
DEV="${PRMERS_TEST_DEVICE:-0}"
OUT="${PRMERS_GM_NAF_GOLDEN_OUT:-/tmp/prmers-gm-naf-golden}"
LOG="$OUT/run.log"
EXPECTED="482978801775374901713"
SIGMA="3059155915320676093"

rm -rf "$OUT"
mkdir -p "$OUT"

"$BIN" 21403643 \
  -gm-ecm \
  -gm-family GM \
  -b1 2000 \
  -b2 2000 \
  -K 1 \
  -sigma "$SIGMA" \
  -gm-sieve 0 \
  -edwards \
  -aevum-auto \
  -d "$DEV" \
  -r \
  -t 120 \
  -f "$OUT" \
  2>&1 | tee "$LOG"

grep -F "[GM ECM] curve 1/1 sigma=$SIGMA" "$LOG" >/dev/null
grep -F ">>> Gaussian pair ECM Stage 1 factor: $EXPECTED" "$LOG" >/dev/null

echo "GM ECM v99.97 NAF golden GPU regression: OK"
