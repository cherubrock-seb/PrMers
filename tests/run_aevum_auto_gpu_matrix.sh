#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
BASE="${AEVUM_AUTO_MATRIX_DIR:-$ROOT/tests/aevum-auto-gpu-matrix}"
SHORT="${AEVUM_AUTO_MATRIX_SHORT_SECONDS:-12}"
LONG="${AEVUM_AUTO_MATRIX_LONG_SECONDS:-35}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing PrMers binary: $BIN" >&2
  exit 2
fi

rm -rf "$BASE"
mkdir -p "$BASE"

run_case() {
  local name="$1" expected="$2" seconds="$3"
  shift 3
  local dir="$BASE/$name"
  local log="$dir/run.log"
  mkdir -p "$dir"

  echo
  echo "== $name: expecting $expected =="
  set +e
  (
    cd "$dir"
    timeout --signal=INT --kill-after=15s "${seconds}s" \
      "$BIN" "$@" -d "$DEVICE" --noask -f "$dir"
  ) >"$log" 2>&1
  local rc=$?
  set -e

  cat "$log"
  if grep -Eqi 'Segmentation fault|Memory access fault|core dumped|Abandon|Aevum create failed' "$log"; then
    echo "FAIL: crash signature in $name" >&2
    exit 1
  fi
  if ! grep -Fq "$expected" "$log"; then
    echo "FAIL: expected backend decision not found in $name" >&2
    exit 1
  fi
  if [[ $rc -ne 0 && $rc -ne 1 && $rc -ne 124 && $rc -ne 130 ]]; then
    echo "FAIL: unexpected exit=$rc in $name" >&2
    exit 1
  fi
  echo "PASS: $name (exit=$rc)"
}

# Ratio 4.00: Marin is clearly smaller.
run_case prp-small "[Backend Auto] PRP: Marin selected" "$SHORT" \
  1362763 -prp -proof 0

# Ratio 0.50: Aevum is clearly smaller.
run_case prp-large "[Backend Auto] PRP: Aevum selected" "$SHORT" \
  136279841 -prp -proof 0

# Measured Radeon VII case: Aevum ~331 IPS versus Marin ~301 IPS after warm-up.
run_case pm1-stage1-large "[Backend Auto] P-1: Aevum selected" "$LONG" \
  136279841 -pm1 -b1 1000000

# ECM is conservative: small transforms remain on Marin, a 2x Aevum
# transform advantage selects Aevum for the 51-register curve engine.
run_case ecm-small "[Backend Auto] ECM: Marin selected" "$SHORT" \
  1362763 -ecm -b1 100 -b2 1000 -K 1
run_case ecm-large "[Backend Auto] ECM: Aevum selected" "$SHORT" \
  136279841 -ecm -b1 100 -b2 1000 -K 1

echo
echo "Aevum automatic GPU backend matrix passed. Logs: $BASE"
