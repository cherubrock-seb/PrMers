#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEVICE="${1:-1}"
DURATION="${2:-180}"
EXPONENT="${3:-175000039}"

run_bench() {
  local name="$1"; shift
  local dir="$ROOT/bench-$name"
  rm -rf "$dir"; mkdir -p "$dir"
  printf '\n===== %s =====\n' "$name"
  set +e
  timeout -s INT "${DURATION}s" "$@" -f "$dir" 2>&1 | tee "$dir/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then return "$rc"; fi
}

run_bench pow2-type4-lead \
  env AEVUM_REG_LEAD_CACHE=1 AEVUM_TYPE4_MULTI_Q=1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft 4:512:8:512:202 -proof 0

run_bench pow2-type4-canonical \
  env AEVUM_REG_LEAD_CACHE=0 AEVUM_TYPE4_MULTI_Q=1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft 4:512:8:512:202 -proof 0

run_bench pow2-type4-singleq \
  env AEVUM_REG_LEAD_CACHE=1 AEVUM_TYPE4_MULTI_Q=0 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft 4:512:8:512:202 -proof 0

run_bench pfa9-type1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:1:512:9:512:202 -proof 0

run_bench pow2-type1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -pfa-off -proof 0

run_bench marin \
  ./prmers "$EXPONENT" -d "$DEVICE" -engine-marin -proof 0

printf '\n===== SUMMARY =====\n'
for log in bench-*/run.log; do
  echo "--- $log"
  grep -E 'version=|register lead cache|multi-queue|FFT:|Progress:' "$log" | tail -12 || true
done

PRPLL_BIN="${PRPLL_BIN:-$HOME/mgpu/gpuowl/build-release/prpll}"
if [[ -x "$PRPLL_BIN" ]]; then
  echo
  echo "PRPLL reference command:"
  echo "$PRPLL_BIN -device $DEVICE -prp $EXPONENT -fft 4:512:8:512:202"
fi
