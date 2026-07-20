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
  printf '
===== %s =====
' "$name"
  set +e
  timeout -s INT "${DURATION}s" "$@" -f "$dir" 2>&1 | tee "$dir/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then return "$rc"; fi
}

# New default: throughput:auto must choose 4:512:8:512:202 at M175.
run_bench auto-throughput \
  ./prmers "$EXPONENT" -d "$DEVICE" -proof 0

run_bench pow2-type4-lead \
  env AEVUM_REG_LEAD_CACHE=1 AEVUM_TYPE4_MULTI_Q=1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft 4:512:8:512:202 -proof 0

run_bench pow2-type4-canonical \
  env AEVUM_REG_LEAD_CACHE=0 AEVUM_TYPE4_MULTI_Q=1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft 4:512:8:512:202 -proof 0

# PFA bridge is deliberately opt-in until exact validation + A/B speed pass.
run_bench pfa9-bridge \
  env AEVUM_PFA_LEAD_BRIDGE=1 AEVUM_REG_LEAD_CACHE=1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:1:512:9:512:202 -proof 0

run_bench pfa9-canonical \
  env AEVUM_PFA_LEAD_BRIDGE=0 AEVUM_REG_LEAD_CACHE=0 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:1:512:9:512:202 -proof 0

# Same 4.50M transform with the alternative 1024x256 geometry.
run_bench pfa9-1k-canonical \
  env AEVUM_PFA_LEAD_BRIDGE=0 AEVUM_REG_LEAD_CACHE=0 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:1:1K:9:256:202 -proof 0

run_bench pfa9-1k-bridge \
  env AEVUM_PFA_LEAD_BRIDGE=1 AEVUM_REG_LEAD_CACHE=1 \
  ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:1:1K:9:256:202 -proof 0

run_bench marin \
  ./prmers "$EXPONENT" -d "$DEVICE" -engine-marin -proof 0

printf '
===== SUMMARY =====
'
for log in bench-*/run.log; do
  echo "--- $log"
  grep -E 'throughput auto|throughput candidate|version=|lead cache|lead bridge|multi-queue|FFT:|Progress:' "$log" | tail -16 || true
done

PRPLL_BIN="${PRPLL_BIN:-$HOME/mgpu/gpuowl/build-release/prpll}"
if [[ -x "$PRPLL_BIN" ]]; then
  echo
  echo "PRPLL reference command:"
  echo "$PRPLL_BIN -device $DEVICE -prp $EXPONENT -fft 4:512:8:512:202"
fi
