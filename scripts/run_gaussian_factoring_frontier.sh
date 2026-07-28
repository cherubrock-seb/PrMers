#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-1}"
B1_PM1="${B1_PM1:-100000}"
B2_PM1="${B2_PM1:-5000000}"
B1_ECM="${B1_ECM:-50000}"
B2_ECM="${B2_ECM:-5000000}"
CURVES="${CURVES:-20}"
OUT="${OUT:-$ROOT/gm-factor-frontier}"
EXPONENTS="${EXPONENTS:-15317251 15400031 16000057 18000041 20000003}"
cd "$ROOT"
mkdir -p "$OUT"

for p in $EXPONENTS; do
  echo "===== Gaussian-Mersenne p=$p P-1 ====="
  ./prmers "$p" -gm-pm1 -b1 "$B1_PM1" -b2 "$B2_PM1" -aevum -d "$DEVICE" \
    -gm-factor-chunk-bits 262144 -r -f "$OUT/p${p}-pm1" \
    2>&1 | tee "$OUT/p${p}-pm1.log" || true

  echo "===== Gaussian-Mersenne p=$p ECM ====="
  ./prmers "$p" -gm-ecm -b1 "$B1_ECM" -b2 "$B2_ECM" -K "$CURVES" \
    -aevum -d "$DEVICE" -gm-factor-chunk-bits 131072 -r \
    -f "$OUT/p${p}-ecm" 2>&1 | tee "$OUT/p${p}-ecm.log" || true
done
