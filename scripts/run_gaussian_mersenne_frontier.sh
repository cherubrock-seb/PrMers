#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-1}"
FACTOR_LIMIT="${FACTOR_LIMIT:-1000000000000}"
MODE="${MODE:--gm-prp}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/gm-results}"
cd "$ROOT"
mkdir -p "$RESULT_ROOT"

# Prime exponents beyond 15,317,227. They are examples, not a claim that
# the corresponding work has never been assigned or tested elsewhere.
EXPONENTS=(15317251 15400031 16000057 18000041 20000003)

for p in "${EXPONENTS[@]}"; do
  echo "===== Gaussian-Mersenne p=$p ====="
  ./prmers "$p" "$MODE" -gm-base 3 -aevum -d "$DEVICE" \
    -gm-sieve "$FACTOR_LIMIT" -t 1800 -f "$RESULT_ROOT/$p" \
    2>&1 | tee "$RESULT_ROOT/gm-${p}.log" || true
done
