#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-1}"
START="${START:-45951681}"
COUNT="${COUNT:-20}"
PM1_B1="${PM1_B1:-100000}"
PM1_B2="${PM1_B2:-1000000}"
ECM_B1="${ECM_B1:-0}"
ECM_B2="${ECM_B2:-0}"
ECM_CURVES="${ECM_CURVES:-0}"
SIEVE="${SIEVE:-1000000000000}"
CHUNK_BITS="${CHUNK_BITS:-262144}"
CAMPAIGN="${CAMPAIGN:-gm-record-${START}-${COUNT}}"
OUT="${OUT:-$ROOT/$CAMPAIGN}"
WORKTODO="${WORKTODO:-$OUT/worktodo-gm.txt}"

mkdir -p "$OUT/results"
cd "$ROOT"

if [[ ! -s "$WORKTODO" ]]; then
  ./scripts/generate_gaussian_worktodo.py \
    --start "$START" --count "$COUNT" --output "$WORKTODO" --mode chain \
    --pm1-b1 "$PM1_B1" --pm1-b2 "$PM1_B2" \
    --ecm-b1 "$ECM_B1" --ecm-b2 "$ECM_B2" --curves "$ECM_CURVES" \
    --sieve "$SIEVE" --chunk-bits "$CHUNK_BITS"
fi

cat <<EOF
Gaussian-Mersenne record campaign
  worktodo : $WORKTODO
  output   : $OUT/results
  GPU      : $DEVICE
  P-1      : B1=$PM1_B1 B2=$PM1_B2
  ECM      : B1=$ECM_B1 B2=$ECM_B2 curves=$ECM_CURVES
  sieve    : $SIEVE
EOF

# PrMers removes a completed line, archives it in worktodo_save.txt and re-execs
# itself. An interrupted line stays first and resumes from its checkpoint.
set +e
./prmers \
  -worktodo "$WORKTODO" \
  -aevum -d "$DEVICE" -r \
  -f "$OUT/results" \
  2>&1 | tee -a "$OUT/campaign.log"
rc=${PIPESTATUS[0]}
set -e

printf 'PrMers exit code: %s\n' "$rc"
exit "$rc"
