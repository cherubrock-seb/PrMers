#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
OUT="${2:-$ROOT/tests/aevum-forced-pm1-small-factor}"
SECONDS_TO_RUN="${AEVUM_SMALL_FACTOR_SECONDS:-45}"
rm -rf "$OUT"
mkdir -p "$OUT"
pushd "$OUT" >/dev/null
set +e
set -o pipefail
timeout --signal=INT --kill-after=10s "$SECONDS_TO_RUN" \
  "$ROOT/prmers" 136279841 -pm1 -b1 1000000 -aevum \
  -d "$DEVICE" --noask -f "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
set -e
popd >/dev/null
grep -q '\[Backend Aevum\] engine::Reg adapter active' "$OUT/run.log"
grep -q 'Aevum uses the generic square plus base-3 multiply path; fast3 is disabled' "$OUT/run.log"
if grep -Eq 'Chunk .*\[fast3\]|Loaded regScale|REGSCALE|Segmentation fault|Memory access fault|Aevum .* failed|Abandon|core dumped' "$OUT/run.log"; then
  exit 1
fi
if ! grep -Eq 'Progress:|P-1 factor stage 1 found|Elapsed time' "$OUT/run.log"; then
  echo 'Aevum did not reach the P-1 iteration loop' >&2
  exit 1
fi
if [ "$SECONDS_TO_RUN" -ge 20 ] && ! grep -q '\[Gerbicz Li\] Check passed' "$OUT/run.log"; then
  echo 'Aevum did not complete the first Gerbicz check' >&2
  exit 1
fi
echo "Forced Aevum P-1 generic-base regression passed for ${SECONDS_TO_RUN}s, exit=$rc"
