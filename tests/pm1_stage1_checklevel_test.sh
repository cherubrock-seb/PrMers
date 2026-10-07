#!/usr/bin/env bash
# -checklevel N must make the normal stage-1 loop run a Gerbicz-Li check every
# N blocks.  M269 with B1=20000 has 28830 bits per chunk, so B=169 and about
# 171 blocks; -checklevel 2 gives about 85 mid-chunk checks (it used to run only
# the final one).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

( cd "$WORK" && ln -s "$ROOT/kernels" kernels &&
  timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 269 -pm1 -b1 20000 -b2 20000 -checklevel 2 \
    -d "$DEVICE" --noask >run.log 2>&1 || true )
checks=$(grep -ac 'Start a Gerbicz Li check' "$WORK/run.log" || true)
if [ "$checks" -lt 50 ]; then
  echo "expected about 85 Gerbicz-Li checks with -checklevel 2, saw $checks" >&2; exit 1
fi
if grep -aq 'Gerbicz Li\] Mismatch' "$WORK/run.log"; then
  echo "unexpected Gerbicz-Li mismatch" >&2; exit 1
fi
grep -aq '"factors":\["13822297"\]' "$WORK/results.txt" || { echo "factor 13822297 not found" >&2; exit 1; }
echo "pm1 stage-1 checklevel test passed ($checks checks)"
