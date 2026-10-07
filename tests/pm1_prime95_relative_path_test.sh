#!/usr/bin/env bash
# -p95path may be a relative directory.  Prime95 is started after "cd <dir>",
# so a relative log-file path built from the directory used to point at a
# non-existent <dir>/<dir>/..., the shell redirect failed, and Prime95 never
# ran ("did not produce results.json.txt").  A stub "mprime" stands in for
# Prime95: it copies a canned results.json.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

mkdir -p "$WORK/p95"
printf '%s\n' '{"status":"NF", "exponent":113, "worktype":"P-1", "b1":4, "b2":2141, "d":30}' >"$WORK/p95/canned.json"
printf '#!/bin/sh\ncp canned.json results.json.txt\n' >"$WORK/p95/mprime"
chmod +x "$WORK/p95/mprime"
( cd "$WORK" && ln -s "$ROOT/kernels" kernels &&
  timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 113 -pm1 -b1 4 -b2 2141 -p95path ./p95 \
    -d "$DEVICE" --noask >run.log 2>&1 || true )
if grep -aq 'did not produce results.json.txt' "$WORK/run.log"; then
  echo "relative -p95path: Prime95 did not run" >&2; exit 1
fi
grep -aq 'No factor P-1 (stage 2) until B2 = 2141' "$WORK/run.log" || { echo "relative -p95path: no stage-2 result" >&2; exit 1; }
echo "pm1 Prime95 relative path test passed"
