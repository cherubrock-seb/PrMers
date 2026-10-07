#!/usr/bin/env bash
# -p95path may be a relative directory.  Prime95 is started after "cd <dir>",
# so a relative executable/log path built from the directory used to point at
# a non-existent <dir>/<dir>/..., the shell command failed, and Prime95 never
# ran ("did not produce results.json.txt").  A stub "mprime" stands in for
# Prime95: it copies a canned results.json.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

mkdir -p "$WORK/p95"
printf '%s\n' '{"status":"NF", "exponent":1693, "worktype":"ECM", "b1":3, "b2":2000}' >"$WORK/p95/canned.json"
printf '#!/bin/sh\ncp canned.json results.json.txt\n' >"$WORK/p95/mprime"
chmod +x "$WORK/p95/mprime"
( cd "$WORK" && ln -s "$ROOT/kernels" kernels &&
  timeout --signal=INT --kill-after=10s 50 "$BIN" 1693 -ecm -ced -notorsion -b1 3 -b2 2000 -K 1 -p95path ./p95 \
    -d "$DEVICE" --noask >run.log 2>&1 || true )
if grep -aq 'did not produce results.json.txt' "$WORK/run.log"; then
  echo "relative -p95path: Prime95 did not run" >&2; exit 1
fi
if grep -aq 'Prime95 Stage2 disabled' "$WORK/run.log"; then
  echo "relative -p95path: Prime95 stage 2 was disabled" >&2; exit 1
fi
grep -aq 'Stage2 .*Prime95\|Prime95 .*finished\|Prime95 Stage2' "$WORK/run.log" || { echo "relative -p95path: no Prime95 stage-2 activity" >&2; exit 1; }
grep -a 'Prime95' "$WORK/run.log" | head -4
echo "ecm Prime95 relative path test passed"
