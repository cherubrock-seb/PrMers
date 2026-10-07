#!/usr/bin/env bash
# The result line of an external Prime95 stage 2 must be taken at face value:
#  - the B2 Prime95 reports is the bound recorded, not the one requested;
#  - every factor in "factors" is recorded, not only the first.
# A stub "mprime" stands in for Prime95: it only copies a canned
# results.json.txt, so no Prime95 install is needed.  M113 has the factors
# 23279 and 65993; with B1=4 stage 1 finds neither.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# run_case <name> <canned results.json.txt line>
run_case() {
  local name="$1" line="$2"
  mkdir -p "$WORK/$name/p95"
  printf '%s\n' "$line" >"$WORK/$name/p95/canned.json"
  cat >"$WORK/$name/p95/mprime" <<'STUB'
#!/bin/sh
cp canned.json results.json.txt
STUB
  chmod +x "$WORK/$name/p95/mprime"
  ( cd "$WORK/$name" && ln -s "$ROOT/kernels" kernels &&
    timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 113 -pm1 -b1 4 -b2 2141 -p95path "$WORK/$name/p95" \
      -d "$DEVICE" --noask >run.log 2>&1 || true )
}

fail() { echo "$1" >&2; exit 1; }

# 1. Prime95 stops short of the requested B2 and finds nothing.
run_case short '{"status":"NF", "exponent":113, "worktype":"P-1", "b1":4, "b2":1000, "d":30}'
grep -aq 'Prime95 Stage2 reached B2=1000 instead of the requested B2=2141' "$WORK/short/run.log" || fail "short: no B2 warning"
grep -aq 'No factor P-1 (stage 2) until B2 = 1000' "$WORK/short/run.log" || fail "short: B2 not reported as 1000"
grep -aq '"b2":1000' "$WORK/short/results.txt" || fail "short: results.txt does not record b2=1000"
ls "$WORK"/short/stage2_result_B2_1000_p_113.txt >/dev/null || fail "short: no stage-2 result file for B2=1000"

# 2. Prime95 lists two factors.
run_case two '{"status":"F", "exponent":113, "worktype":"P-1", "b1":4, "b2":2141, "d":30, "factors":["23279","65993"]}'
grep -aq 'P-1 factor stage 2 found: 23279' "$WORK/two/run.log" || fail "two: first factor missing"
grep -aq 'P-1 factor stage 2 found: 65993' "$WORK/two/run.log" || fail "two: second factor missing"
grep -aq '"factors":\[[^]]*23279[^]]*\]' "$WORK/two/results.txt" || fail "two: results.txt lacks 23279"
grep -aq '"factors":\[[^]]*65993[^]]*\]' "$WORK/two/results.txt" || fail "two: results.txt lacks 65993"

echo "pm1 Prime95 stage-2 result test passed"
