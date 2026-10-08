#!/usr/bin/env bash
# The result line of an external Prime95 stage 2 is trusted only where it is
# plausible:
#  - the B2 Prime95 reports is the bound recorded, not the one requested, when
#    it lies between B1 and the request;
#  - a B2 that is zero or not above B1 invalidates the result (the internal
#    stage 2 runs instead) and is never recorded;
#  - a B2 beyond the request is not claimed;
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

# 3. A reported B2 of 0, B1 or less than B1 is impossible: it must not become
# the recorded bound in any result, file or JSON.  The internal stage 2 runs
# for the requested B2 instead.
for bad in 0 3 4; do
  run_case "bad$bad" "{\"status\":\"NF\", \"exponent\":113, \"worktype\":\"P-1\", \"b1\":4, \"b2\":$bad, \"d\":30}"
  log="$WORK/bad$bad/run.log"
  grep -aq "Prime95 Stage2 error: result reports B2=$bad," "$log" || fail "bad$bad: invalid B2 not rejected"
  if grep -aq "reached B2=$bad" "$log"; then fail "bad$bad: invalid B2 adopted"; fi
  if grep -aq "until B2 = $bad\$" "$log"; then fail "bad$bad: invalid B2 reported"; fi
  if ls "$WORK/bad$bad"/stage2_result_B2_"$bad"_p_113.txt >/dev/null 2>&1; then fail "bad$bad: result file for B2=$bad"; fi
  if grep -aq "\"b2\":$bad[^0-9]" "$WORK/bad$bad/results.txt" 2>/dev/null; then fail "bad$bad: results.txt records b2=$bad"; fi
  if ls "$WORK/bad$bad"/*stage2_ext* >/dev/null 2>&1; then fail "bad$bad: external-stage-2 JSON written"; fi
  ls "$WORK/bad$bad"/stage2*_result_B2_2141_p_113.txt >/dev/null || fail "bad$bad: internal stage 2 did not produce the B2=2141 result"
done

# 4. Prime95 claims more than was requested: keep the requested bound.
run_case big '{"status":"NF", "exponent":113, "worktype":"P-1", "b1":4, "b2":99999999, "d":30}'
grep -aq 'reported B2=99999999 beyond the requested B2=2141' "$WORK/big/run.log" || fail "big: no warning"
grep -aq 'No factor P-1 (stage 2) until B2 = 2141' "$WORK/big/run.log" || fail "big: B2 not kept at 2141"
grep -aq '"b2":2141' "$WORK/big/results.txt" || fail "big: results.txt does not record b2=2141"
if grep -aq '99999999' "$WORK/big/results.txt"; then fail "big: unrequested B2 recorded"; fi

echo "pm1 Prime95 stage-2 result test passed"
