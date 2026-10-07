#!/usr/bin/env bash
# Regression checks for small P-1 driver defects:
#  1. -nogcd-stage1 without stage 2 must not record a "no factor" result (the
#     GCD never ran; M269 B1=2141 does have the factor 13822297).
#  2. -s3 must run stage 3 only, not fall through into stage 1.
#  3. A GMP-ECM .save file whose CHECKSUM does not match its residue must be
#     rejected when extending a stage-1 run.
#  4. A classic stage-2 run must resume from pm1_s2_m_<p>.ckpt.old when the
#     main checkpoint is missing (a crash between the two renames of a save).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off
fail() { echo "$1" >&2; exit 1; }

# run <dir> <timeout seconds> <prmers args...>
run() {
  local dir="$1" secs="$2"; shift 2
  mkdir -p "$WORK/$dir"
  [ -e "$WORK/$dir/kernels" ] || ln -s "$ROOT/kernels" "$WORK/$dir/kernels"
  ( cd "$WORK/$dir" && timeout --signal=INT --kill-after=10s "$secs" "$ROOT/prmers" "$@" \
      -d "$DEVICE" --noask >>"$WORK/$dir/run.log" 2>&1 || true )
}

# 1. -nogcd-stage1, no stage 2: no result line.
run nogcd 50 269 -pm1 -b1 2141 -nogcd-stage1
grep -aq 'ordinary GCD skipped' "$WORK/nogcd/run.log" || fail "nogcd: stage 1 GCD was not skipped"
if [ -s "$WORK/nogcd/results.txt" ]; then fail "nogcd: results.txt records a result although no GCD was run"; fi

# 2. -s3 must not fall through into stage 1.
run s3 50 269 -pm1 -b1 100 -b2 100
rm -f "$WORK/s3/run.log"
run s3 50 269 -pm1 -b1 100 -b3 20 -s3
grep -aq 'No factor P-1 (stage 3) until B3 = 20' "$WORK/s3/run.log" || fail "s3: stage 3 did not complete"
if grep -aq 'running Stage 1' "$WORK/s3/run.log"; then fail "s3: stage 1 ran after stage 3"; fi

# 3. Corrupted residue in a .save file with a CHECKSUM.
run chk 50 269 -pm1 -b1 100 -b2 100
rm -f "$WORK/chk/resume_p269_B1_100.p95" "$WORK/chk/run.log"
sed -i 's/X=0x\([0-9a-f]\)/X=0x1\1/' "$WORK/chk/resume_p269_B1_100.save"
run chk 50 269 -pm1 -b1 200 -b1old 100
grep -aq 'does not match the residue' "$WORK/chk/run.log" || fail "chk: corrupt .save was accepted"

# 4. Stage-2 resume from the .old checkpoint (D=6 keeps the run slow enough to interrupt).
export PRMERS_PM1_CLASSIC_D=6
A4=(677 -pm1 -b1 10 -b2 100000 -pm1-vtrace-off)
run old 3 "${A4[@]}"
grep -aq 'Stage 2 state saved at prime' "$WORK/old/run.log" || fail "old: stage 2 was not interrupted"
mv "$WORK/old/pm1_s2_m_677.ckpt" "$WORK/old/pm1_s2_m_677.ckpt.old"
rm -f "$WORK/old/run.log"
run old 50 "${A4[@]}"
grep -aq 'Resuming Stage 2 from checkpoint' "$WORK/old/run.log" || fail "old: stage 2 did not resume from the .old checkpoint"
grep -aq 'P-1 factor stage 2 found: 1943118631' "$WORK/old/run.log" || fail "old: resumed stage 2 missed the factor"

echo "pm1 small items test passed"
