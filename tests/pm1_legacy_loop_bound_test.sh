#!/usr/bin/env bash
# The legacy (-marin) P-1 stage 1 keeps its resume position in a text .loop file.  A
# position beyond the bits of E (a damaged file, or one left by another B1) must be
# refused and stage 1 started from the beginning, not resumed with extra bits.
# A genuine interrupted run must still resume.  M269 has the P-1 factor 13822297.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
B1=20000
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

newdir() { mkdir -p "$WORK/$1"; ln -s "$ROOT/kernels" "$WORK/$1/kernels"; }
run() { ( cd "$WORK/$1" && timeout -k 10 "${2:-300}" "$ROOT/prmers" 269 -pm1 -b1 $B1 -marin -d "$DEVICE" --noask >"${3:-run}.log" 2>&1 || true ); }

newdir ctl
START=$(date +%s)
run ctl
ELAPSED=$(( $(date +%s) - START ))
grep -aq 'P-1 factor stage 1 found: 13822297' "$WORK/ctl/run.log" || { echo "control: factor not found" >&2; exit 1; }

# A completed run leaves a finished state; tamper with its position.
for POS in 99999 4294967301 18446744073709551615; do
  d="beyond_$POS"
  newdir "$d"
  cp "$WORK/ctl/269pm1$B1".* "$WORK/$d/"
  printf '%s' "$POS" >"$WORK/$d/269pm1$B1.loop"
  # Without the manifest (".state"), as an earlier version left them: the bound check refuses it.
  # With it, the edited .loop does not match the saved set, which is refused before.
  rm -f "$WORK/$d/269pm1$B1".state*
  run "$d"
  grep -aq 'is beyond the .* bits of E; ignoring it' "$WORK/$d/run.log" || { echo "$d: position was not refused" >&2; exit 1; }
  grep -aq 'P-1 factor stage 1 found: 13822297' "$WORK/$d/run.log" || { echo "$d: factor not found after restart" >&2; exit 1; }
done

# An edited .loop with the manifest still there is not used either.
newdir edited
cp "$WORK/ctl/269pm1$B1".* "$WORK/edited/"
printf '%s' 99999 >"$WORK/edited/269pm1$B1.loop"
run edited
grep -aq 'Resuming from iteration 99999' "$WORK/edited/run.log" && { echo "edited: resumed from an edited .loop" >&2; exit 1; }
grep -aq 'P-1 factor stage 1 found: 13822297' "$WORK/edited/run.log" || { echo "edited: factor not found" >&2; exit 1; }

# A genuine interruption resumes.
newdir half
CUT=$(( ELAPSED / 2 )); [ "$CUT" -ge 1 ] || CUT=1
( cd "$WORK/half" && timeout --signal=INT -k 10 "$CUT" "$ROOT/prmers" 269 -pm1 -b1 $B1 -marin -d "$DEVICE" --noask >first.log 2>&1 || true )
[ -f "$WORK/half/269pm1$B1.loop" ] || { echo "half: no loop file after the interruption" >&2; exit 1; }
run half 300 second
grep -aq 'Resuming from iteration' "$WORK/half/second.log" || { echo "half: the run did not resume" >&2; exit 1; }
grep -aq 'P-1 factor stage 1 found: 13822297' "$WORK/half/second.log" || { echo "half: factor not found after resuming" >&2; exit 1; }
echo "pm1 legacy loop bound test passed"
