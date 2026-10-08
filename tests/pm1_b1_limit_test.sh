#!/usr/bin/env bash
# P-1 stage-1 size limits: a B1 or -maxe the stage-1 exponent cannot represent is
# refused up front with a clear error (command line and worktodo.txt), while B1 at
# and beyond 2^32 on the default chunked Marin path is accepted and starts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$ROOT/kernels" "$WORK/kernels"
cd "$WORK"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# refused <name> <fragment of the error> <prmers args...>
refused() {
  local name="$1" frag="$2"; shift 2
  local rc=0
  timeout -k 5 60 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >"$name.log" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] || { echo "$name: expected a failure" >&2; exit 1; }
  grep -aq "^Error: .*$frag" "$name.log" || { echo "$name: no clear error ('$frag'):" >&2; head -5 "$name.log" >&2; exit 1; }
  if grep -aq 'Building E\|Chunk 1/' "$name.log"; then echo "$name: the run started anyway" >&2; exit 1; fi
}

# accepted <name> <seconds> <prmers args...>: not refused by the limit check; the run starts (and is interrupted).
accepted() {
  local name="$1" secs="$2"; shift 2
  timeout --signal=INT -k 10 "$secs" "$ROOT/prmers" "$@" -d "$DEVICE" --noask >"$name.log" 2>&1 || true
  if grep -aq '^Error: .*\(is too large\|-maxe asks\|build in one piece\)' "$name.log"; then echo "$name: rejected:" >&2; grep -a '^Error:' "$name.log" >&2; exit 1; fi
  grep -aq 'Start a P-1 factoring stage 1 up to B1=' "$name.log" || { echo "$name: stage 1 did not start" >&2; exit 1; }
}

# --- command line ---
refused too_big_b1      'B1=4611686018427387905 is too large'  269 -pm1 -b1 4611686018427387905
refused max_b1          'B1=18446744073709551615 is too large' 269 -pm1 -b1 18446744073709551615
refused huge_maxe       '-maxe asks for chunks'                269 -pm1 -b1 1000 -maxe 9999999999999999
refused big_maxe        '-maxe asks for chunks'                269 -pm1 -b1 1000 -maxe 16384
refused legacy_big_b1   'can build in one piece'               269 -pm1 -b1 50000000000 -marin
refused torus_big_b1    'can build in one piece'               269 -pm1 -b1 50000000000 -torus
refused ext_big_b1      'extending from -b1old 1000'           269 -pm1 -b1 50000000000 -b1old 1000
refused gm_big_b1       'can build in one piece'               269 -gm-pm1 -b1 50000000000

# --- worktodo.txt (bounds that bypass the command-line check) ---
printf 'Pminus1=1,2,269,-1,50000000000,0\n' >worktodo.txt
rc=0
timeout -k 5 60 "$ROOT/prmers" -d "$DEVICE" --noask -marin >wt_legacy.log 2>&1 || rc=$?
rm -f worktodo.txt
[ "$rc" -ne 0 ] && grep -aq '^Error: .*can build in one piece' wt_legacy.log || { echo "worktodo: legacy B1 limit not enforced" >&2; head -5 wt_legacy.log >&2; exit 1; }

# --- accepted: B1 around 2^32 on the chunked path (no result expected; just that it starts) ---
for B1 in 4294967295 4294967296 4294967297 100000000000; do
  accepted "b1_$B1" 6 269 -pm1 -b1 "$B1"
done
# An extension beyond 2^32 from a nearby B1old only builds the delta.
accepted ext_ok 6 269 -pm1 -b1 4294967296 -b1old 4294967200
echo "pm1 B1 limit test passed"
