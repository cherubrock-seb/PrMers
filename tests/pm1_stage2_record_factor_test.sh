#!/usr/bin/env bash
# The low-memory, ultra-low-memory and n^K stage-2 variants must record the
# factor they find: a result line with the factor in results.txt, not only a
# message on the console.
#   M269: 13822297 = 2*269*(2^2*3*2141) + 1, found by stage 2 with B1=4, B2=2141.
#   M113: 418152599391647 = 65993 * 6336317479, found by the n^K variant with nmax=40, K=2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# check <name> <factor> <prmers args...>
check() {
  local name="$1" factor="$2"; shift 2
  mkdir -p "$WORK/$name"
  ( cd "$WORK/$name" && ln -s "$ROOT/kernels" kernels &&
    timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >run.log 2>&1 || true )
  grep -q "P-1 factor stage 2 found: $factor" "$WORK/$name/run.log" || { echo "$name: factor $factor not found" >&2; exit 1; }
  if ! grep -q "\"status\":\"F\".*\"factors\":\[\"$factor\"\]" "$WORK/$name/results.txt"; then
    echo "$name: no factor result line in results.txt" >&2; exit 1
  fi
  ls "$WORK/$name"/stage2_result_B2_*_p_*.txt >/dev/null 2>&1 || { echo "$name: no stage-2 result file" >&2; exit 1; }
}

check lowmem 13822297 269 -pm1 -b1 4 -b2 2141 -pm1-lowmem
check ultralowmem 13822297 269 -pm1 -b1 4 -b2 2141 -pm1-ultralowmem
check nk 418152599391647 113 -pm1 -b1 4 -nmax 40 -K 2
echo "pm1 stage-2 factor recording test passed"
