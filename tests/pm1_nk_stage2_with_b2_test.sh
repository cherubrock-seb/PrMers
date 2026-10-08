#!/usr/bin/env bash
# The n^K stage-2 variant must also run when a classic -b2 stage 2 ran first.
# The classic stage 2 releases the stage-1 engine, and the n^K hand-off used to
# write its checkpoint through that engine, which crashed with SIGSEGV.
#   M269, B1=4, B2=100, n^K with nmax=60, K=2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# check <name> <prmers args...>
check() {
  local name="$1"; shift
  mkdir -p "$WORK/$name"
  local rc=0
  ( cd "$WORK/$name" && ln -s "$ROOT/kernels" kernels &&
    timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >run.log 2>&1 ) || rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then echo "$name: prmers exited with status $rc" >&2; exit 1; fi
  grep -q "P-1 STAGE 2 IN \*\*\*\* n^K variant" "$WORK/$name/run.log" || { echo "$name: n^K stage 2 did not start" >&2; exit 1; }
  grep -q "Elapsed (n^K)" "$WORK/$name/run.log" || { echo "$name: n^K stage 2 did not finish" >&2; exit 1; }
  grep -q "stage 2 n^K" "$WORK/$name/run.log" || { echo "$name: no n^K stage-2 result" >&2; exit 1; }
}

check nk-b2 269 -pm1 -b1 4 -b2 100 -nmax 60 -K 2
check nk-b2-lowmem 269 -pm1 -b1 4 -b2 100 -nmax 60 -K 2 -pm1-lowmem
echo "pm1 n^K stage-2 with -b2 test passed"
