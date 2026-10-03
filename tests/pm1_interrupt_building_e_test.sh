#!/usr/bin/env bash
# Ctrl-C while P-1 is still building the exponent E must stop the run.  It used
# to clear the interrupt, carry on with the partial E and report the full B1.
# The run is given SIGINT 2 s in (E for B1=3e8 takes much longer) and must exit
# on its own, before starting a chunk and without writing a checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$ROOT/kernels" "$WORK/kernels"
cd "$WORK"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# -maxe 64 builds E in one piece, -maxe 8 in chunks.
for maxe in 64 8; do
  log="maxe$maxe.log"
  rc=0
  timeout --signal=INT --kill-after=15s 2 "$ROOT/prmers" 269 -pm1 -b1 300000000 -maxe "$maxe" -d "$DEVICE" --noask >"$log" 2>&1 || rc=$?
  if [ "$rc" -ne 124 ]; then
    echo "-maxe $maxe: run did not stop after SIGINT (rc=$rc)" >&2; exit 1
  fi
  grep -q 'Interrupted by user while building E' "$log" || { echo "-maxe $maxe: no interrupt message" >&2; exit 1; }
  if grep -q '^Chunk ' "$log" || grep -q 'partial E' "$log"; then
    echo "-maxe $maxe: run continued with a partial E" >&2; exit 1
  fi
  if ls pm1_m_269.ckpt* >/dev/null 2>&1; then
    echo "-maxe $maxe: a checkpoint was written for a partial E" >&2; exit 1
  fi
done
echo "pm1 interrupt while building E test passed"
