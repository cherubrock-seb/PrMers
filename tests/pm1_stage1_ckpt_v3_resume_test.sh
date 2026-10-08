#!/usr/bin/env bash
# P-1 stage-1 resume with the 64-bit counter checkpoint (version 4) and with an
# old 32-bit one (version 3).  M269, B1=60000: an interrupted run leaves a
# checkpoint; resuming from it must give the same stage-1 residue (the X= field of
# the .save file) as an uninterrupted run, for
#   * the checkpoint the program writes now (version 4),
#   * the same checkpoint rewritten in the old format (version 3),
#   * hostile files: a resume position beyond the chunk (stored in 64 bits) and a
#     truncated file; both must be refused and stage 1 must start over, still
#     reaching the same residue.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
B1=60000
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off
TOOL="$ROOT/tests/pm1_ckpt_tool.py"

newdir() { mkdir -p "$WORK/$1"; ln -s "$ROOT/kernels" "$WORK/$1/kernels"; }
residue() { grep -ao 'X=0x[0-9a-f]*' "$1/resume_p269_B1_$B1.save"; }
plain() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

# Control: uninterrupted run.
newdir control
START=$(date +%s)
( cd "$WORK/control" && timeout -k 10 300 "$ROOT/prmers" 269 -pm1 -b1 $B1 -d "$DEVICE" --noask >run.log 2>&1 )
ELAPSED=$(( $(date +%s) - START ))
CONTROL="$(residue "$WORK/control")"
[ -n "$CONTROL" ] || { echo "control: no stage-1 residue" >&2; exit 1; }
BITS=$(grep -ao 'bits=[0-9]*' "$WORK/control/run.log" | head -1 | cut -d= -f2)
[ -n "$BITS" ] || { echo "control: no chunk size in the log" >&2; exit 1; }
CUT=$(( ELAPSED / 2 )); [ "$CUT" -ge 2 ] || CUT=2

# interrupt <dir>: leave a checkpoint behind by interrupting a run half way.
interrupt() {
  newdir "$1"
  ( cd "$WORK/$1" && timeout --signal=INT -k 10 "$CUT" "$ROOT/prmers" 269 -pm1 -b1 $B1 -d "$DEVICE" --noask >first.log 2>&1 || true )
  [ -f "$WORK/$1/pm1_m_269.ckpt" ] || { echo "$1: no stage-1 checkpoint was written" >&2; exit 1; }
}
resume() {
  ( cd "$WORK/$1" && timeout -k 10 300 "$ROOT/prmers" 269 -pm1 -b1 $B1 -d "$DEVICE" --noask >second.log 2>&1 || true )
}
# First chunk-progress percentage printed by the resumed run ("Chunk 1/1 NN.NN%").
first_pct() { plain "$WORK/$1/second.log" | grep -ao 'Chunk 1/1 [0-9.]*%' | head -1 | grep -ao '[0-9.]*%' | tr -d %; }
expect_pct() { python3 -I -c "import sys; b=$BITS; r=int(sys.argv[1]); print('%.2f' % ((b-r)*100.0/b))" "$1"; }

# 1. Version 4 (what is written now).
interrupt v4
read -r VER P COUNTER CRC < <(python3 -I "$TOOL" info "$WORK/v4/pm1_m_269.ckpt")
[ "$VER" = 4 ] && [ "$P" = 269 ] && [ "$CRC" = 1 ] || { echo "v4: unexpected checkpoint header: $VER $P $COUNTER $CRC" >&2; exit 1; }
[ "$COUNTER" -gt 0 ] && [ "$COUNTER" -lt "$BITS" ] || { echo "v4: counter $COUNTER outside (0,$BITS)" >&2; exit 1; }
cp "$WORK/v4/pm1_m_269.ckpt" "$WORK/saved.ckpt"
resume v4
[ "$(residue "$WORK/v4")" = "$CONTROL" ] || { echo "v4: resumed residue differs from the uninterrupted run" >&2; exit 1; }
[ "$(first_pct v4)" = "$(expect_pct "$COUNTER")" ] || { echo "v4: resumed at $(first_pct v4)%, expected $(expect_pct "$COUNTER")%" >&2; exit 1; }

# 2. The same checkpoint in the old 32-bit format.
newdir v3
cp "$WORK/saved.ckpt" "$WORK/v3/pm1_m_269.ckpt"
cp "$WORK/v4/pm1_m_269.ckpt.backend" "$WORK/v3/" 2>/dev/null || true
python3 -I "$TOOL" to-v3 "$WORK/v3/pm1_m_269.ckpt"
read -r VER _ C3 CRC < <(python3 -I "$TOOL" info "$WORK/v3/pm1_m_269.ckpt")
[ "$VER" = 3 ] && [ "$C3" = "$COUNTER" ] && [ "$CRC" = 1 ] || { echo "v3: conversion failed: $VER $C3 $CRC" >&2; exit 1; }
resume v3
[ "$(residue "$WORK/v3")" = "$CONTROL" ] || { echo "v3: residue differs after resuming an old-format checkpoint" >&2; exit 1; }
[ "$(first_pct v3)" = "$(expect_pct "$COUNTER")" ] || { echo "v3: resumed at $(first_pct v3)%, expected $(expect_pct "$COUNTER")%" >&2; exit 1; }

# 3. Hostile: resume position beyond the chunk (needs the 64-bit field).
for N in $(( BITS + 1 )) 4294967301 18446744073709551615; do
  d="beyond_$N"
  newdir "$d"
  cp "$WORK/saved.ckpt" "$WORK/$d/pm1_m_269.ckpt"
  cp "$WORK/v4/pm1_m_269.ckpt.backend" "$WORK/$d/" 2>/dev/null || true
  python3 -I "$TOOL" set-counter "$WORK/$d/pm1_m_269.ckpt" "$N"
  resume "$d"
  grep -aq "beyond this run's chunk of $BITS bits" "$WORK/$d/second.log" || { echo "$d: counter $N was not refused" >&2; exit 1; }
  [ "$(residue "$WORK/$d")" = "$CONTROL" ] || { echo "$d: residue differs after refusing the checkpoint" >&2; exit 1; }
done

# 4. Hostile: truncated checkpoints (version 4 and version 3).
for kind in v4 v3; do
  for LEN in 0 7 12 15 16 100 500; do
    d="trunc_${kind}_$LEN"
    newdir "$d"
    cp "$WORK/saved.ckpt" "$WORK/$d/pm1_m_269.ckpt"
    cp "$WORK/v4/pm1_m_269.ckpt.backend" "$WORK/$d/" 2>/dev/null || true
    [ "$kind" = v3 ] && python3 -I "$TOOL" to-v3 "$WORK/$d/pm1_m_269.ckpt"
    python3 -I "$TOOL" truncate "$WORK/$d/pm1_m_269.ckpt" "$LEN"
    # Only the cheap start-up needs to be seen: it must not resume (the first
    # progress line is at 0%), so stop the run early.
    ( cd "$WORK/$d" && timeout --signal=INT -k 10 3 "$ROOT/prmers" 269 -pm1 -b1 $B1 -d "$DEVICE" --noask >second.log 2>&1 || true )
    pct="$(first_pct "$d")"
    if [ -n "$pct" ] && [ "${pct%%.*}" -ge 20 ]; then
      echo "$d: truncated checkpoint was resumed at $pct%" >&2; exit 1
    fi
  done
done
echo "pm1 stage-1 v3/v4 checkpoint resume test passed"
