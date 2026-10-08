#!/usr/bin/env bash
# The low-memory stage 2 reads H back from the stage-1 checkpoint.  It must check
# the CRC and the backend marker first: a damaged or foreign checkpoint must stop
# the run instead of feeding a wrong H to stage 2 (which then reports "no factor").
#   M269: 13822297 = 2*269*(2^2*3*2141) + 1, found by stage 2 with B1=4, B2=2141.
# Stage 1 hands over to stage 2 after a 1 s pause, which leaves time to damage the
# checkpoint file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# run <name> <none|corrupt|backend>
run() {
  local name="$1" damage="$2"
  mkdir -p "$WORK/$name"
  cd "$WORK/$name"
  ln -s "$ROOT/kernels" kernels
  timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 269 -pm1 -b1 4 -b2 2141 -pm1-lowmem \
    -d "$DEVICE" --noask >run.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 1500); do
    grep -aq 'waiting briefly for driver VRAM retirement' run.log && break
    sleep 0.02
  done
  [ -f pm1_m_269.ckpt ] || { echo "$name: no stage-1 checkpoint at the hand-over" >&2; kill "$pid" 2>/dev/null || true; exit 1; }
  case "$damage" in
    corrupt) printf '\377' | dd of=pm1_m_269.ckpt bs=1 seek=100 conv=notrunc 2>/dev/null ;;
    backend)
      [ -f pm1_m_269.ckpt.backend ] || { echo "$name: no backend marker" >&2; kill "$pid" 2>/dev/null || true; exit 1; }
      if grep -q aevum pm1_m_269.ckpt.backend; then echo marin >pm1_m_269.ckpt.backend; else echo aevum >pm1_m_269.ckpt.backend; fi ;;
  esac
  wait "$pid" || true
  cd "$WORK"
}

run control none
grep -q 'P-1 factor stage 2 found: 13822297' "$WORK/control/run.log" || { echo "control: factor not found" >&2; exit 1; }

run corrupt corrupt
grep -q 'cannot load PM1 Stage 1 checkpoint' "$WORK/corrupt/run.log" || { echo "corrupt: damaged checkpoint was accepted" >&2; exit 1; }
if grep -q 'Low-memory Stage 2 loaded H' "$WORK/corrupt/run.log"; then echo "corrupt: H was loaded" >&2; exit 1; fi

run backend backend
grep -q 'Stage 1 checkpoint ignored: checkpoint backend is' "$WORK/backend/run.log" || { echo "backend: foreign checkpoint was accepted" >&2; exit 1; }
if grep -q 'Low-memory Stage 2 loaded H' "$WORK/backend/run.log"; then echo "backend: H was loaded" >&2; exit 1; fi
echo "pm1 low-memory stage-1 checkpoint test passed"
