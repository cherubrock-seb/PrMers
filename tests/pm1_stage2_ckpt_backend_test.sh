#!/usr/bin/env bash
# Stage-2 checkpoints (classic BSGS and V-trace) carry a backend marker, like the
# stage-1 ones: a checkpoint written by the other backend must be ignored, not
# loaded into the engine, and a matching one must still resume.
# M521 has no factor, so stage 2 runs until it is interrupted.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off PRMERS_PM1_CLASSIC_D=6

# check <name> <checkpoint glob> <resume message> <ignored message> <prmers args...>
check() {
  local name="$1" glob="$2" resume_msg="$3" ignored_msg="$4"; shift 4
  mkdir -p "$WORK/$name/first"
  ( cd "$WORK/$name/first" && ln -s "$ROOT/kernels" kernels &&
    timeout --signal=INT --kill-after=10s 4 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >run.log 2>&1 || true )
  grep -q 'Stage 2 state saved at prime' "$WORK/$name/first/run.log" || { echo "$name: stage 2 was not interrupted" >&2; exit 1; }
  local ckpt
  ckpt="$(cd "$WORK/$name/first" && ls $glob 2>/dev/null | grep -E '\.ckpt$' | head -1)"
  [ -n "$ckpt" ] || { echo "$name: no stage-2 checkpoint" >&2; exit 1; }
  [ -f "$WORK/$name/first/$ckpt.backend" ] || { echo "$name: no backend marker for $ckpt" >&2; exit 1; }

  # Same backend: the checkpoint resumes.
  cp -a "$WORK/$name/first" "$WORK/$name/same"
  ( cd "$WORK/$name/same" &&
    timeout --signal=INT --kill-after=10s 10 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >resume.log 2>&1 || true )
  grep -q "$resume_msg" "$WORK/$name/same/resume.log" || { echo "$name: matching checkpoint did not resume" >&2; exit 1; }

  # Other backend: the checkpoint is ignored.
  cp -a "$WORK/$name/first" "$WORK/$name/other"
  if grep -q aevum "$WORK/$name/other/$ckpt.backend"; then echo marin >"$WORK/$name/other/$ckpt.backend"; else echo aevum >"$WORK/$name/other/$ckpt.backend"; fi
  ( cd "$WORK/$name/other" &&
    timeout --signal=INT --kill-after=10s 10 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >resume.log 2>&1 || true )
  grep -q "$ignored_msg" "$WORK/$name/other/resume.log" || { echo "$name: checkpoint of the other backend was not ignored" >&2; exit 1; }
  if grep -q "$resume_msg" "$WORK/$name/other/resume.log"; then echo "$name: checkpoint of the other backend was resumed" >&2; exit 1; fi
}

check classic 'pm1_s2_m_521*' 'Resuming Stage 2 from checkpoint' '\[PM1\] Stage 2 checkpoint ignored' \
  521 -pm1 -b1 10 -b2 3000000 -pm1-vtrace-off
check vtrace 'pm1_s2_vtrace_*' 'Resuming Stage 2 V-trace from checkpoint' '\[PM1-VTRACE\] Stage 2 checkpoint ignored' \
  521 -pm1 -b1 10 -b2 3000000 -pm1-vtrace-pair95-off -pm1-vtrace-d 6
echo "pm1 stage-2 checkpoint backend test passed"
