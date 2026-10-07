#!/usr/bin/env bash
# A GMTF worktodo line is an ordinary queue entry: it runs only when it is the
# first actionable line, is archived to worktodo_save.txt when it finishes, and
# PrMers restarts for the next line.
#   queue: two GMTF lines run back to back and both end up archived.
#   order: a PRP line in front of a GMTF line runs (and is archived) first.
#   modeflag: a mode flag on the command line (the GUI's generated settings always name one) does not
#             keep a GMTF line on top of the worktodo from running.
#   unrunnable: a line nothing can run behind the last GMTF entry does not send the restart into the
#               "no valid entry" prompt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

run() {
  local name="$1"; shift
  mkdir -p "$WORK/$name"
  printf '%s\n' "$@" > "$WORK/$name/worktodo.txt"
  ( cd "$WORK/$name" && ln -s "$ROOT/kernels" kernels &&
    timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" -d "$DEVICE" ${EXTRA_ARGS:-} --noask >run.log 2>&1 || true )
}

run queue 'GMTF=1009,20,24' 'GMTF=1031,20,24'
[[ ! -s "$WORK/queue/worktodo.txt" ]] || { echo "queue: worktodo.txt not drained" >&2; cat "$WORK/queue/worktodo.txt" >&2; exit 1; }
grep -qx 'GMTF=1009,20,24' "$WORK/queue/worktodo_save.txt" || { echo "queue: first GMTF not archived" >&2; exit 1; }
grep -qx 'GMTF=1031,20,24' "$WORK/queue/worktodo_save.txt" || { echo "queue: second GMTF not archived" >&2; exit 1; }
[[ -f "$WORK/queue/gm_tf_p1009_20_24_BOTH_result.json" && -f "$WORK/queue/gm_tf_p1031_20_24_BOTH_result.json" ]] ||
  { echo "queue: missing TF result files" >&2; exit 1; }

run order 'PRP=N/A,1,2,9941,-1,70,0' 'GMTF=1031,20,24'
[[ ! -s "$WORK/order/worktodo.txt" ]] || { echo "order: worktodo.txt not drained" >&2; exit 1; }
[[ -f "$WORK/order/9941_prp_result.json" ]] || { echo "order: PRP line did not run" >&2; exit 1; }
[[ "$(head -1 "$WORK/order/worktodo_save.txt")" == PRP=* ]] || { echo "order: GMTF was archived before the PRP line" >&2; exit 1; }
EXTRA_ARGS="-prp" run modeflag 'GMTF=1009,20,24' 'PRP=N/A,1,2,9941,-1,70,0'
[[ ! -s "$WORK/modeflag/worktodo.txt" ]] || { echo "modeflag: worktodo.txt not drained" >&2; cat "$WORK/modeflag/worktodo.txt" >&2; exit 1; }
[[ "$(head -1 "$WORK/modeflag/worktodo_save.txt")" == GMTF=* ]] || { echo "modeflag: GMTF line was skipped" >&2; exit 1; }
[[ -f "$WORK/modeflag/gm_tf_p1009_20_24_BOTH_result.json" && -f "$WORK/modeflag/9941_prp_result.json" ]] ||
  { echo "modeflag: missing result files" >&2; exit 1; }

run unrunnable 'GMTF=1009,20,24' 'Pfactor=N/A,1,2,127,-1,70,0'
grep -qx 'GMTF=1009,20,24' "$WORK/unrunnable/worktodo_save.txt" || { echo "unrunnable: GMTF not archived" >&2; exit 1; }
if grep -q 'No valid entry found\|Restarting for next worktodo entry' "$WORK/unrunnable/run.log"; then
  echo "unrunnable: restarted although nothing can run" >&2; exit 1
fi
echo "gmtf worktodo queue test passed"
