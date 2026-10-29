#!/usr/bin/env bash
# The P-1 checkpoints (stage 1 and every stage-2 variant: classic BSGS, V-trace, pair95) live
# under the save path (-f), with the other state of the run, not in the working directory.
# Like the LL-SAFE/ECM checkpoints (ckpt_save_path_test.sh):
#   1. an interrupted run leaves its checkpoints under -f and none in the working directory;
#   2. the same command resumes them;
#   3. a run that finds none under -f but some in the working directory (left by an older
#      version) resumes from them, says so, and once the job is done no checkpoint is left in
#      either place.
# M521 has no factor, so stage 2 runs until it is interrupted.  Run on PoCL or any OpenCL device.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off PRMERS_PM1_CLASSIC_D=6

FAILURES=0
fail() { echo "FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
flat() { tr '\r' '\n' <"$1" | sed 's/\x1b\[[0-9;]*m//g'; }
has() { compgen -G "$1" >/dev/null; }   # has <glob>
# run <dir> <log> <seconds> <args...>: prmers in <dir>, interrupted (SIGINT) after <seconds>
run() {
  local dir=$1 log=$2 secs=$3; shift 3
  ( cd "$dir" && timeout --signal=INT --kill-after=10s "$secs" "$BIN" "$@" -d "$DEVICE" --noask >"$log" 2>&1 ) || true
}
first_pct() { flat "$1" | grep -ao 'Chunk 1/1 [0-9.]*%' | head -1 | grep -ao '[0-9.]*' | head -1; }

# ---- stage 1 ----
S1=(521 -pm1 -b1 400000 -b2 0)
d="$WORK/s1"; mkdir -p "$d"; ln -s "$ROOT/kernels" "$d/kernels"
run "$d" a.log 8 "${S1[@]}" -f "$d/save"
has "$d/save/pm1_m_521.ckpt" || fail "s1: no stage-1 checkpoint under -f after an interrupt"
has "$d/pm1_m_521.ckpt*" && fail "s1: stage-1 checkpoint written to the working directory"
run "$d" b.log 8 "${S1[@]}" -f "$d/save"
pct=$(first_pct "$d/b.log")
python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) > 0 else 1)" "${pct:-0}" || fail "s1: the run did not resume from the checkpoint under -f (first progress ${pct:-none}%)"
flat "$d/b.log" | grep -q 'from the working directory' && fail "s1: used the working directory although -f holds a checkpoint"
# an older version's checkpoint in the working directory
mv "$d"/save/pm1_m_521.ckpt* "$d"/
run "$d" c.log 300 "${S1[@]}" -f "$d/save2"
flat "$d/c.log" | grep -q "No checkpoint under .*save2; using pm1_m_521.ckpt from the working directory" || fail "s1: no note about the old checkpoint"
pct=$(first_pct "$d/c.log")
python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) > 0 else 1)" "${pct:-0}" || fail "s1: the old checkpoint was not resumed (first progress ${pct:-none}%)"
flat "$d/c.log" | grep -q 'No P-1 (stage 1) factor\|P-1 factor stage 1 found' || fail "s1: the job did not finish"
has "$d/pm1_m_521.ckpt*" && fail "s1: old checkpoint left in the working directory: $(ls "$d"/pm1_m_521.ckpt*)"
has "$d/save2/pm1_m_521.ckpt*" && fail "s1: checkpoint left under -f after the job finished"

# ---- stage 2 ----
# check_s2 <name> <checkpoint name> <resume message> <args...>
check_s2() {
  local name=$1 ck=$2 resumed=$3; shift 3
  local d="$WORK/$name"; mkdir -p "$d"; ln -s "$ROOT/kernels" "$d/kernels"
  run "$d" a.log 6 "$@" -f "$d/save"
  has "$d/save/$ck" || { fail "$name: no $ck under -f after an interrupt ($(ls "$d"/save 2>/dev/null | tr '\n' ' '))"; return; }
  has "$d/save/$ck.backend" || fail "$name: no backend sidecar next to $ck under -f"
  has "$d/save/pm1_m_521.ckpt" || fail "$name: no stage-1 checkpoint under -f"
  has "$d/pm1_*ckpt*" && fail "$name: checkpoint written to the working directory: $(ls "$d"/pm1_*ckpt*)"
  run "$d" b.log 10 "$@" -f "$d/save"
  flat "$d/b.log" | grep -q "$resumed" || fail "$name: the run did not resume stage 2 from the checkpoint under -f"
  flat "$d/b.log" | grep -q 'from the working directory' && fail "$name: used the working directory although -f holds the checkpoints"
  # an older version's checkpoints (stage 1 and stage 2) in the working directory
  mv "$d"/save/pm1_* "$d"/
  run "$d" c.log 300 "$@" -f "$d/save2"
  flat "$d/c.log" | grep -q "No checkpoint under .*save2; using $ck from the working directory" || fail "$name: no note about the old stage-2 checkpoint"
  flat "$d/c.log" | grep -q "$resumed" || fail "$name: the old stage-2 checkpoint was not resumed"
  flat "$d/c.log" | grep -q 'No factor P-1' || fail "$name: the job did not finish"
  has "$d/pm1_*ckpt*" && fail "$name: old checkpoint left in the working directory: $(ls "$d"/pm1_*ckpt*)"
  has "$d/save2/pm1_*ckpt*" && fail "$name: checkpoint left under -f after the job finished: $(ls "$d"/save2/pm1_*ckpt*)"
}
check_s2 classic pm1_s2_m_521.ckpt 'Resuming Stage 2 from checkpoint' \
  521 -pm1 -b1 10 -b2 3000000 -pm1-vtrace-off
check_s2 vtrace pm1_s2_vtrace_negadd_m_521.ckpt 'Resuming Stage 2 V-trace from checkpoint' \
  521 -pm1 -b1 10 -b2 3000000 -pm1-vtrace-pair95-off -pm1-vtrace-d 6
check_s2 pair95 pm1_s2_vtrace_pair95_m_521.ckpt 'Resuming Stage 2 V-trace from checkpoint' \
  521 -pm1 -b1 10 -b2 3000000

[ "$FAILURES" -eq 0 ] || exit 1
echo "P-1 checkpoint save-path test passed"
