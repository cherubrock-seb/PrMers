#!/usr/bin/env bash
# The default P-1 stage 2 (V-trace with the pair95 planner) checkpoints its
# progress (pm1_s2_vtrace_pair95_m_<p>.ckpt) at the backup interval and on a
# stop, and a restart resumes from it.  The resumed run must give exactly the
# result of an uninterrupted run: the same factor, or "no factor" with the same
# stage-2 GCD input (its res64 is printed), and must delete the checkpoint once
# the job is finished.
#
#   M9941, B1=1000: no factor; the res64 of the GCD input is compared.
#   M677,  B1=10:   stage 2 finds 1943118631 = 2*677*45*31891 + 1, run with two
#                   baby windows (-pm1-vtrace-baby-batch 40) so a resume inside
#                   the second pass is covered too.
#
# The interrupts are sent on log markers (the line printed just before the
# stage-2 loop, or the start of the second pass), not after a fixed time, and the
# delays after the marker grow from attempt to attempt; a run that finishes
# before the interrupt lands is repeated with a larger B2.  The test fails if no
# interrupt inside stage 2 could be achieved.
#
# Damaged and foreign checkpoints: a truncated or corrupt file falls back to
# .old, then to a restart; a checkpoint for another B2, another exponent, another
# backend or an unknown version is never resumed, and each case says why.
#
# With gdb available, a run that is killed (SIGKILL) while it writes the
# stage-2 result line must not lose the factor: the rerun resumes at the GCD,
# records the factor once and retires the worktodo entry.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
KEEP=0
trap '[ "$KEEP" = 1 ] && echo "work dir kept: $WORK" >&2 || rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

LOOP_MARK="executing greedy irregular pair plan"
fail() { KEEP=1; echo "FAIL: $1" >&2; exit 1; }

# run DIR LOG ARGS...: run prmers to completion.
run() {
  local dir="$1" log="$2"; shift 2
  ( cd "$dir" && timeout --signal=INT --kill-after=10s 1200 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >"$log" 2>&1 ) || true
}

# interrupt_run DIR LOG MARKER DELAY ARGS...: run prmers, send SIGINT once MARKER
# is in the log (plus DELAY seconds) and wait for it to exit.
interrupt_run() {
  local dir="$1" log="$2" marker="$3" delay="$4"; shift 4
  ( cd "$dir" && exec timeout --signal=INT --kill-after=10s 1200 "$ROOT/prmers" "$@" -d "$DEVICE" --noask >"$log" 2>&1 ) &
  local pid=$!
  until grep -aq -- "$marker" "$dir/$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.005
  done
  [[ "$delay" == 0 ]] || sleep "$delay"
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" || true
}

newdir() { local d="$WORK/$1"; mkdir -p "$d"; ln -sfn "$ROOT/kernels" "$d/kernels"; echo "$d"; }
res64_of() { sed -n 's/.*Stage 2 GCD input res64=\([0-9a-f]*\) terms=\([0-9]*\).*/\1 \2/p' "$1" | tail -1; }
# "pass task terms" of the last "state saved" line, or empty
saved_at() { tr '\r' '\n' <"$1" | sed -n 's/.*pair95 Stage 2 state saved at pass \([0-9]*\)\/[0-9]* task \([0-9]*\) (terms done=\([0-9]*\)\/.*/\1 \2 \3/p' | tail -1; }
total_terms() { sed -n 's/.*PAIR95\] enabled: .*| terms=\([0-9]*\) .*/\1/p' "$1" | head -1; }
finished() { grep -aq "Stage 2 GCD input res64=" "$1"; }
s2_files() { ls "$1"/pm1_s2_* 2>/dev/null || true; }
# u64 field of a checkpoint header: 0 exponent .. 11 pass, 12 task position
ckpt_field() { od -An -t u8 -j $((4 + 8 * $2)) -N 8 "$1" | tr -d ' '; }

# interrupt_stage2 DIR LOG MIN MARKER ARGS...: interrupt DIR's run inside stage
# 2, retrying with a growing delay until a state with more than MIN terms done is
# saved (an attempt that stops earlier leaves a checkpoint that the next attempt
# resumes).  Prints "pass task terms", or "finished" when stage 2 completed
# before the interrupt landed.
interrupt_stage2() {
  local dir="$1" log="$2" min="$3" marker="$4"; shift 4
  local try delay at
  for try in 1 2 3 4 5 6 7 8; do
    delay=$(awk -v t="$try" 'BEGIN { printf "%.2f", (t - 1) * (t - 1) * 0.1 }')
    interrupt_run "$dir" "$log" "$marker" "$delay" "$@"
    if finished "$dir/$log"; then echo finished; return 0; fi
    at="$(saved_at "$dir/$log")"
    if [[ -n "$at" ]] && (( $(awk '{print $3}' <<<"$at") > min )); then echo "$at"; return 0; fi
    echo "  attempt $try: the interrupt landed before stage 2 got past $min terms (${at:-no state saved}), retrying" >&2
  done
  tail -5 "$dir/$log" >&2
  fail "$dir: no stage-2 state past $min terms was saved after 8 interrupts"
}

# ---------------------------------------------------------------------------
# 1. No factor (M9941): interrupt twice, resume, compare with an uninterrupted run.
# ---------------------------------------------------------------------------
B1=1000
found=""
for B2 in 1000000 4000000 16000000; do
  ARGS=(9941 -pm1 -b1 "$B1" -b2 "$B2" -t 1)
  base="$(newdir "nf-base-$B2")"
  run "$base" run.log "${ARGS[@]}"
  ref="$(res64_of "$base/run.log")"
  [[ -n "$ref" ]] || { tail -5 "$base/run.log" >&2; fail "M9941 B2=$B2: the uninterrupted run printed no GCD input"; }
  grep -aq "No factor P-1 (stage 2 V-trace) until B2 = $B2" "$base/run.log" || fail "M9941: unexpected stage-2 result"
  terms="$(total_terms "$base/run.log")"

  dir="$(newdir "nf-$B2")"
  a="$(interrupt_stage2 "$dir" run1.log 0 "$LOOP_MARK" "${ARGS[@]}")"
  if [[ "$a" == finished ]]; then echo "B2=$B2: stage 2 finished before the first interrupt, using a larger B2"; continue; fi
  read -r a_pass a_task a_terms <<<"$a"
  [[ -s "$dir/pm1_s2_vtrace_pair95_m_9941.ckpt" ]] || fail "run 1: no pair95 checkpoint after the interrupt"
  grep -aq "Stage 2 interrupted; keeping the checkpoint" "$dir/run1.log" || fail "run 1: the job was not kept"
  (( a_terms > 0 && a_terms < terms )) || { echo "B2=$B2: run 1 saved at $a_terms of $terms terms, using a larger B2"; continue; }

  b="$(interrupt_stage2 "$dir" run2.log "$a_terms" "$LOOP_MARK" "${ARGS[@]}")"
  grep -aq "Resuming Stage 2 V-trace from checkpoint at pass $a_pass/1, task $a_task/" "$dir/run2.log" \
    || { tail -5 "$dir/run2.log" >&2; fail "run 2 did not resume at the saved position (pass $a_pass task $a_task)"; }
  if [[ "$b" == finished ]]; then echo "B2=$B2: run 2 finished before its interrupt, using a larger B2"; continue; fi
  read -r b_pass b_task b_terms <<<"$b"
  (( b_terms > a_terms )) || fail "run 2 did not advance ($b_terms terms after $a_terms)"
  cp -a "$dir" "$WORK/nf-snapshot"

  run "$dir" run3.log "${ARGS[@]}"
  grep -aq "Resuming Stage 2 V-trace from checkpoint at pass $b_pass/1, task $b_task/" "$dir/run3.log" || fail "run 3 did not resume"
  got="$(res64_of "$dir/run3.log")"
  [[ "$got" == "$ref" ]] || fail "resumed GCD input '$got' differs from the uninterrupted '$ref'"
  grep -aq "No factor P-1 (stage 2 V-trace) until B2 = $B2" "$dir/run3.log" || fail "run 3: unexpected stage-2 result"
  [[ -z "$(s2_files "$dir")" ]] || fail "run 3 left stage-2 checkpoint files: $(s2_files "$dir")"
  [[ ! -e "$dir/pm1_m_9941.ckpt" ]] || fail "run 3 left the stage-1 checkpoint"
  n="$(grep -c "\"b2\":$B2" "$dir/results.txt" || true)"
  [[ "$n" == 1 ]] || fail "expected one stage-2 result line, found $n"
  echo "M9941 B2=$B2: interrupted at terms $a_terms and $b_terms of $terms, resumed GCD input $got matches"
  found="$B2"
  break
done
[[ -n "$found" ]] || fail "could not interrupt the M9941 stage 2 twice with any B2"
SNAP="$WORK/nf-snapshot"
CK=pm1_s2_vtrace_pair95_m_9941.ckpt
[[ -s "$SNAP/$CK" && -s "$SNAP/$CK.old" ]] || fail "snapshot lacks $CK or $CK.old"

# ---------------------------------------------------------------------------
# 2. Damaged and foreign checkpoints (each on a copy of the snapshot).
# ---------------------------------------------------------------------------
# check_case NAME EXPECT_RESUME(yes|no) MSG ARGS...: run the copy until stage 2
# starts its loop, then stop it, and check the decision.
check_case() {
  local name="$1" resume="$2" msg="$3"; shift 3
  local d="$WORK/case-$name"
  interrupt_run "$d" run.log "$LOOP_MARK" 0 "$@"
  grep -aqF -- "$msg" "$d/run.log" || { tail -20 "$d/run.log" >&2; fail "$name: missing message '$msg'"; }
  if [[ "$resume" == yes ]]; then
    grep -aq "Resuming Stage 2 V-trace from checkpoint" "$d/run.log" || fail "$name: did not resume"
  else
    if grep -aq "Resuming Stage 2 V-trace from checkpoint" "$d/run.log"; then fail "$name: resumed a checkpoint it must reject"; fi
    grep -aq "Stage 2 restarts from the beginning" "$d/run.log" || fail "$name: no restart message"
  fi
  echo "case $name: ok"
}
copy_case() { cp -a "$SNAP" "$WORK/case-$1"; }
ARGS=(9941 -pm1 -b1 "$B1" -b2 "$found" -t 1)
old_task="$(ckpt_field "$SNAP/$CK.old" 12)"

copy_case truncated
truncate -s 200 "$WORK/case-truncated/$CK"
check_case truncated yes "resuming from the previous checkpoint $CK.old" "${ARGS[@]}"
grep -aq "task $old_task/" "$WORK/case-truncated/run.log" || fail "truncated: did not resume at the .old position $old_task"

copy_case crc
size=$(stat -c %s "$WORK/case-crc/$CK")
byte=$(od -An -t u1 -j $((size / 2)) -N 1 "$WORK/case-crc/$CK" | tr -d ' ')
printf "\\x$(printf %02x $((byte ^ 0xff)))" | dd of="$WORK/case-crc/$CK" bs=1 seek=$((size / 2)) conv=notrunc status=none
check_case crc yes "fails its CRC check" "${ARGS[@]}"
grep -aq "resuming from the previous checkpoint" "$WORK/case-crc/run.log" || fail "crc: no fallback to .old"

copy_case both-bad
truncate -s 50 "$WORK/case-both-bad/$CK"
truncate -s 0 "$WORK/case-both-bad/$CK.old"
check_case both-bad no "is truncated" "${ARGS[@]}"

copy_case version
printf '\x63\x00\x00\x00' | dd of="$WORK/case-version/$CK" bs=1 seek=0 conv=notrunc status=none
rm -f "$WORK/case-version/$CK.old"
check_case version no "has unknown version 99" "${ARGS[@]}"

copy_case other-b2
check_case other-b2 no "B2 differs (checkpoint $found, this run $((found + 1000)))" \
  9941 -pm1 -b1 "$B1" -b2 $((found + 1000)) -t 1

copy_case other-exponent
for f in "$WORK/case-other-exponent"/pm1_s2_vtrace_pair95_m_9941.ckpt*; do
  mv "$f" "${f/_m_9941/_m_9967}"
done
check_case other-exponent no "exponent differs (checkpoint 9941, this run 9967)" \
  9967 -pm1 -b1 "$B1" -b2 "$found" -t 1

copy_case other-backend
sidecar="$WORK/case-other-backend/$CK.backend"
if grep -q aevum "$sidecar"; then echo marin >"$sidecar"; else echo aevum >"$sidecar"; fi
check_case other-backend no "ignored: checkpoint backend is" "${ARGS[@]}"


# ---------------------------------------------------------------------------
# 3. A stage-2 factor (M677) with two baby windows: interrupt in the first and
#    in the second pass, resume, find the same factor.
# ---------------------------------------------------------------------------
FACTOR=1943118631
done677=""
for B2 in 2000000 8000000 32000000; do
  ARGS=(677 -pm1 -b1 10 -b2 "$B2" -pm1-vtrace-baby-batch 40 -t 1)
  base="$(newdir "f-base-$B2")"
  run "$base" run.log "${ARGS[@]}"
  grep -aq "P-1 factor stage 2 found: $FACTOR" "$base/run.log" || { tail -5 "$base/run.log" >&2; fail "M677: uninterrupted run missed $FACTOR"; }
  grep -aq "PAIR95\] pass 2/2" "$base/run.log" || fail "M677: the run did not use two baby windows"
  ref="$(res64_of "$base/run.log")"

  dir="$(newdir "f-$B2")"
  a="$(interrupt_stage2 "$dir" run1.log 0 "$LOOP_MARK" "${ARGS[@]}")"
  if [[ "$a" == finished ]]; then echo "M677 B2=$B2: finished before the interrupt, using a larger B2"; continue; fi
  read -r a_pass a_task a_terms <<<"$a"
  [[ "$a_pass" == 1 ]] || { echo "M677 B2=$B2: first interrupt landed in pass $a_pass, using a larger B2"; continue; }
  # The second interrupt must land inside pass 2 (past the tasks of pass 1), so
  # the resume rebuilds the second baby window and keeps its giant step.
  pass1="$(sed -n 's/.*PAIR95\] pass 1\/2 babies .*queued-tasks=\([0-9]*\).*/\1/p' "$base/run.log" | head -1)"
  [[ -n "$pass1" ]] || fail "M677: no task count for pass 1"
  b="$(interrupt_stage2 "$dir" run2.log "$pass1" "PAIR95] pass 2/2" "${ARGS[@]}")"
  # (a retry resumes from the previous attempt, so the position varies)
  grep -aq "Resuming Stage 2 V-trace from checkpoint at pass [12]/2, task " "$dir/run2.log" || fail "M677 run 2 did not resume"
  if [[ "$b" == finished ]]; then echo "M677 B2=$B2: second pass finished before the interrupt, using a larger B2"; continue; fi
  read -r b_pass b_task b_terms <<<"$b"
  [[ "$b_pass" == 2 ]] || fail "M677: second interrupt saved pass $b_pass, expected 2"
  run "$dir" run3.log "${ARGS[@]}"
  grep -aq "Resuming Stage 2 V-trace from checkpoint at pass 2/2, task $b_task/" "$dir/run3.log" || fail "M677 run 3 did not resume in pass 2"
  grep -aq "P-1 factor stage 2 found: $FACTOR" "$dir/run3.log" || fail "M677: the resumed run missed $FACTOR"
  [[ "$(res64_of "$dir/run3.log")" == "$ref" ]] || fail "M677: resumed GCD input differs"
  grep -q "\"factors\":\[\"$FACTOR\"\]" "$dir/results.txt" || fail "M677: factor not in results.txt"
  [[ -z "$(s2_files "$dir")" ]] || fail "M677: stage-2 checkpoint files left: $(s2_files "$dir")"
  echo "M677 B2=$B2: interrupted in pass 1 (task $a_task) and pass 2 (task $b_task), factor $FACTOR found after resume"
  done677="$B2"
  break
done
[[ -n "$done677" ]] || fail "could not interrupt the M677 stage 2 in both passes"

# ---------------------------------------------------------------------------
# 4. Killed while the stage-2 result is written (needs gdb): the factor is not lost.
# ---------------------------------------------------------------------------
if ! command -v gdb >/dev/null 2>&1; then
  echo "SKIPPED kill-before-result case: gdb is not installed"
else
  d="$(newdir kill)"
  printf 'Pminus1=1,2,677,-1,10,%s\n' "$done677" >"$d/worktodo.txt"
  cat >"$d/hook.py" <<'EOF'
import gdb
for cmd in ("set pagination off", "set confirm off", "set startup-with-shell off",
            "set breakpoint pending on", "set print thread-events off", "set print inferior-events off"):
    gdb.execute(cmd)
class Hook(gdb.Breakpoint):
    seen = 0
    def stop(self):
        arch = str(gdb.selected_frame().architecture().name())
        reg = "$x0" if "aarch64" in arch else "$rdi"
        path = gdb.parse_and_eval("(char*)" + reg).string("utf-8", "replace")
        if path.endswith("results.txt"):
            Hook.seen += 1
            return Hook.seen == 2   # 1: stage-1 line, 2: stage-2 line
        return False
Hook("fopen", internal=True)
gdb.execute("run")
if Hook.seen >= 2:
    print("HOOK-KILLED")
    gdb.execute("kill")
EOF
  ( cd "$d" && timeout --kill-after=10s 1200 gdb -q -batch -x hook.py --args "$ROOT/prmers" -pm1-vtrace-baby-batch 40 -t 1 -d "$DEVICE" --noask >kill.log 2>&1 ) || true
  if ! grep -aq HOOK-KILLED "$d/kill.log"; then
    if grep -aqi "ptrace\|not permitted" "$d/kill.log"; then
      echo "SKIPPED kill-before-result case: gdb cannot trace here"
    else
      tail -5 "$d/kill.log" >&2; fail "kill case: the run was not stopped at the stage-2 result write"
    fi
  else
    grep -aq "P-1 factor stage 2 found: $FACTOR" "$d/kill.log" || fail "kill case: stage 2 did not find the factor before the kill"
    if grep -q "$FACTOR" "$d/results.txt" 2>/dev/null; then fail "kill case: the result was written before the kill"; fi
    [[ -s "$d/pm1_s2_vtrace_pair95_m_677.ckpt" ]] || fail "kill case: the stage-2 checkpoint is gone although the result was not written"
    grep -q '^Pminus1=1,2,677,' "$d/worktodo.txt" || fail "kill case: the worktodo entry is gone"
    run "$d" rerun.log -pm1-vtrace-baby-batch 40 -t 1
    grep -aq "Resuming Stage 2 V-trace from checkpoint after the last pass" "$d/rerun.log" || fail "kill case: the rerun did not resume at the GCD"
    grep -aq "P-1 factor stage 2 found: $FACTOR" "$d/rerun.log" || fail "kill case: the rerun missed the factor"
    n="$(grep -c "\"factors\":\[\"$FACTOR\"\]" "$d/results.txt" || true)"
    [[ "$n" == 1 ]] || fail "kill case: expected the factor line once in results.txt, found $n"
    if grep -q '^Pminus1=1,2,677,' "$d/worktodo.txt" 2>/dev/null; then fail "kill case: the entry was not retired"; fi
    [[ -z "$(s2_files "$d")" ]] || fail "kill case: stage-2 checkpoint files left: $(s2_files "$d")"
    echo "kill case: factor recorded once after the rerun, entry retired, checkpoint removed"
  fi
fi
echo "pm1 V-trace pair95 resume test passed"
