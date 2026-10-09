#!/usr/bin/env bash
# A default fresh -gm-pm1 run with B2 > B1 uses the V-trace driver. Its Stage 1
# must write the legacy `_stage1.ckpt` checkpoint on a SIGINT, and a restart
# must resume from it through the legacy path instead of starting Stage 1 over.
# GM_20011 has the factor 3922157 = 4*20011*7^2 + 1, found by Stage 1; a resumed
# residue that was not restored exactly cannot produce it.  Every B1 below finds
# the same factor, since 7^2 is covered by B1 >= 49.
#
# The interrupts are sent as soon as the run has reached its Stage 1 loop (a log
# line printed just before it), not after a fixed time, so they land at the same
# point of the run on a slow or a fast device.  The run is then required to have
# been interrupted for real: the checkpoint is reported and written, and the
# resumed run advances past it.  If a run finishes before the interrupt lands,
# the test escalates to a larger B1 (more Stage 1 work); if the interrupt lands
# before the loop has made any progress, the attempt is repeated with a growing
# delay.  The test fails, rather than passing silently, if no interrupt of a run
# that was still in Stage 1 could be achieved.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

B1S=(30000 100000 300000 1000000)   # B2 = B1 + 500
TRIES=4                             # attempts per B1 for run 1
TRIES2=6                            # attempts for run 2 from the same checkpoint
VTRACE_MARK="Stage 1 backend:"      # V-trace driver, right before its Stage 1 loop
LEGACY_MARK="Stage 2 power  :"      # legacy driver, right before its Stage 1 loop
MIN_SAVED=8192                      # bits that must still remain after run 1, so run 2 has work

fail() { echo "$1" >&2; exit 1; }

# interrupt_run DIR LOG MARKER DELAY ARGS...: run prmers in the background, send
# SIGINT once MARKER is in the log (plus DELAY seconds), and wait for it to exit.
# The 900 s limit only keeps a hung run from blocking the test forever.
interrupt_run() {
  local dir="$1" log="$2" marker="$3" delay="$4"; shift 4
  ( cd "$dir" && exec timeout --signal=INT --kill-after=10s 900 "$ROOT/prmers" "$@" >"$log" 2>&1 ) &
  local pid=$!
  until grep -aq -- "$marker" "$dir/$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.005
  done
  [[ "$delay" == 0 ]] || sleep "$delay"
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" || true
}

remaining_in() { sed -n 's/.*checkpoint saved with \([0-9]*\) bits remaining.*/\1/p' "$1" | head -1; }
finished_in() { grep -aq "Stage 1 factor:" "$1"; }

n=0
dir=""
saved=""
for b1 in "${B1S[@]}"; do
  ARGS=(20011 -gm-pm1 -gm-family GM -b1 "$b1" -b2 $((b1 + 500)) -gm-sieve 0 -d "$DEVICE" --noask)
  CKPT=gm_pm1_p20011_stage1.ckpt
  for try in $(seq 1 "$TRIES"); do
    n=$((n + 1))
    dir="$WORK/run$n"
    mkdir "$dir"
    ln -s "$ROOT/kernels" "$dir/kernels"
    delay=$(awk -v t="$try" 'BEGIN { printf "%.2f", (t - 1) * 0.1 }')

    # Run 1: interrupt in the V-trace Stage 1.
    interrupt_run "$dir" run1.log "$VTRACE_MARK" "$delay" "${ARGS[@]}"
    grep -aq "Gaussian pair P-1 factoring v100.13 V-trace" "$dir/run1.log" || fail "run 1 did not use the V-trace driver"
    if finished_in "$dir/run1.log"; then
      echo "B1=$b1: run 1 finished Stage 1 before the interrupt landed, using a larger B1" >&2
      saved=""
      break
    fi
    saved="$(remaining_in "$dir/run1.log")"
    if [[ -z "$saved" || ! -s "$dir/$CKPT" ]]; then
      echo "B1=$b1 try $try: the interrupt landed before Stage 1 made progress (no checkpoint reported), retrying" >&2
      saved=""
      continue
    fi
    if (( saved < MIN_SAVED )); then
      echo "B1=$b1 try $try: only $saved bits remain after run 1, too few to resume from, using a larger B1" >&2
      saved=""
      break
    fi

    # Run 2: the legacy path picks the checkpoint up and continues from it; interrupt again.
    # An interrupt that lands before the resumed loop advanced leaves the same
    # checkpoint behind, so that attempt is repeated from it.
    resumed=""
    for try2 in $(seq 1 "$TRIES2"); do
      delay2=$(awk -v t="$try2" 'BEGIN { printf "%.2f", (t - 1) * 0.1 }')
      interrupt_run "$dir" run2.log "$LEGACY_MARK" "$delay2" "${ARGS[@]}"
      grep -aq "legacy checkpoint detected" "$dir/run2.log" || fail "run 2: legacy checkpoint not detected"
      if grep -aq "Gaussian pair P-1 factoring v100.13 V-trace" "$dir/run2.log"; then fail "run 2 restarted in the V-trace driver"; fi
      if finished_in "$dir/run2.log"; then
        echo "B1=$b1: run 2 finished Stage 1 before the interrupt landed" >&2
        break
      fi
      resumed="$(remaining_in "$dir/run2.log")"
      [[ -n "$resumed" ]] || { tail -5 "$dir/run2.log" >&2; fail "run 2: no checkpoint was reported although it was interrupted in Stage 1"; }
      (( resumed <= saved )) || fail "run 2 did not resume: $resumed bits remaining after run 1 left $saved"
      if (( resumed < saved )); then break; fi
      echo "B1=$b1 try $try: run 2 was interrupted before it advanced ($resumed bits remaining), retrying" >&2
      resumed=""
    done
    if [[ -n "$resumed" ]]; then break 2; fi
    saved=""
    if finished_in "$dir/run2.log"; then break; fi   # larger B1
    fail "run 2 never advanced past the checkpoint in $TRIES2 attempts"
  done
done
[[ -n "$saved" && -n "${resumed:-}" ]] || fail "could not interrupt the Stage 1 of a P-1 run and resume it with any B1 up to ${B1S[-1]}"
echo "interrupted at B1=$b1: run 1 left $saved bits remaining, run 2 left $resumed"

# Run 3: finish; the resumed residue must still give the Stage 1 factor.
cd "$dir"
timeout --signal=INT --kill-after=10s 900 "$ROOT/prmers" "${ARGS[@]}" >run3.log 2>&1 || true
grep -aq "Gaussian pair P-1 Stage 1 factor: 3922157" run3.log || { echo "run 3: Stage 1 factor 3922157 not found" >&2; tail -5 run3.log >&2; exit 1; }
[[ ! -e "$CKPT" ]] || { echo "run 3: Stage 1 checkpoint not removed" >&2; exit 1; }
echo "gm pm1 v-trace checkpoint test passed ($saved -> $resumed bits remaining)"
