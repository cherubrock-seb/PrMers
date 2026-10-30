#!/usr/bin/env bash
# End-to-end proof-generation failure handling (device test; slow: one full PRP of 2^400009-1, then restarts from copies).
#
#   bash tests/proof_retry_gpu_test.sh [aevum binary] [baseline binary]
#
# A. Every proof attempt fails as a GPU error (AEVUM_TEST_PROOF_FAILURES=99).  Expected: every attempt is made (power 6 twice,
#    5, then 4 unverified), the worker stops without reporting the result, and the worktodo line, savefiles and proof residues
#    are kept.
# W. A's directory plus a second task, two workers, failures again: the worker that finishes PRP 400009 stops without
#    reporting, and the stop reaches the other worker too.
# B. Restart A's directory without the hook: the PRP resumes, the proof is generated and the result is reported with it.
# C. A's directory with the middle proof residue (needed by every power) corrupted: every attempt fails on a file error, and
#    the result is reported without a proof.  With a baseline binary, also shows what the baseline does.
# D. A's directory with one injected failure: the retry at the same power succeeds.
# Z. A's directory with the middle proof residue deleted: proof generation is disabled and the result is reported without a proof.
#
# AEVUM_TEST_DEVICE selects the device (default 0).  Work directory: $AEVUM_TEST_WORK or a new temporary directory; A is reused
# when $AEVUM_TEST_WORK/a/.done exists.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$(realpath "${1:-$ROOT/build-release/aevum}")"
BASE="${2:+$(realpath -m "$2")}"
E=400009
E2=500009          # the second task in W; far from its end when the first worker reaches its proof
FFT=0:256:1:256:101
POWER=6
MID=$(( (E + 1) / 2 ))
DEV="${AEVUM_TEST_DEVICE:-0}"
WORK="${AEVUM_TEST_WORK:-$(mktemp -d)}"
mkdir -p "$WORK"
fail=0
bad() { echo "FAIL: $*"; fail=1; }

run() {   # run <dir> <args...>: aevum in <dir>, output to <dir>/out.txt; prints the exit code
  local dir="$1"; shift
  (cd "$dir" && "$BIN_RUN" -dir . -device "$DEV" -fft "$FFT" -proof $POWER -noclean "$@" > out.txt 2>&1); echo $?
}
results() { cat "$1"/results-*.txt 2>/dev/null; }
logs() { cat "$1"/aevum*.log 2>/dev/null; }

prepare() {   # prepare <name> [task 2]: a copy of A's final state, its task in worktodo-0.txt (and task 2 in worktodo-1.txt)
  local d="$WORK/$1"
  rm -rf "$d"; cp -a "$WORK/a" "$d"; rm -f "$d"/.done "$d"/worktodo*.txt "$d"/aevum*.log "$d"/out.txt
  printf 'PRP=N/A,1,2,%s,-1,0,0\n' "$E" > "$d/worktodo-0.txt"
  [[ -n "${2:-}" ]] && printf 'PRP=N/A,1,2,%s,-1,0,0\n' "$2" > "$d/worktodo-1.txt"
  echo "$d"
}

BIN_RUN="$BIN"

# ---- A ----
A="$WORK/a"
if [[ ! -f "$A/.done" ]]; then
  rm -rf "$A"; mkdir -p "$A"
  printf 'PRP=N/A,1,2,%s,-1,0,0\n' "$E" > "$A/worktodo-0.txt"
  rc=$(AEVUM_TEST_PROOF_FAILURES=99 run "$A")
  echo "A: exit code $rc"
  touch "$A/.done"
fi
logs "$A" | grep -q "Stopping without reporting the result" || bad "A: no stop after the failed proof"
for p in $POWER 5 4; do logs "$A" | grep -q "Proof of power $p failed" || bad "A: no attempt at power $p"; done
[[ $(logs "$A" | grep -c "Proof of power $POWER failed") -eq 2 ]] || bad "A: power $POWER was not tried twice"
results "$A" | grep -q "$E" && bad "A: the result was reported"
cat "$A"/worktodo*.txt 2>/dev/null | grep -q "$E" || bad "A: the worktodo line is gone"
[[ -s "$A/$E/proof/$MID" ]] || bad "A: proof residues were not kept"
compgen -G "$A/$E/*.prp" >/dev/null || bad "A: no savefile kept"

# ---- W ----
Wd=$(prepare w "$E2")
rc=$(AEVUM_TEST_PROOF_FAILURES=99 run "$Wd" -workers 2)
echo "W: exit code $rc"
stopper=$(grep -l "Stopping without reporting the result" "$Wd"/aevum*.log 2>/dev/null | head -n 1)
if [[ -z "$stopper" ]]; then
  bad "W: no worker stopped after the failed proof"
else
  other=$(ls "$Wd"/aevum*.log | grep -v -x "$stopper" | head -n 1)
  grep -q "Stopping, please wait" "$other" 2>/dev/null || bad "W: the other worker ($other) did not stop"
fi
results "$Wd" | grep -q . && bad "W: a result was reported"
cat "$Wd"/worktodo*.txt 2>/dev/null | grep -q "$E" || bad "W: the $E worktodo line is gone"
cat "$Wd"/worktodo*.txt 2>/dev/null | grep -q "$E2" || bad "W: the $E2 worktodo line is gone"

# ---- B ----
B=$(prepare b)
rc=$(run "$B")
echo "B: exit code $rc"
results "$B" | grep "$E" | grep -q '"proof"' || bad "B: the result with a proof was not reported after the restart"
cat "$B"/worktodo*.txt 2>/dev/null | grep -q "$E" && bad "B: the worktodo line was not retired"

# ---- C ----
corrupt() { printf 'corrupt' | dd of="$1/$E/proof/$MID" bs=1 seek=100 conv=notrunc status=none; }
C=$(prepare c); corrupt "$C"
rc=$(run "$C")
echo "C: exit code $rc"
logs "$C" | grep -q "reporting the result without a proof" || bad "C: no 'reporting the result without a proof'"
results "$C" | grep -q "$E" || bad "C: the result was not reported"
results "$C" | grep "$E" | grep -q '"proof"' && bad "C: the result claims a proof"
if [[ -n "$BASE" ]]; then
  C0=$(prepare c-base); corrupt "$C0"
  BIN_RUN="$BASE"; rc=$(run "$C0"); BIN_RUN="$BIN"
  echo "C (baseline): exit code $rc"
  if results "$C0" | grep -q "$E"; then echo "C (baseline): result reported"; else echo "C (baseline): result LOST"; fi
fi

# ---- D ----
D=$(prepare d)
rc=$(AEVUM_TEST_PROOF_FAILURES=1 run "$D")
echo "D: exit code $rc"
logs "$D" | grep -q "Proof of power $POWER failed (injected" || bad "D: the injected failure did not happen"
results "$D" | grep "$E" | grep -q "\"power\":$POWER" || bad "D: no power-$POWER proof after one retry"

# ---- Z ----
# A's directory with the middle proof residue deleted: no proof power can be built from the savefile on, so proof generation is
# disabled (power 0).  The test must still run to the end and report its result without a proof.
Z=$(prepare z); rm -f "$Z/$E/proof/$MID"
rc=$(run "$Z")
echo "Z: exit code $rc"
logs "$Z" | grep -q "Proof generation disabled" || bad "Z: proof generation was not disabled"
results "$Z" | grep -q "$E" || bad "Z: the result was not reported"
results "$Z" | grep "$E" | grep -q '"proof"' && bad "Z: the result claims a proof"

if [[ $fail -ne 0 ]]; then echo "proof retry GPU test FAILED (work dir $WORK)"; exit 1; fi
echo "proof retry GPU test passed (work dir $WORK)"
