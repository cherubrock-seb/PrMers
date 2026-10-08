#!/usr/bin/env bash
# Proof files and proof residues live under -f, with the checkpoints and
# results.txt, and not in the working directory.
#   1. -f elsewhere (absolute, relative, trailing slash, symlink): the proof,
#      its residues (kept when the result cannot be saved) and results.txt are
#      under -f, nothing proof-related appears in the working directory, and the
#      proof verifies.
#   2. A run resumed with residues left in <E>/proof under the working directory
#      by an older version (and none under -f) uses them in place, makes a
#      verified proof under -f and deletes them from where it used them. Residues
#      in both places: the ones under -f are used and the old ones are left alone.
#   3. Two runs with different -f in one working directory do not share files.
#   4. An -f that cannot be written: the PRP finishes, no proof, nothing in the
#      working directory.
# Both drivers (default Marin, and -marin, the legacy loop).
# usage: run_proof_save_path_regression.sh <device> [exponent]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-11213}"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# Failures are counted in a file: the checks run in subshells.
FAILS="$(mktemp)"
trap 'rm -f "$FAILS"' EXIT
bad() { echo "FAIL: $*"; echo x >> "$FAILS"; }

# prmers <log> <args...>: run in the current directory.
prmers() {
    local log="$1"; shift
    $RUN timeout 300 "$BIN" "$P" -prp -proof 3 "$@" -d "$DEVICE" -noask > "$log" 2>&1
}

# Nothing of the proof machinery in the directory $1.
no_proof_files() {
    local where="$1" what="$2"
    [ ! -e "$where/proof" ] && [ ! -e "$where/proof-tmp" ] && [ ! -e "$where/$P" ] \
        && ! ls "$where"/m_"$P".ckpt* > /dev/null 2>&1 \
        || bad "$what: proof files or residues appeared in $where: $(cd "$where" && ls -d proof proof-tmp "$P" 2>/dev/null | tr '\n' ' ')"
}
has_proof() {
    [ -s "$1/proof/$P-3.proof" ] || bad "$2: no proof file in $1/proof"
}
verified() {
    grep -q 'Verification result: SUCCESS' "$1" || bad "$2: proof was not verified"
}

new_work() {
    local w
    w="$(mktemp -d)"
    ln -s "$ROOT/kernels" "$w/kernels"
    echo "$w"
}

for driver in marin legacy; do
    if [ "$driver" = legacy ]; then flag=(-marin); else flag=(); fi

    # ---- 1. -f elsewhere ----
    W="$(new_work)"
    mkdir "$W/cwd" "$W/save"
    ln -s "$W/save" "$W/link"
    ( cd "$W/cwd" || exit 1
      for spelling in "$W/save" "$W/save/" "../save" "../link"; do
          rm -rf "$W/save"/* ./*.log
          prmers run.log "${flag[@]}" -f "$spelling"
          verified run.log "$driver -f $spelling"
          has_proof "$W/save" "$driver -f $spelling"
          [ -s "$W/save/results.txt" ] || bad "$driver -f $spelling: no results.txt under -f"
          no_proof_files "$W/cwd" "$driver -f $spelling"
          grep -q "Proof file saved" run.log || bad "$driver -f $spelling: no proof file notice"
          grep -q "\"md5\":\"[0-9a-f]\{32\}\"" run.log || bad "$driver -f $spelling: result has no proof md5"
      done
      echo "$driver: SAVE_PATH_SPELLINGS=done"

      # Residues are checkpoint data: they stay under -f when the result cannot
      # be saved (results.txt is a directory), and go once it can.
      rm -rf "$W/save"/*; mkdir "$W/save/results.txt"
      prmers keep.log "${flag[@]}" -f "$W/save"
      [ -n "$(ls -A "$W/save/$P/proof" 2>/dev/null)" ] || bad "$driver: residues not kept under -f"
      no_proof_files "$W/cwd" "$driver kept residues"
      grep -q "Proof residues kept in $W/save/$P/proof" keep.log || bad "$driver: kept notice does not name the -f directory"
      rmdir "$W/save/results.txt"
      prmers done.log "${flag[@]}" -f "$W/save"
      [ ! -e "$W/save/$P" ] || bad "$driver: residues left under -f after the result was saved"
      no_proof_files "$W/cwd" "$driver after cleanup"
      echo "$driver: RESIDUES_UNDER_SAVE_PATH=done"
    ) || bad "$driver: a step could not run"
    rm -rf "$W"

    # ---- 2. residues in the old location ----
    W="$(new_work)"
    mkdir "$W/cwd" "$W/save2"
    ( cd "$W/cwd" || exit 1
      # A first run is stopped short of saving its result (results.txt is a
      # directory), so its state and residues stay. Its residues are then put
      # where an older version left them: <E>/proof under the working directory.
      mkdir "$W/save2/results.txt"
      prmers first.log "${flag[@]}" -f "$W/save2"
      [ -n "$(ls -A "$W/save2/$P/proof" 2>/dev/null)" ] || bad "$driver: first run left no residues"
      mv "$W/save2/$P" "./$P"
      rm -rf "$W/save2/proof" "$W/save2/proof-tmp"
      rmdir "$W/save2/results.txt"
      cp -a "$P" old-residues.keep
      prmers resume.log "${flag[@]}" -f "$W/save2"
      grep -q 'old location' resume.log || bad "$driver: no note about the old location"
      [ "$(grep -c 'old location' resume.log)" = 1 ] || bad "$driver: the old-location note is not one line"
      verified resume.log "$driver resume from old location"
      has_proof "$W/save2" "$driver resume from old location"
      [ ! -e "$P/proof" ] || bad "$driver: old residues left after the job"
      [ ! -e "$W/save2/$P" ] || bad "$driver: residues under -f left after the job"
      [ ! -e proof ] && [ ! -e proof-tmp ] || bad "$driver: proof written to the working directory"
      echo "$driver: RESUME_FROM_OLD_LOCATION=done"

      # A new run starts from 0 with a stale old set lying here: it writes its
      # own residues and never reads or deletes the old ones. Interrupted at the
      # end (results.txt a directory), the rerun then finds every residue under
      # -f and uses those: no note, and the stale set is still there.
      rm -rf "$W/save2"/*
      cp -a old-residues.keep "$P"
      mkdir "$W/save2/results.txt"
      prmers fresh.log "${flag[@]}" -f "$W/save2"
      grep -q 'old location' fresh.log && bad "$driver: a fresh run used the old location"
      rmdir "$W/save2/results.txt"
      prmers both.log "${flag[@]}" -f "$W/save2"
      verified both.log "$driver both locations"
      has_proof "$W/save2" "$driver both locations"
      grep -q 'old location' both.log && bad "$driver: old location used although -f has them all"
      [ -n "$(ls -A "$P/proof" 2>/dev/null)" ] || bad "$driver: stale old residues deleted although they were not used"
      echo "$driver: RESIDUES_IN_BOTH=done"
    ) || bad "$driver: a step could not run"
    rm -rf "$W"

    # ---- 2b. the same, for a run interrupted part way ----
    # Residues are written as the run goes; an older version left them in
    # <E>/proof under the working directory. Resuming reads them in place.
    W="$(new_work)"
    mkdir "$W/cwd" "$W/s1" "$W/s2"
    ( cd "$W/cwd" || exit 1
      M=44497
      $RUN timeout -s INT 5 "$BIN" "$M" -prp -proof 4 "${flag[@]}" -t 1 -d "$DEVICE" -noask -f "$W/s1" > first.log 2>&1
      if [ -z "$(ls -A "$W/s1/$M/proof" 2>/dev/null)" ] || ! grep -q 'state saved at iteration' first.log; then
          echo "SKIP: $driver mid-run resume (no residues before the interrupt)"
      else
          if [ "$driver" = marin ]; then
              # The default driver keeps its checkpoint under -f too.
              [ -e "$W/s1/m_$M.ckpt" ] || bad "marin mid-run: checkpoint not under -f"
              ls m_"$M".ckpt* > /dev/null 2>&1 && bad "marin mid-run: checkpoint written to the working directory"
              # Put it where an older version kept it, to be resumed from there.
              mv "$W"/s1/m_"$M".ckpt* .
          fi
          mv "$W/s1/$M" "./$M"
          # A residue only power 4 needs is gone: the resumed run must look in
          # both places to find that power 3 is still possible.
          rm -f "./$M/proof/2782"
          if [ "$driver" = legacy ]; then F="$W/s1"; else F="$W/s2"; fi
          $RUN timeout 300 "$BIN" "$M" -prp -proof 4 "${flag[@]}" -t 1 -d "$DEVICE" -noask -f "$F" > second.log 2>&1
          grep -q 'Resuming from' second.log || bad "$driver mid-run: did not resume"
          [ "$(grep -c 'old location' second.log)" = 1 ] || bad "$driver mid-run: no one-line note about the old location"
          if [ "$driver" = marin ]; then
              grep -q 'No checkpoint under .*; using m_.*from the working directory' second.log || bad "marin mid-run: no note about the old checkpoint"
              ls m_"$M".ckpt* > /dev/null 2>&1 && bad "marin mid-run: old checkpoint left in the working directory"
              ls "$F"/m_"$M".ckpt* > /dev/null 2>&1 && bad "marin mid-run: checkpoint left under -f"
          fi
          verified second.log "$driver mid-run resume"
          grep -q 'proof of power 3 (instead of 4)' second.log || bad "$driver mid-run: power not lowered to 3"
          [ -s "$F/proof/$M-3.proof" ] || bad "$driver mid-run: no power 3 proof under -f"
          [ ! -e "$M" ] || bad "$driver mid-run: old residues left"
          [ ! -e "$F/$M" ] || bad "$driver mid-run: residues under -f left"
          [ ! -e proof ] && [ ! -e proof-tmp ] || bad "$driver mid-run: proof written to the working directory"
          echo "$driver: MIDRUN_RESUME_FROM_OLD_LOCATION=done"
      fi
    ) || bad "$driver: a step could not run"
    rm -rf "$W"

    # ---- 3. two -f in one working directory, at the same time ----
    W="$(new_work)"
    mkdir "$W/cwd" "$W/a" "$W/b"
    ( cd "$W/cwd" || exit 1
      prmers a.log "${flag[@]}" -f "$W/a" &
      pa=$!
      prmers b.log "${flag[@]}" -f "$W/b" &
      pb=$!
      wait "$pa"; wait "$pb"
      for d in a b; do
          verified "$d.log" "$driver two -f ($d)"
          has_proof "$W/$d" "$driver two -f ($d)"
          [ -s "$W/$d/results.txt" ] || bad "$driver two -f ($d): no results.txt"
      done
      no_proof_files "$W/cwd" "$driver two -f"
      echo "$driver: TWO_SAVE_PATHS=done"
    ) || bad "$driver: a step could not run"
    rm -rf "$W"

    # ---- 4. -f that cannot be written ----
    if [ "$(id -u)" != 0 ]; then
        W="$(new_work)"
        mkdir "$W/cwd" "$W/ro"
        chmod 555 "$W/ro"
        ( cd "$W/cwd" || exit 1
          prmers ro.log "${flag[@]}" -f "$W/ro"
          grep -q 'probable prime\|probably prime' ro.log || bad "$driver unwritable -f: the PRP did not finish"
          [ -z "$(ls -A "$W/ro")" ] || bad "$driver unwritable -f: something was written"
          no_proof_files "$W/cwd" "$driver unwritable -f"
          grep -q 'Proof generation failed\|No proof will be produced\|proof checkpoint' ro.log || bad "$driver unwritable -f: no word about the missing proof"
          echo "$driver: UNWRITABLE_SAVE_PATH=done"
        ) || bad "$driver: a step could not run"
        chmod 755 "$W/ro"
        rm -rf "$W"
    else
        echo "SKIP: unwritable -f (running as root)"
    fi
done

if [ ! -s "$FAILS" ]; then echo "PROOF_SAVE_PATH=PASS"; exit 0; fi
echo "PROOF_SAVE_PATH=FAIL"
exit 1
