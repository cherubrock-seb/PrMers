#!/usr/bin/env bash
# P-1 B1 extension (-b1old) must not be disturbed by an extension checkpoint
# it does not resume from, and must not leave its own one behind.
#
# M269 has the factor q = 13822297, q - 1 = 2^3 * 3 * 269 * 2141, so stage 1
# finds it at B1=2141 and not below.
#
# Steps (Marin backend, on OpenCL device 0 unless PRMERS_TEST_DEVICE is set):
#   1. -b1 100
#   2. -b1old 100 -b1 1000            -> no _ext.ckpt may remain afterwards
#   3. plant a stale _ext.ckpt (fixture: written by an older build during
#      the 100 -> 1000 extension), then -b1old 1000 -b1 2141
#                                     -> the stale file is ignored and the
#                                        factor is found
#   4. the extension's resume file equals the one of a direct -b1 2141 run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEV="${PRMERS_TEST_DEVICE:-0}"
FIXTURE="$ROOT/tests/fixtures/pm1_extend_stale_ckpt"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export AEVUM_CARRY_WMUL="${AEVUM_CARRY_WMUL:-1}" AEVUM_AUTOTUNE="${AEVUM_AUTOTUNE:-off}"

run() {
  local dir="$1" name="$2"
  shift 2
  mkdir -p "$dir"
  [[ -e "$dir/kernels" ]] || ln -s "$ROOT/kernels" "$dir/kernels"
  set +e
  (cd "$dir" && timeout -s INT 50 "$BIN" 269 -pm1 -engine-marin -d "$DEV" "$@") >"$WORK/$name.log" 2>&1
  local rc=$?
  set -e
  # 0 = factor found, 1 = no factor; anything else is a failure.
  if [[ "$rc" != 0 && "$rc" != 1 ]]; then
    echo "$name: prmers exited with $rc" >&2
    tail -n 30 "$WORK/$name.log" >&2
    exit 1
  fi
}

fail() { echo "FAIL: $1" >&2; [[ -n "${2:-}" ]] && tail -n 30 "$WORK/$2.log" >&2; exit 1; }

EXT="$WORK/ext"
run "$EXT" step1 -b1 100
run "$EXT" step2 -b1old 100 -b1 1000 -t 0
if compgen -G "$EXT/pm1_m_269_ext.ckpt*" >/dev/null; then
  fail "the extension left $(cd "$EXT" && echo pm1_m_269_ext.ckpt*) behind" step2
fi

cp "$FIXTURE/pm1_m_269_ext.ckpt" "$FIXTURE/pm1_m_269_ext.ckpt.backend" "$EXT/"
run "$EXT" step3 -b1old 1000 -b1 2141
grep -q 'Ignoring extension checkpoint' "$WORK/step3.log" || fail "the stale extension checkpoint was not reported as ignored" step3
grep -q 'P-1 factor stage 1 found: 13822297' "$WORK/step3.log" || fail "-b1old 1000 -b1 2141 missed 13822297" step3

DIRECT="$WORK/direct"
run "$DIRECT" direct -b1 2141
grep -q 'P-1 factor stage 1 found: 13822297' "$WORK/direct.log" || fail "direct -b1 2141 missed 13822297" direct
cmp -s "$DIRECT/resume_p269_B1_2141.save" "$EXT/resume_p269_B1_2141.save" \
  || fail "extended and direct B1=2141 resume files differ"

echo "P-1 extension checkpoint test passed"
