#!/usr/bin/env bash
# P-1 B1 extension (-b1old) must honour -pm1-lowmem. It used to drop low-memory mode silently and allocate
# the full 11-register set; it now runs the 3-register delta extension.
#
# M269 has the factor q = 13822297, q - 1 = 2^3 * 3 * 269 * 2141, so stage 1 finds it at B1=2141 and not below.
#
# Steps (Marin backend, on OpenCL device 0 unless PRMERS_TEST_DEVICE is set):
#   1. -b1 100, then -b1old 100 -b1 1000 with -pm1-lowmem: the log says low-memory (3 registers), never
#      "EXTEND mode" (the normal-memory label) and never silently ignores the flag
#   2. -b1old 1000 -b1 2141 -pm1-lowmem finds 13822297, and its resume file is byte-identical to the one
#      of a direct -b1 2141 run (so the 3-register delta extension computes the same stage-1 state)
#   3. the same with -pm1-ultralowmem still takes the ultra-low-memory label
#   4. without -pm1-lowmem the extension is the normal-memory one
#   5. -pm1-lowmem extension followed by stage 2 (-b2) still completes
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEV="${PRMERS_TEST_DEVICE:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export AEVUM_CARRY_WMUL="${AEVUM_CARRY_WMUL:-1}" AEVUM_AUTOTUNE="${AEVUM_AUTOTUNE:-off}"

run() {
  local dir="$1" name="$2"
  shift 2
  mkdir -p "$dir"
  [[ -e "$dir/kernels" ]] || ln -s "$ROOT/kernels" "$dir/kernels"
  set +e
  (cd "$dir" && timeout -s INT 120 "$BIN" 269 -pm1 -engine-marin -d "$DEV" "$@") >"$WORK/$name.log" 2>&1
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
has() { grep -q -- "$2" "$WORK/$1.log"; }

LOW="$WORK/low"
run "$LOW" low1 -b1 100
run "$LOW" low2 -b1old 100 -b1 1000 -pm1-lowmem -t 0
has low2 'Low-memory B1 extension requested' || fail "-pm1-lowmem -b1old did not report a low-memory extension" low2
has low2 'Low-memory delta extension enabled: using 3 GPU registers' || fail "-pm1-lowmem -b1old did not use 3 registers" low2
has low2 'LOWMEM DELTA 3-REG' || fail "-pm1-lowmem -b1old: missing the low-memory banner" low2
has low2 'EXTEND mode' && fail "-pm1-lowmem -b1old fell back to the normal-memory extension" low2
has low2 'Ultra-low-memory' && fail "-pm1-lowmem -b1old reported ultra-low-memory" low2

run "$LOW" low3 -b1old 1000 -b1 2141 -pm1-lowmem
has low3 'P-1 factor stage 1 found: 13822297' || fail "-pm1-lowmem -b1old 1000 -b1 2141 missed 13822297" low3

DIRECT="$WORK/direct"
run "$DIRECT" direct -b1 2141
has direct 'P-1 factor stage 1 found: 13822297' || fail "direct -b1 2141 missed 13822297" direct
cmp -s "$DIRECT/resume_p269_B1_2141.save" "$LOW/resume_p269_B1_2141.save" \
  || fail "low-memory extended and direct B1=2141 resume files differ"

ULTRA="$WORK/ultra"
run "$ULTRA" ultra1 -b1 100
run "$ULTRA" ultra2 -b1old 100 -b1 1000 -pm1-ultralowmem -t 0
has ultra2 'Ultra-low-memory B1 extension requested' || fail "-pm1-ultralowmem -b1old lost the ultra-low-memory label" ultra2
has ultra2 'ULTRALOWMEM DELTA 3-REG' || fail "-pm1-ultralowmem -b1old: missing the ultra banner" ultra2

NORMAL="$WORK/normal"
run "$NORMAL" normal1 -b1 100
run "$NORMAL" normal2 -b1old 100 -b1 1000 -t 0
has normal2 'EXTEND mode' || fail "a plain -b1old extension lost the normal-memory banner" normal2
has normal2 'delta extension enabled' && fail "a plain -b1old extension used the low-memory path" normal2

STAGE2="$WORK/stage2"
run "$STAGE2" stage2a -b1 100
run "$STAGE2" stage2b -b1old 100 -b1 1000 -b2 5000 -pm1-lowmem -t 0
has stage2b 'Low-memory delta extension enabled: using 3 GPU registers' || fail "low-memory extension with -b2 did not use 3 registers" stage2b

echo "P-1 low-memory extension test passed"
