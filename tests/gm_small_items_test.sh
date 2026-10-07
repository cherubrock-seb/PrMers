#!/usr/bin/env bash
# Gaussian trial factoring must reject a composite exponent before touching
# the GPU, and the Gaussian-Mersenne driver must not call GQ_2 prime.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRMERS="${PRMERS_BIN:-$ROOT/prmers}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

fail() { echo "FAIL: $*" >&2; exit 1; }

# Composite p: validateRequest throws before any OpenCL context is created.
for p in 9 15 21 25; do
  out="$("$PRMERS" "$p" -gm-tf 20 26 -d 0 -f "$WORK/tf" 2>&1)"
  rc=$?
  [[ $rc -ne 0 ]] || fail "composite p=$p was accepted by -gm-tf"
  grep -q "requires a prime exponent" <<<"$out" || fail "p=$p: unexpected message: $out"
  [[ ! -e "$WORK/tf" ]] || fail "p=$p wrote output for a rejected request"
done

# Source checks for the two guards.
grep -q 'Gaussian TF requires a prime exponent p' "$ROOT/src/modes/RunGaussianTrialFactor.cpp" \
  || fail "TF prime-exponent guard missing"
grep -q '_2 = 1 is not prime' "$ROOT/src/modes/RunGaussianMersenne.cpp" \
  || fail "GQ_2 guard missing"

echo "GM small items test passed"
