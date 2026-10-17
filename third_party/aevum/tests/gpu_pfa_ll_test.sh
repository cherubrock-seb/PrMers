#!/usr/bin/env bash
# LL on a native PFA plan must apply the -2 on every iteration, including the
# retained-width (non-log) ones.  Reference residues are from an independent
# GMP computation of s(k) = s(k-1)^2 - 2 mod 2^6972593 - 1, s(0) = 4.
set -euo pipefail
bin=${1:-./build-release/aevum}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
log="$work/aevum-pfa-ll.log"
"$bin" -dir "$work" -ll 6972593 -fft pfa3:1:256:3:256:101 -device "${AEVUM_TEST_DEVICE:-0}" \
  -iters 2000 -log 1000 2>&1 | tee "$log"
if ! grep -q 'pfa3:1:256:3:256' "$log"; then
  echo "Aevum PFA LL test did not run the pfa3 plan" >&2
  exit 1
fi
if grep -q 'Using long carry' "$log"; then
  echo "Aevum PFA LL test took the long-carry path; it must exercise the retained-width path" >&2
  exit 1
fi
fail=0
for expect in "1000 ef833400dc07adae" "2000 89300d1cafbc4a80"; do
  if ! grep -Eq " ${expect%% *} ${expect##* } " "$log"; then
    echo "Aevum PFA LL residue mismatch: expected $expect" >&2
    fail=1
  fi
done
[ "$fail" = 0 ] || exit 1
echo "Aevum PFA LL test passed"
