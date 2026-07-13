#!/usr/bin/env bash
set -euo pipefail
bin=${1:-./build-release/aevum}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
log="$work/aevum-gpu-smoke.log"
"$bin" -dir "$work" -prp 1362763 -device "${AEVUM_TEST_DEVICE:-0}" -proof 0 -iters 10000 -noclean 2>&1 | tee "$log"
if grep -Eiq 'segmentation fault|memory access fault|core dumped|invalid fft|fft size too large' "$log"; then
  echo "Aevum GPU smoke test failed" >&2
  exit 1
fi
if ! grep -Eq 'Testing 2\^1362763|FFT: 256K' "$log"; then
  echo "Aevum GPU smoke test did not start the FFT3161 path" >&2
  exit 1
fi
echo "Aevum GPU FFT3161 smoke test passed"
