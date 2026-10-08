#!/usr/bin/env bash
# A Gaussian-Mersenne factoring checkpoint that cannot be written (full or
# read-only disk) must not end the run: the writer reports an error, keeps the
# previous checkpoint, and the drivers warn and carry on.  Host test, no GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'chmod -R u+rwX "$WORK" 2>/dev/null || true; rm -rf "$WORK"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  ${GMP_PREFIX:+-I"$GMP_PREFIX/include" -L"$GMP_PREFIX/lib"} \
  "$ROOT/tests/gm_factor_checkpoint_write_failure_test.cpp" -o "$WORK/test" -lgmpxx -lgmp
"$WORK/test" "$WORK"

# No driver may call the throwing writer directly: every save goes through the
# non-throwing wrapper.
for f in src/modes/RunGaussianMersennePm1VTrace.cpp src/modes/RunGaussianMersenneFactor.cpp; do
  if grep -n 'save_factor_checkpoint(' "$ROOT/$f" | grep -v 'try_save_factor_checkpoint('; then
    echo "$f calls the throwing save_factor_checkpoint directly" >&2; exit 1
  fi
done
# The optimized ECM and NAF ECM drivers: only the wrappers may call their throwing writers.
for pair in "src/modes/RunGaussianMersenneEcmOptimized.cpp:save_opt_checkpoint" "src/modes/RunGaussianMersenneEcmFast.cpp:save_naf_checkpoint"; do
  f="${pair%%:*}"; fn="${pair##*:}"
  if grep -n "$fn(" "$ROOT/$f" | grep -v "try_$fn(" | grep -v "^[0-9]*:void $fn(" | grep -v "^[0-9]*:        $fn(path"; then
    echo "$f calls the throwing $fn directly" >&2; exit 1
  fi
done
echo "gm factoring checkpoint source check passed"
