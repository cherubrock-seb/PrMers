#!/usr/bin/env bash
# Checkpoint loaders must not size an allocation from a length read from the
# file before its CRC is checked: a garbage length makes the loader reject the
# checkpoint (like a CRC failure) instead of zero-filling gigabytes or throwing
# bad_alloc.  Host test, no GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/checkpoint_length_test.cpp" -o "$WORK/test"
mkdir "$WORK/scratch"
"$WORK/test" "$WORK/scratch"

# Source check: no loader sizes a string from the *_len / len it just read
# (the already bounded loader that frees its engine first is exempt).
bad=0
for f in src/modes/RunPM1.cpp src/modes/RunEcm.cpp src/modes/RunEcmTwistedEdwards.cpp; do
  if grep -nE '\.resize\((eacc|wbits)_len\)|std::string [a-z_]*\((eacc_len|wbits_len|len), ' "$ROOT/$f" \
       | grep -v 'delete e; return -2'; then
    echo "$f sizes a string from a length read from the file" >&2; bad=1
  fi
done
[ "$bad" = 0 ] || exit 1
echo "checkpoint length source check passed"
