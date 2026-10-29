#!/usr/bin/env bash
# Stage-1 P-1 / ECM checkpoint writers must stop when the final write, flush or
# close fails, and must swap `<ckpt>.new` into place through
# core::ckptcommit::commit (never deleting the only good checkpoint).
# Host test, no GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/checkpoint_commit_test.cpp" -o "$WORK/test"
mkdir "$WORK/scratch"
"$WORK/test" "$WORK/scratch"

# Source check: no checkpoint writer ignores write_crc32(), and none swaps the
# files by hand any more.
bad=0
for f in src/modes/RunPM1.cpp src/modes/RunEcm.cpp src/modes/RunEcmTwistedEdwards.cpp; do
  if grep -nE '(^|[;{}])\s*f\.write_crc32\(\)\s*;' "$ROOT/$f"; then
    echo "$f ignores the result of write_crc32()" >&2; bad=1
  fi
  if grep -nE 'fs::rename\((ckpt[A-Za-z0-9_]*)\s*,\s*\1\s*\+\s*"\.old"|std::rename\(ckpt_file_s[0-9]' "$ROOT/$f"; then
    echo "$f swaps a checkpoint by hand instead of core::ckptcommit::commit" >&2; bad=1
  fi
done
[ "$bad" = 0 ] || exit 1
echo "checkpoint commit source check passed"
