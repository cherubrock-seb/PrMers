#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-worktodo-small-items"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

# The settings generator and worktodo builder in the web GUI page must use
# option and field forms the CLI/parser understand.
GUI="$ROOT/src/ui/WebGuiServer.cpp"
grep -qF "parts.push('-kernelpath',kp)" "$GUI" \
  || { echo "FAIL: GUI must emit -kernelpath (the CLI has no -kernel_path)"; exit 1; }
grep -qF 'line+=`,0,0,`+basert[0]+`,`+basert[1]' "$GUI" \
  || { echo "FAIL: GUI PRP line must pad tf/tests_saved before base,residue_type"; exit 1; }

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  $(pkg-config --cflags gmpxx gmp 2>/dev/null || true) \
  "$ROOT/tests/worktodo_small_items_test.cpp" \
  "$ROOT/src/io/WorktodoParser.cpp" \
  "$ROOT/src/util/StringUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/math/Pm1Bounds.cpp" \
  "$ROOT/src/core/Logger.cpp" \
  $(pkg-config --libs gmpxx gmp 2>/dev/null || echo "-lgmpxx -lgmp") \
  -o "$BUILD/worktodo-small-items-test"

"$BUILD/worktodo-small-items-test"
