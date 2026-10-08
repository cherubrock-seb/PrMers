#!/usr/bin/env bash
# Host test: GUI "Append & Run" worktodo append (trailing-newline handling).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-worktodo-append"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gmpxx gmp; then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"
fi

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -pthread \
  -I"$ROOT/include" \
  "${GMP_CFLAGS[@]}" \
  "$ROOT/tests/worktodo_append_test.cpp" \
  "$ROOT/src/io/WorktodoParser.cpp" \
  "$ROOT/src/util/StringUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/math/Pm1Bounds.cpp" \
  "${GMP_LIBS[@]}" \
  -o "$BUILD/worktodo-append-test"

# The test creates its files (and worktodo_save.txt) in the current directory.
cd "$BUILD"
./worktodo-append-test
