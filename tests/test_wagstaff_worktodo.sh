#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TMPDIR:-/tmp}/prmers_wagstaff_worktodo_test"
CXX_BIN="${CXX:-c++}"

cd "$ROOT"

# GMP flags are only needed by the sources WorktodoParser.cpp pulls in.
declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gmpxx gmp; then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"
fi

"$CXX_BIN" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Iinclude \
  "${GMP_CFLAGS[@]}" \
  tests/wagstaff_worktodo_test.cpp \
  src/io/WorktodoParser.cpp \
  src/util/StringUtils.cpp \
  src/math/Cofactor.cpp \
  src/math/Pm1Bounds.cpp \
  "${GMP_LIBS[@]}" \
  -o "$BIN"

"$BIN"
rm -f "$BIN"
