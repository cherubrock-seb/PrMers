#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TMPDIR:-/tmp}/prmers_gaussian_worktodo_parser_test"

CXX_BIN="${CXX:-c++}"

declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)

cd "$ROOT"

if [[ -n "${GMP_PREFIX:-}" ]]; then
  test -f "$GMP_PREFIX/include/gmpxx.h" || {
    echo "ERREUR: gmpxx.h absent de $GMP_PREFIX/include" >&2
    exit 1
  }

  GMP_CFLAGS+=("-I${GMP_PREFIX}/include")
  GMP_LIBS=("-L${GMP_PREFIX}/lib" -lgmpxx -lgmp)

  if [[ "$(uname -s)" == "Darwin" ]]; then
    GMP_LIBS+=("-Wl,-rpath,${GMP_PREFIX}/lib")
  fi

  echo "Gaussian worktodo parser test uses GMP_PREFIX=$GMP_PREFIX"

elif command -v pkg-config >/dev/null 2>&1 &&
     pkg-config --exists gmpxx gmp
then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"

  echo "Gaussian worktodo parser test uses pkg-config"

elif command -v brew >/dev/null 2>&1
then
  BREW_GMP_PREFIX="$(brew --prefix gmp)"

  GMP_CFLAGS+=("-I${BREW_GMP_PREFIX}/include")
  GMP_LIBS=(
    "-L${BREW_GMP_PREFIX}/lib"
    -lgmpxx
    -lgmp
    "-Wl,-rpath,${BREW_GMP_PREFIX}/lib"
  )

  echo "Gaussian worktodo parser test uses Homebrew GMP"

else
  echo "Gaussian worktodo parser test uses system GMP"
fi

"$CXX_BIN" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Iinclude \
  "${GMP_CFLAGS[@]}" \
  tests/gaussian_worktodo_parser_test.cpp \
  src/io/WorktodoParser.cpp \
  src/util/StringUtils.cpp \
  src/math/Cofactor.cpp \
  "${GMP_LIBS[@]}" \
  -o "$BIN"

"$BIN"
rm -f "$BIN"
