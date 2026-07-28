#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TMPDIR:-/tmp}/prmers_gaussian_worktodo_parser_test"
cd "$ROOT"
g++ -std=c++20 -O2 -Wall -Wextra -Iinclude \
  tests/gaussian_worktodo_parser_test.cpp \
  src/io/WorktodoParser.cpp src/util/StringUtils.cpp src/math/Cofactor.cpp \
  -lgmpxx -lgmp -o "$BIN"
"$BIN"
rm -f "$BIN"
