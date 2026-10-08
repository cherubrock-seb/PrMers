#!/usr/bin/env bash
# Host test: the web GUI cannot change path/network options or append anything but worktodo entries.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-gui-paths"
rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gmpxx gmp; then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"
fi

"${CXX:-g++}" -std=c++20 -O2 -Wall -Wextra -pthread -I"$ROOT/include" "${GMP_CFLAGS[@]}" \
  "$ROOT/tests/web_gui_paths_test.cpp" \
  "$ROOT/src/ui/WebGuiServer.cpp" \
  "$ROOT/src/io/WorktodoParser.cpp" \
  "$ROOT/src/util/StringUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "$ROOT/src/math/Pm1Bounds.cpp" \
  "${GMP_LIBS[@]}" \
  -o "$BUILD/web_gui_paths_test"
cd "$BUILD" && ./web_gui_paths_test
python3 "$ROOT/tests/gui_settings_arity_source_test.py"
