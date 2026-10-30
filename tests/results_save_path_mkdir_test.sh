#!/usr/bin/env bash
# Result files under a save path (-f) that does not exist yet (host only, no device).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
"${CXX:-g++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/results_save_path_mkdir_test.cpp" "$ROOT/src/io/WorktodoManager.cpp" -o "$BUILD/test"
"$BUILD/test" "$BUILD/work"
