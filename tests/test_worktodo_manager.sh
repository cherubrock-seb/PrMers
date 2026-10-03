#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-worktodo-manager"

rm -rf "$BUILD"
mkdir -p "$BUILD"

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/worktodo_manager_test.cpp" \
  "$ROOT/src/io/WorktodoManager.cpp" \
  -o "$BUILD/worktodo-manager-test"

"$BUILD/worktodo-manager-test"
rm -rf "$BUILD"
