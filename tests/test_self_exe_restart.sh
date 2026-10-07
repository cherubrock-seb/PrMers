#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-self-exe-restart"

rm -rf "$BUILD"
mkdir -p "$BUILD/bin" "$BUILD/cwd"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "$ROOT/tests/self_exe_restart_test.cpp" \
  -o "$BUILD/bin/prmers-restart-test"

# Started through PATH from an unrelated working directory: argv[0] is the bare
# name "prmers-restart-test", which execv(argv[0]) cannot find.
out="$(cd "$BUILD/cwd" && PATH="$BUILD/bin:$PATH" prmers-restart-test)"
[ "$out" = "RESTART_OK" ] || { echo "FAIL (PATH invocation): $out"; exit 1; }

# Relative path from another directory.
out="$(cd "$BUILD" && ./bin/prmers-restart-test)"
[ "$out" = "RESTART_OK" ] || { echo "FAIL (relative invocation): $out"; exit 1; }

echo "self exe restart test passed"
