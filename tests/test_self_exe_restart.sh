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

# A restart that cannot exec must end the process with the dedicated status, not return to the
# caller (which would exit with the test result and look like a finished queue).
rc=0
"$BUILD/bin/prmers-restart-test" fail 2>/dev/null || rc=$?
[ "$rc" = "3" ] || { echo "FAIL (failed exec exits with $rc, expected 3)"; exit 1; }

# restart_self() must go through that path after its exec attempt fails.
grep -q 'util::exitRestartFailed();' "$ROOT/include/core/AlgoUtils.hpp" \
  || { echo "FAIL (restart_self does not exit after a failed restart)"; exit 1; }

echo "self exe restart test passed"
