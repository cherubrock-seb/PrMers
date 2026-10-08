#!/usr/bin/env bash
# Host test for the ECM Prime95 stage-2 handoff helpers (no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'chmod -R u+w "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -pthread -I"$ROOT/include" \
  "$ROOT/tests/ecm_prime95_handoff_test.cpp" -o "$WORK/test"
"$WORK/test" "$WORK"

# The driver must start Prime95 through the interruptible runner, not std::system().
src="$ROOT/src/modes/RunEcmTwistedEdwards.cpp"
grep -q 'core::ecmRunShellInterruptible(' "$src" || { echo "Prime95 is not started through the interruptible runner" >&2; exit 1; }
if grep -v '^ *//' "$src" | grep -q 'std::system('; then echo "std::system() is still used to start Prime95" >&2; exit 1; fi
echo "ecm Prime95 handoff source check passed"
