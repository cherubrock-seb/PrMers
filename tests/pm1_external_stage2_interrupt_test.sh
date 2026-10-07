#!/usr/bin/env bash
# The external Prime95 stage 2 is only a first choice: when it is interrupted the
# run must stop and keep the checkpoint and worktodo line, and only a real
# Prime95 failure may fall through to the internal stage 2.  Host test, no GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/pm1_external_stage2_interrupt_test.cpp" -o "$WORK/test"
"$WORK/test" "$WORK"

# Both call sites in the driver must use the decision helper, not a bare
# `if (!external_used)`.
src="$ROOT/src/modes/RunPM1.cpp"
n=$(grep -c 'pm1AfterExternalStage2(external_used, interrupted)' "$src")
[ "$n" -eq 2 ] || { echo "expected 2 handoff call sites, found $n" >&2; exit 1; }
if grep -q 'if (!external_used)' "$src"; then
  echo "a handoff still falls through on !external_used" >&2; exit 1
fi
grep -q 'result.interrupted = core::pm1SystemStatusInterrupted(rc)' "$src" || { echo "p95 task does not record an interrupted child" >&2; exit 1; }
echo "pm1 external stage-2 source check passed"
