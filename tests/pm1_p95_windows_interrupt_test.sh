#!/usr/bin/env bash
# The Windows Prime95 handoff must flag an interrupt (result.interrupted) like the POSIX one does.
# Source check always; the behaviour test needs MinGW to build and Wine to run, and says so when
# either is missing. No GPU needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

src="$ROOT/src/modes/RunPM1.cpp"
grep -q 'core::pm1WaitProcessInterruptible(pi.hProcess, stop, tick, interrupted)' "$src" \
  || { echo "the Windows runner does not poll the interrupt flag" >&2; exit 1; }
grep -q 'p95_run_windows_process(p95_exe, p95_dir, log_path, interrupted, show_progress, run_interrupted)' "$src" \
  || { echo "the Windows handoff does not pass the interrupt flag" >&2; exit 1; }
n=$(grep -c 'result.interrupted = run_interrupted' "$src")
[ "$n" -eq 2 ] || { echo "expected result.interrupted to be set on both platforms, found $n" >&2; exit 1; }
if grep -q 'WaitForSingleObject(pi.hProcess, INFINITE)' "$src"; then
  echo "the Windows runner still waits without polling" >&2; exit 1
fi

CXXWIN="${CXXWIN:-x86_64-w64-mingw32-g++}"
if ! command -v "$CXXWIN" >/dev/null 2>&1; then
  echo "SKIPPED behaviour test: no MinGW compiler ($CXXWIN)"
  echo "pm1 Windows Prime95 interrupt source check passed"
  exit 0
fi
"$CXXWIN" -std=c++20 -O2 -Wall -Wextra -static -I"$ROOT/include" \
  "$ROOT/tests/pm1_p95_windows_interrupt_test.cpp" -o "$WORK/test.exe"
if command -v wine >/dev/null 2>&1; then
  WINEDEBUG=-all timeout 300 wine "$WORK/test.exe"
else
  echo "SKIPPED behaviour test: built with MinGW but there is no wine to run it"
fi
echo "pm1 Windows Prime95 interrupt test passed"
