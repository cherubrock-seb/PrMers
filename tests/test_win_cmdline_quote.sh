#!/usr/bin/env bash
# Host unit test of the Windows command-line quoting helper. When a MinGW cross compiler and Wine
# are installed it also runs the same test as a Windows program, which round-trips the arguments
# through the real CommandLineToArgvW and through a real child process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-win-cmdline"
mkdir -p "$BUILD"
"${CXX:-g++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/win_cmdline_quote_test.cpp" -o "$BUILD/win_cmdline_quote_test"
"$BUILD/win_cmdline_quote_test"
MINGW="${MINGW_CXX:-x86_64-w64-mingw32-g++}"
if command -v "$MINGW" >/dev/null 2>&1 && command -v wine >/dev/null 2>&1; then
  "$MINGW" -std=c++20 -O2 -Wall -Wextra -static -I"$ROOT/include" \
    "$ROOT/tests/win_cmdline_quote_test.cpp" -o "$BUILD/win_cmdline_quote_test.exe" -lshell32
  (cd "$BUILD" && WINEDEBUG=-all wine ./win_cmdline_quote_test.exe)
else
  echo "(MinGW and/or Wine not found; skipping the Windows run)"
fi
