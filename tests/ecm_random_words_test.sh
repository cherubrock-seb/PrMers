#!/usr/bin/env bash
# ECM random words must be 64-bit on every platform.
#   - native: reference-vector test with GMP (tests/ecm_random_words_test.cpp)
#   - Windows (LLP64): when MinGW and Wine are installed, build the GMP-free half for
#     x86_64-w64-mingw32, run it under Wine and compare it with the native output.
#     Skipped (with a message) when either is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CXX="${CXX:-g++}"

"$CXX" -std=c++20 -Wall -Wextra -I"$ROOT/include" "$ROOT/tests/ecm_random_words_test.cpp" \
  -o "$WORK/words_gmp" -lgmpxx -lgmp
"$WORK/words_gmp"

"$CXX" -std=c++20 -Wall -Wextra -I"$ROOT/include" "$ROOT/tests/ecm_random_words_portable_test.cpp" \
  -o "$WORK/words_native"
"$WORK/words_native" >"$WORK/native.txt"

MINGW="${MINGW_CXX:-x86_64-w64-mingw32-g++}"
if command -v "$MINGW" >/dev/null 2>&1 && command -v wine >/dev/null 2>&1; then
  "$MINGW" -std=c++20 -Wall -Wextra -static -I"$ROOT/include" "$ROOT/tests/ecm_random_words_portable_test.cpp" \
    -o "$WORK/words_win.exe"
  WINEPREFIX="$WORK/wineprefix" WINEDEBUG=-all wine "$WORK/words_win.exe" 2>/dev/null | tr -d '\r' >"$WORK/win.txt"
  grep -q '^sizeof_unsigned_long=4$' "$WORK/win.txt" || { echo "MinGW build does not have a 32-bit unsigned long" >&2; exit 1; }
  grep -q '^narrowing_lossless=0$' "$WORK/win.txt" || { echo "the (unsigned long) narrowing should be lossy on LLP64" >&2; exit 1; }
  # Everything but the platform lines must be identical to the native output.
  diff <(grep -v -e '^sizeof_unsigned_long=' -e '^narrowing_lossless=' "$WORK/native.txt") \
       <(grep -v -e '^sizeof_unsigned_long=' -e '^narrowing_lossless=' "$WORK/win.txt") \
    || { echo "random words differ between LP64 and LLP64" >&2; exit 1; }
  echo "ecm random words test passed (native + MinGW/Wine LLP64)"
else
  echo "ecm random words test passed (native only; MinGW/Wine not installed)"
fi
