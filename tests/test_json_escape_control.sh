#!/usr/bin/env bash
# Build and run the result-JSON escaping test, then parse its JSON with Python (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-json-escape-control"

rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

declare -a GMP_CFLAGS=()
declare -a GMP_LIBS=(-lgmpxx -lgmp)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gmpxx gmp; then
  read -r -a GMP_CFLAGS <<< "$(pkg-config --cflags gmpxx gmp)"
  read -r -a GMP_LIBS <<< "$(pkg-config --libs gmpxx gmp)"
fi

"${CXX:-c++}" \
  -std=c++20 \
  -O1 \
  -Wall \
  -Wextra \
  -I"$ROOT/include" \
  "${GMP_CFLAGS[@]}" \
  "$ROOT/tests/json_escape_control_test.cpp" \
  "$ROOT/src/io/JsonBuilder.cpp" \
  "$ROOT/src/io/md5.cpp" \
  "$ROOT/src/util/Crc32.cpp" \
  "$ROOT/src/util/GmpUtils.cpp" \
  "$ROOT/src/math/Cofactor.cpp" \
  "${GMP_LIBS[@]}" \
  -o "$BUILD/json-escape-control-test"

"$BUILD/json-escape-control-test" 2> "$BUILD/err.log" > "$BUILD/out.json" || { cat "$BUILD/err.log"; exit 1; }

# The JSON line must parse, and the strings must come back exactly as they went in.
python3 - "$BUILD/out.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    line = [l for l in f.read().splitlines() if l.startswith("{")][-1]
d = json.loads(line)
want_user = "".join(chr(c) for c in range(1, 0x20)) + "\"\\ café"
assert d["user"] == want_user, repr(d["user"])
assert d["computer"] == "box\x7f\x1b[31m", repr(d["computer"])
assert d["aid"] == "0123456789ABCDEF0123456789ABCDEF\x01", repr(d["aid"])
PY
echo "Result JSON control-character escaping: PASS"
