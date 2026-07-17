#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
BIN="$ROOT/prmers"
LIB="$ROOT/third_party/aevum/build-engine/libaevum_engine.so"
GMP="$ROOT/lib/libgmp.10.dylib"
GMPXX="$ROOT/lib/libgmpxx.4.dylib"
API_TEST="$ROOT/scripts/aevum-engine-api-load-test"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "This helper is intended for macOS." >&2
  exit 2
fi

xattr -dr com.apple.quarantine "$ROOT" 2>/dev/null || true
[[ -f "$GMP" ]] && codesign --force --sign - "$GMP" >/dev/null 2>&1 || true
[[ -f "$GMPXX" ]] && codesign --force --sign - "$GMPXX" >/dev/null 2>&1 || true
[[ -f "$LIB" ]] && codesign --force --sign - "$LIB" >/dev/null 2>&1 || true
[[ -f "$API_TEST" ]] && codesign --force --sign - "$API_TEST" >/dev/null 2>&1 || true
[[ -f "$BIN" ]] && codesign --force --sign - "$BIN" >/dev/null 2>&1 || true
chmod +x "$BIN" "$API_TEST" 2>/dev/null || true

echo "macOS quarantine removed and ad-hoc signatures refreshed."
echo "Run: $BIN -v"
