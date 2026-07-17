#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
  echo "check_macos_minos.sh is intended for macOS" >&2
  exit 2
fi

TARGET="${1:?usage: $0 MAX_VERSION FILE...}"
shift
[[ $# -gt 0 ]] || { echo "No Mach-O files supplied" >&2; exit 2; }

version_le() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, A, "."); split(b, B, ".");
    for (i = 1; i <= 3; ++i) {
      av = (A[i] == "" ? 0 : A[i]) + 0;
      bv = (B[i] == "" ? 0 : B[i]) + 0;
      if (av < bv) exit 0;
      if (av > bv) exit 1;
    }
    exit 0;
  }'
}

for file in "$@"; do
  [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
  minos="$(vtool -show-build "$file" 2>/dev/null | awk '/^[[:space:]]*minos /{print $2; exit}')"
  [[ -n "$minos" ]] || { echo "Cannot determine minos for $file" >&2; exit 1; }
  printf '%s: minos=%s (maximum=%s)
' "$file" "$minos" "$TARGET"
  if ! version_le "$minos" "$TARGET"; then
    echo "ERROR: $file requires macOS $minos, newer than allowed $TARGET" >&2
    exit 1
  fi
done
