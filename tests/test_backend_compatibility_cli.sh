#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"

expect_reject() {
  local name="$1" required="$2"
  shift 2
  local log
  log="$(mktemp)"
  set +e
  "$BIN" "$@" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" != 2 ]]; then
    echo "$name: expected exit 2, got $rc" >&2
    cat "$log" >&2
    rm -f "$log"
    exit 1
  fi
  if ! grep -Eq "$required" "$log"; then
    echo "$name: missing rejection text" >&2
    cat "$log" >&2
    rm -f "$log"
    exit 1
  fi
  if grep -q 'Transform Size' "$log"; then
    echo "$name: rejection happened after transform construction" >&2
    cat "$log" >&2
    rm -f "$log"
    exit 1
  fi
  rm -f "$log"
}

expect_reject ultralow-aevum 'cannot be forced to Aevum' \
  2147483647 -pm1 -b1 100 -b2 5000 -pm1-ultralowmem -aevum

expect_reject llunsafe-internal 'not validated for Lucas-Lehmer' \
  216091 -llunsafe -marin

expect_reject small-forced-aevum 'Forced Aevum request cannot be satisfied' \
  216091 -ll -aevum

echo "Backend compatibility CLI tests passed"
