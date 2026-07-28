#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-1}"
cd "$ROOT"

run_expect() {
  local expected="$1"; shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" -ne "$expected" ]]; then
    echo "Unexpected exit code $rc (expected $expected): $*" >&2
    exit 1
  fi
}

make test-gm

# Prime proof and composite factor path.
run_expect 0 ./prmers 7 -gm -aevum -gm-sieve 0 -d "$DEVICE"
run_expect 1 ./prmers 13 -gm -aevum -gm-sieve 1000000 -d "$DEVICE"

# Both Legendre-sign branches and independent replay/error injection.
run_expect 0 ./prmers 19 -gm -aevum -gm-sieve 0 -d "$DEVICE"
run_expect 0 ./prmers 113 -gm -aevum -gm-safe -erroriter 20 -gm-sieve 0 -d "$DEVICE"

echo "Gaussian-Mersenne validation passed on OpenCL device $DEVICE"
