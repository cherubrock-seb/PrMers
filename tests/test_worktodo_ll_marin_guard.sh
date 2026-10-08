#!/usr/bin/env bash
# -marin selects the legacy internal NTT path, which is not validated for Lucas-Lehmer. The command-line
# check in main.cpp used to be the only guard: a worktodo Test= LL line set mode "ll" later (App.cpp) and
# ran on that path. Every case below must be rejected with exit status 2 before any OpenCL device is
# touched, so this test never needs a GPU. With -allow-unvalidated-legacy-ll the same cases are let through
# and warn; the warning is printed before any device is opened, so those cases only check that the run
# gets past the guard (they are stopped by a short timeout or fail later without a device).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
BUILD="$ROOT/tests/build-legacy-ll-guard"
WORK="$(mktemp -d)"
rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$WORK" "$BUILD"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/legacy_ll_guard_test.cpp" -o "$BUILD/legacy-ll-guard-test"
"$BUILD/legacy-ll-guard-test"

AID=0123456789ABCDEF0123456789ABCDEF
fail=0
expect_reject() {
  local name="$1" line="$2"
  shift 2
  mkdir -p "$WORK/$name"
  printf '%s\n' "$line" > "$WORK/$name/worktodo.txt"
  set +e
  (cd "$WORK/$name" && timeout 60 "$BIN" "$@" -worktodo worktodo.txt > out.log 2>&1)
  local rc=$?
  set -e
  if [[ "$rc" != 2 ]] || ! grep -q 'not validated for Lucas-Lehmer' "$WORK/$name/out.log" \
     || grep -q 'Transform Size' "$WORK/$name/out.log"; then
    echo "FAIL $name: rc=$rc" >&2
    cat "$WORK/$name/out.log" >&2
    fail=1
  else
    echo "ok $name"
  fi
}

expect_reject plain          'Test=N/A,216091,60,1' -marin
expect_reject aid            "Test=$AID,216091,60,1" -marin
expect_reject kbnc           'Test=1,2,127,-1' -marin
expect_reject lowest-exp     'Test=N/A,89,1,1' -marin
expect_reject big-exp        'Test=N/A,4294967291,60,1' -marin
expect_reject iterforce      'Test=N/A,216091,60,1' -marin -iterforce 10
# The same flag coming from a -config file.
mkdir -p "$WORK/cfg"
printf -- '-marin\n' > "$WORK/cfg/settings.cfg"
expect_reject config         'Test=N/A,216091,60,1' -config "$WORK/cfg/settings.cfg"

# The command-line check still fires too.
set +e
(cd "$WORK" && timeout 60 "$BIN" 216091 -llunsafe -marin > cli.log 2>&1)
rc=$?
set -e
if [[ "$rc" != 2 ]] || ! grep -q 'not validated for Lucas-Lehmer' "$WORK/cli.log"; then
  echo "FAIL cli: rc=$rc" >&2; cat "$WORK/cli.log" >&2; fail=1
else
  echo "ok cli"
fi

# With the opt-in, both sources get past the guard, and both warn exactly once.
expect_allow() {
  local name="$1" line="$2"
  shift 2
  mkdir -p "$WORK/$name"
  ln -sfn "$ROOT/kernels" "$WORK/$name/kernels"
  printf '%s\n' "$line" > "$WORK/$name/worktodo.txt"
  set +e
  (cd "$WORK/$name" && timeout -s INT 15 "$BIN" "$@" -allow-unvalidated-legacy-ll -noask -worktodo worktodo.txt > out.log 2>&1)
  local rc=$?
  set -e
  local warns
  warns=$(grep -c 'Warning: -allow-unvalidated-legacy-ll: .*not validated for Lucas-Lehmer' "$WORK/$name/out.log" || true)
  if grep -q 'cannot use the legacy internal' "$WORK/$name/out.log" || [[ "$warns" != 1 ]]; then
    echo "FAIL $name: rc=$rc warnings=$warns" >&2
    cat "$WORK/$name/out.log" >&2
    fail=1
  else
    echo "ok $name"
  fi
}

expect_allow allow-worktodo 'Test=N/A,2203,60,1' -marin
mkdir -p "$WORK/allow-cfg"
printf -- '-marin\n-allow-unvalidated-legacy-ll\n' > "$WORK/allow-cfg/settings.cfg"
expect_allow allow-config 'Test=N/A,2203,60,1' -config "$WORK/allow-cfg/settings.cfg"
# Command line: -llunsafe -marin with an exponent and no worktodo entry.
mkdir -p "$WORK/allow-cli"
ln -sfn "$ROOT/kernels" "$WORK/allow-cli/kernels"
set +e
(cd "$WORK/allow-cli" && timeout -s INT 15 "$BIN" 2203 -llunsafe -marin -allow-unvalidated-legacy-ll -noask > out.log 2>&1)
rc=$?
set -e
warns=$(grep -c 'Warning: -allow-unvalidated-legacy-ll: .*not validated for Lucas-Lehmer' "$WORK/allow-cli/out.log" || true)
if grep -q 'cannot use the legacy internal' "$WORK/allow-cli/out.log" || [[ "$warns" != 1 ]]; then
  echo "FAIL allow-cli: rc=$rc warnings=$warns" >&2; cat "$WORK/allow-cli/out.log" >&2; fail=1
else
  echo "ok allow-cli"
fi
# A PRP entry with the opt-in does not warn.
mkdir -p "$WORK/allow-prp"
ln -sfn "$ROOT/kernels" "$WORK/allow-prp/kernels"
printf 'PRP=N/A,1,2,2203,-1,60,0\n' > "$WORK/allow-prp/worktodo.txt"
set +e
(cd "$WORK/allow-prp" && timeout -s INT 15 "$BIN" -marin -allow-unvalidated-legacy-ll -noask -worktodo worktodo.txt > out.log 2>&1)
set -e
if grep -q 'Warning: -allow-unvalidated-legacy-ll' "$WORK/allow-prp/out.log"; then
  echo "FAIL allow-prp: warned for a PRP entry" >&2; cat "$WORK/allow-prp/out.log" >&2; fail=1
else
  echo "ok allow-prp"
fi

if [[ "$fail" != 0 ]]; then exit 1; fi
echo "Worktodo LL -marin guard tests passed"
