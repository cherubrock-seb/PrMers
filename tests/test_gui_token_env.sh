#!/usr/bin/env bash
# The GUI access token (PRMERS_GUI_TOKEN) must stay out of the environment of child processes (Prime95,
# shell commands). No OpenCL device needed. POSIX only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-gui-token-env"
rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-g++}" -std=c++20 -O2 -pthread -Wall -Wextra -I"$ROOT/include" \
  "$ROOT/tests/gui_token_env_test.cpp" \
  "$ROOT/src/ui/WebGuiServer.cpp" \
  -o "$BUILD/gui-token-env-test"

OUT="$("$BUILD/gui-token-env-test")"
echo "$OUT"
if ! grep -q 'CHILD env=0123456789abcdef0123456789abcdef token=0123456789abcdef0123456789abcdef' <<<"$OUT"; then
  echo "FAIL: the relaunched program did not get the token" >&2
  exit 1
fi
if grep -q '^FAIL' <<<"$OUT"; then exit 1; fi

# restart_self must export the token only around the relaunch, and nothing else may put it back.
HDR="$ROOT/include/core/AlgoUtils.hpp"
exp_line=$(grep -n 'exportTokenForRestart' "$HDR" | head -1 | cut -d: -f1)
exec_line=$(grep -n 'util::execSelf(args)' "$HDR" | head -1 | cut -d: -f1)
cp_line=$(grep -n 'CreateProcessA' "$HDR" | head -1 | cut -d: -f1)
[[ -n "$exp_line" && -n "$exec_line" && -n "$cp_line" && "$exp_line" -lt "$cp_line" && "$exp_line" -lt "$exec_line" ]] \
  || { echo "FAIL: restart_self does not export the token before relaunching" >&2; exit 1; }
if [[ "$(grep -c 'clearTokenEnv' "$HDR")" -lt 2 ]]; then
  echo "FAIL: restart_self does not clear the token when the relaunch fails" >&2; exit 1
fi
if grep -rn 'setenv(\s*"PRMERS_GUI_TOKEN"\|_putenv_s(\s*"PRMERS_GUI_TOKEN"' "$ROOT/src" "$ROOT/include" | grep -v 'kTokenEnv'; then
  echo "FAIL: something puts PRMERS_GUI_TOKEN in the environment directly" >&2; exit 1
fi
if grep -rn 'exportTokenForRestart' "$ROOT/src" "$ROOT/include" | grep -v 'WebGuiServer\|AlgoUtils.hpp'; then
  echo "FAIL: exportTokenForRestart is used outside the restart helper" >&2; exit 1
fi
echo "GUI token environment test passed"
