#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -pthread -I"$ROOT/include" \
  "$ROOT/tests/stop_restart_gate_test.cpp" -o "$BUILD/stop-restart-gate-test"
"$BUILD/stop-restart-gate-test"
