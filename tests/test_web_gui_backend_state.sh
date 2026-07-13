#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-gui-state"
mkdir -p "$BUILD"
"${CXX:-g++}" -std=c++20 -O2 -pthread -I"$ROOT/include" \
  "$ROOT/tests/test_web_gui_backend_state.cpp" \
  "$ROOT/src/ui/WebGuiServer.cpp" \
  -o "$BUILD/test_web_gui_backend_state"
"$BUILD/test_web_gui_backend_state"
