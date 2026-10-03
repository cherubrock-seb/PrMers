#!/usr/bin/env bash
# End-to-end HTTP checks of the GUI server (token, Host/Origin, results path, request limits).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-gui-http"
mkdir -p "$BUILD"
"${CXX:-g++}" -std=c++20 -O2 -pthread -I"$ROOT/include" \
  "$ROOT/tests/test_web_gui_http.cpp" \
  "$ROOT/src/ui/WebGuiServer.cpp" \
  -o "$BUILD/test_web_gui_http"
cd "$BUILD" && ./test_web_gui_http
