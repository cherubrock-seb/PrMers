#!/usr/bin/env bash
# Build and run the pair95 stage-2 checkpoint header host test (no OpenCL device needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Wsign-conversion -Werror \
  -I"$ROOT/include" \
  "$ROOT/tests/pm1_vtrace_ckpt_header_test.cpp" \
  -o "$BUILD/pm1-vtrace-ckpt-header-test"

"$BUILD/pm1-vtrace-ckpt-header-test"
