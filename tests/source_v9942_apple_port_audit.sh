#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A="$ROOT/third_party/aevum"
grep -q '4.20.49-alpha-v99.42-aevum-apple-pm1-smallfactor-safety' "$ROOT/include/core/Version.hpp"
grep -q 'AEVUM_VERSION ?= v0.3.37' "$A/Makefile"
python3 "$ROOT/tests/aevum_pm1_small_factor_policy_test.py"
python3 "$A/tests/apple_generic_mul_safety_test.py"
grep -q 'AEVUM_APPLE_UNSAFE_GENERIC_MUL' "$A/src/EngineApi.cpp"
grep -q '#if defined(__APPLE__)' "$A/src/EngineApi.cpp"
grep -q 'Aevum P-1 Stage 1 would require generic multiplication across multiple chunks' "$ROOT/src/modes/RunPM1.cpp"
# Queue submission fix remains Apple-only.
grep -A4 'markerEvent = enqueueMarker(get());' "$A/src/Queue.cpp" | grep -q '#if defined(__APPLE__)'
# Existing Apple staged kernels are unchanged and still audited.
python3 "$A/tests/apple_gf61_tailmul_staging_test.py"
python3 "$A/tests/apple_gf61_middleout_staging_test.py"
echo 'PrMers v99.42 Apple P-1 small-factor safety source audit passed'
