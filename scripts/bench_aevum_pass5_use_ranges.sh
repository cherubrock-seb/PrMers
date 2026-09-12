#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${AEVUM_TARGET:-rtx3080}"
DEVICE="${AEVUM_DEVICES:-0}"
case "$TARGET" in rtx3080|radeonVII|rtx5090) ;; *) echo 'Use AEVUM_TARGET=rtx3080, radeonVII, or rtx5090' >&2; exit 2;; esac
OUT="$(mktemp -d "$ROOT/pass5-r5-ranges-${TARGET}-XXXXXXXX")"
exec > >(tee "$OUT/driver.log") 2>&1
finish() {
  local rc=$?
  ((rc==0)) || printf 'driver/build failed: %s\n' "$rc" > "$OUT/FAILED.txt"
  python3 "$ROOT/scripts/pass5_validate.py" pack "$ROOT" "$OUT" "$DEVICE" || true
  echo "Return: $OUT/tuning-output.zip"
}
trap finish EXIT
uname -a > "$OUT/system.txt"
command -v clinfo >/dev/null && clinfo > "$OUT/devices.txt" || true
python3 "$ROOT/scripts/pass5_validate.py" prepare "$ROOT" "$OUT" "$DEVICE"
make -C "$OUT/baseline" -j"${JOBS:-4}" engine-lib > "$OUT/build-baseline.log" 2>&1
make -C "$ROOT/third_party/aevum" -j"${JOBS:-4}" engine-lib test-prp-use > "$OUT/host-tests.log" 2>&1
python3 "$ROOT/tests/pass5_source_test.py" >> "$OUT/host-tests.log" 2>&1
python3 "$ROOT/tests/pass5_hwfix_test.py" >> "$OUT/host-tests.log" 2>&1
python3 "$ROOT/tests/pass5_campaign_test.py" >> "$OUT/host-tests.log" 2>&1
c++ -O3 -std=c++20 "$ROOT/scripts/aevum_engine_bench.cpp" -ldl -o "$OUT/bench"
python3 "$ROOT/scripts/pass5_validate.py" ranges "$ROOT" "$OUT" "$DEVICE" "$TARGET"
