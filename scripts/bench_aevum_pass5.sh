#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${AEVUM_TARGET:-rtx3080}"
DEVICE="${AEVUM_DEVICES:-0}"
OUT="$(mktemp -d "$ROOT/pass5-${TARGET}-XXXXXXXX")"
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
command -v nvidia-smi >/dev/null && nvidia-smi -q > "$OUT/nvidia.txt" || true
python3 "$ROOT/scripts/pass5_validate.py" prepare "$ROOT" "$OUT" "$DEVICE"
make -C "$OUT/baseline" -j"${JOBS:-4}" engine-lib > "$OUT/build-baseline.log" 2>&1
make -C "$ROOT/third_party/aevum" -j"${JOBS:-4}" engine-lib test-host test-engine-api test-prp-use > "$OUT/host-tests.log" 2>&1
python3 "$ROOT/tests/aevum_pass4_source_test.py" >> "$OUT/host-tests.log"
python3 "$ROOT/tests/pass5_source_test.py" >> "$OUT/host-tests.log"
python3 "$ROOT/tests/pass5_campaign_test.py" >> "$OUT/host-tests.log"
make -C "$ROOT" test-aevum-auto test-aevum-default >> "$OUT/host-tests.log" 2>&1
make -C "$ROOT" -j"${JOBS:-4}" prmers KERNEL_PATH="$ROOT/kernels/" > "$OUT/build-prmers.log" 2>&1
if ! bash "$ROOT/scripts/test_pass5_kernel_syntax.sh" > "$OUT/epoch-syntax.log" 2>&1; then
  export AEVUM_PASS5_SKIP_EPOCH=1
  echo 'Structural candidate rejected by syntax gate; implementation campaign continues.'
fi
c++ -O3 -std=c++20 "$ROOT/scripts/aevum_engine_bench.cpp" -ldl -o "$OUT/bench"
python3 "$ROOT/scripts/pass5_validate.py" run "$ROOT" "$OUT" "$DEVICE" "${1:-full}"
