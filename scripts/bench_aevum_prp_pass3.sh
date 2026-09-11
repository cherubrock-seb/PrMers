#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICES="${AEVUM_DEVICES:-0}"
JOBS="${JOBS:-4}"
OUT="$(mktemp -d "$ROOT/benchmark-results-prp-pass3-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee "$OUT/driver.log") 2>&1
on_exit() {
  local rc=$?
  if (( rc )); then
    [[ -f "$OUT/FAILED.txt" ]] || printf 'Build/driver failure rc=%s; see logs\n' "$rc" > "$OUT/FAILED.txt"
  fi
  python3 "$ROOT/scripts/aevum_prp_pass3.py" pack "$ROOT" "$OUT" || true
  printf '\nReturn: %s/tuning-output.zip\n' "$OUT"
}
trap on_exit EXIT
printf 'PRP PASS 3 FAST results: %s\n' "$OUT"
clinfo > "$OUT/devices.txt"
uname -a > "$OUT/system.txt"
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/nvidia-before.txt" || true; fi
python3 "$ROOT/scripts/aevum_prp_pass3.py" prepare "$ROOT" "$OUT"
for variant in baseline optimized; do
  make -C "$OUT/build/$variant/third_party/aevum" -j"$JOBS" engine-lib > "$OUT/build-$variant.log" 2>&1
done
make -C "$OUT/build/optimized" -j"$JOBS" prmers KERNEL_PATH="$ROOT/kernels/" > "$OUT/build-prmers.log" 2>&1
make -C "$OUT/build/optimized/third_party/aevum" test-host test-engine-api > "$OUT/host-tests.log" 2>&1
make -C "$OUT/build/optimized" test-aevum-reg test-aevum-auto test-aevum-default >> "$OUT/host-tests.log" 2>&1
printf '%s\n' 'NOTE: test-aevum-source skipped in benchmark ZIP because .github/workflows is not packaged; issue #36/v100.10 source state was verified separately.' >> "$OUT/host-tests.log"
python3 "$ROOT/tests/test_aevum_prp_pass3_campaign.py" >> "$OUT/host-tests.log"
python3 "$ROOT/tests/test_aevum_prp_pass3_plans.py" "$OUT/build/optimized/third_party/aevum/build-engine/libaevum_engine.so" >> "$OUT/host-tests.log"
c++ -O2 -std=c++20 "$ROOT/third_party/aevum/tests/prp_middle1_twiddle_test.cpp" \
  "$OUT/build/optimized/third_party/aevum/build-engine/"*.o -lOpenCL -ldl -o "$OUT/middle1-twiddle-test"
"$OUT/middle1-twiddle-test" >> "$OUT/host-tests.log"
if ! bash "$ROOT/scripts/test_prp_pass3_kernel_syntax.sh" >> "$OUT/host-tests.log" 2>&1; then
  export AEVUM_SKIP_PRP_MIDDLE1=1
  printf 'Reject fused-middle candidate: syntax gate failed. Plan screening continues.\n'
fi
c++ -O3 -std=c++20 "$ROOT/scripts/aevum_engine_bench.cpp" -ldl -o "$OUT/engine-bench"
python3 "$ROOT/scripts/aevum_prp_pass3.py" run "$ROOT" "$OUT" "$DEVICES"
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/nvidia-after.txt" || true; fi
