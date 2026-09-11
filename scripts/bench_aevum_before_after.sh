#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICES="${AEVUM_DEVICES:-0}"
JOBS="${JOBS:-4}"
OUT="$(mktemp -d "$ROOT/benchmark-results-pass2-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee "$OUT/driver.log") 2>&1
on_exit() {
  local rc=$?
  if (( rc )); then
    [[ -f "$OUT/FAILED.txt" ]] || printf 'Build/driver failure rc=%s; see logs\n' "$rc" > "$OUT/FAILED.txt"
  fi
  python3 "$ROOT/scripts/aevum_bench_runner.py" pack "$ROOT" "$OUT" || true
  printf '\nReturn: %s/tuning-output.zip\n' "$OUT"
}
trap on_exit EXIT
printf 'PASS 2 FAST results: %s\n' "$OUT"
clinfo > "$OUT/devices.txt"
uname -a > "$OUT/system.txt"
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/nvidia-before.txt" || true; fi
python3 "$ROOT/scripts/aevum_bench_runner.py" prepare "$ROOT" "$OUT"
for variant in baseline optimized; do
  make -C "$OUT/build/$variant/third_party/aevum" -j"$JOBS" engine-lib > "$OUT/build-$variant.log" 2>&1
done
make -C "$OUT/build/optimized" -j"$JOBS" prmers KERNEL_PATH="$ROOT/kernels/" > "$OUT/build-prmers.log" 2>&1
make -C "$OUT/build/optimized/third_party/aevum" test-host test-engine-api > "$OUT/host-tests.log" 2>&1
make -C "$OUT/build/optimized" test-aevum-reg test-aevum-auto test-aevum-default test-aevum-source >> "$OUT/host-tests.log" 2>&1
c++ -O2 -std=c++20 "$ROOT/third_party/aevum/tests/gf61_limb_test.cpp" -o "$OUT/gf61-test"
"$OUT/gf61-test" >> "$OUT/host-tests.log"
python3 "$ROOT/tests/test_aevum_fast_campaign.py" >> "$OUT/host-tests.log"
if ! bash "$ROOT/scripts/test_pass2_kernel_syntax.sh" >> "$OUT/host-tests.log" 2>&1; then
  export AEVUM_SKIP_GF61_LIMB32=1
  printf 'Reject GF61 limb candidate: syntax gate failed. Carry screening continues.\n'
fi
c++ -O3 -std=c++20 "$ROOT/scripts/aevum_engine_bench.cpp" -ldl -o "$OUT/engine-bench"
python3 "$ROOT/scripts/aevum_bench_runner.py" run "$ROOT" "$OUT" "$DEVICES"
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/nvidia-after.txt" || true; fi
