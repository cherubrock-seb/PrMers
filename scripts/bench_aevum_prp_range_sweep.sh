#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICES="${AEVUM_DEVICES:-0}"
JOBS="${JOBS:-4}"
OUT="$(mktemp -d "$ROOT/benchmark-results-prp-range-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee "$OUT/driver.log") 2>&1
on_exit() {
  local rc=$?
  if (( rc )); then
    [[ -f "$OUT/FAILED.txt" ]] || printf 'Build/driver failure rc=%s; see logs\n' "$rc" > "$OUT/FAILED.txt"
  fi
  python3 "$ROOT/scripts/aevum_prp_range_sweep.py" pack "$ROOT" "$OUT" || true
  printf '\nReturn: %s/tuning-output.zip\n' "$OUT"
  [[ -f "$OUT/prp-range-summary.csv" ]] && printf 'Summary: %s/prp-range-summary.csv\n' "$OUT"
}
trap on_exit EXIT

printf 'AEVUM PRP RANGE SWEEP results: %s\n' "$OUT"
printf 'Reference: TRUE AUTO on the same Pass-3 source, never a forced Type4 baseline.\n'
printf 'No full/Gerbicz PRP is run; this is a word-exact engine plan map.\n'

clinfo > "$OUT/devices.txt" 2>&1 || true
uname -a > "$OUT/system.txt"
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/nvidia-before.txt" || true; fi

python3 "$ROOT/scripts/aevum_prp_range_sweep.py" prepare "$ROOT" "$OUT"
make -C "$OUT/build/optimized/third_party/aevum" -j"$JOBS" engine-lib > "$OUT/build-engine.log" 2>&1
make -C "$OUT/build/optimized/third_party/aevum" test-host test-engine-api > "$OUT/host-tests.log" 2>&1
c++ -O3 -std=c++20 "$ROOT/scripts/aevum_engine_bench.cpp" -ldl -o "$OUT/engine-bench"

python3 "$ROOT/scripts/aevum_prp_range_sweep.py" run "$ROOT" "$OUT" "$DEVICES"

if command -v nvidia-smi >/dev/null; then nvidia-smi -q > "$OUT/nvidia-after.txt" || true; fi
