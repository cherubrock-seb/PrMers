#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${AEVUM_TARGET:-rtx5090}"
DEVICE="${AEVUM_DEVICES:-0}"
SMI_DEVICE="${AEVUM_NVIDIA_SMI_ID:-$DEVICE}"
EXP="${AEVUM_TIMING_EXP:-147800003}"
SHAPE="${AEVUM_TIMING_SHAPE:-1:512:8:512:202}"
PROFILE="${AEVUM_TIMING_USE:-INPLACE=1,LOADS=10040,MODM31=2,STORES=22}"
SHORT="${AEVUM_TIMING_SHORT_ITERS:-512}"
WINDOW_ITERS="${AEVUM_TIMING_WINDOW_ITERS:-4096}"
WINDOWS="${AEVUM_TIMING_WINDOWS:-16}"
OUT="$(mktemp -d "$ROOT/issue36-timing-${TARGET}-XXXXXXXX")"
LIB="$ROOT/third_party/aevum/build-engine/libaevum_engine.so"
sampler_pid=''
cleanup_sampler(){ if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; sampler_pid=''; fi; }
on_exit(){
  local rc=$?
  trap - EXIT
  cleanup_sampler
  if ((rc!=0)); then printf 'driver failed: %s\n' "$rc" > "$OUT/FAILED.txt"; fi
  (cd "$OUT" && zip -qr tuning-output.zip . -x tuning-output.zip) || true
  echo "Return: $OUT/tuning-output.zip"
  exit "$rc"
}
trap on_exit EXIT
exec > >(tee "$OUT/driver.log") 2>&1
printf '%s\n' 'baseline_commit=9d560848b625bae4a5aade79b01479ffd960f015' 'baseline_tree=5ba504f55110cba40d564c46eb14e8d0a3e98312' > "$OUT/current-main-base.txt"
uname -a > "$OUT/system.txt"
command -v clinfo >/dev/null && clinfo > "$OUT/devices.txt" 2>&1 || true
make -C "$ROOT/third_party/aevum" -j"${JOBS:-4}" engine-lib > "$OUT/build-engine.log" 2>&1
c++ -O3 -std=c++20 "$ROOT/scripts/aevum_issue36_timing.cpp" -ldl -o "$OUT/timing"

if command -v nvidia-smi >/dev/null 2>&1; then
  (nvidia-smi --id="$SMI_DEVICE" --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1 > "$OUT/gpu-clock-thermal.csv" 2>&1) & sampler_pid=$!
else
  printf 'unavailable\n' > "$OUT/gpu-clock-thermal.csv"
fi
# Deliberately scrub every runtime/manual tuning input. The harness then sets
# only the exact explicit shape/profile above. AEVUM_TUNE_DIR is never injected.
env -u AEVUM_TUNE_DIR -u AEVUM_AUTOTUNE_CACHE -u AEVUM_GB202_TUNE \
  -u AEVUM_PRP_USE -u AEVUM_PRP_USE_TUNE -u PRMERS_AEVUM_FFT \
  "$OUT/timing" "$LIB" "$DEVICE" "$EXP" "$SHAPE" "$ROOT/third_party/aevum" "$PROFILE" \
  "$SHORT" "$WINDOW_ITERS" "$WINDOWS" "$OUT"
cleanup_sampler
python3 - "$OUT/timing-decomposition.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
assert d['exact_timing_on_vs_off']
print('Issue36 timing exactness: PASS')
print('short_us_per_iter=',d['A_raw_square_hot_path']['short_us_per_iter'])
print('long_us_per_iter=',d['A_raw_square_hot_path']['long_us_per_iter'])
print('long_over_short_ratio=',d['G_short_vs_long']['long_over_short_ratio'])
PY
