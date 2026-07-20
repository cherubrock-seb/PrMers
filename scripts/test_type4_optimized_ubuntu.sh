#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-1}"
SECONDS="${2:-180}"
EXPONENT="${3:-175000039}"
cd "$ROOT"

make -j"$(nproc)" KERNEL_PATH=./kernels/

echo
echo '===== word-exact full type-4 differential validation ====='
./third_party/aevum/scripts/test_type4_pfa9_ubuntu.sh "$DEVICE" "$EXPONENT" 2

run_bench() {
  local name="$1"; shift
  local dir="$ROOT/bench-$name"
  rm -rf "$dir"; mkdir -p "$dir"
  echo
  echo "===== $name ====="
  set +e
  timeout -s INT "${SECONDS}s" "$@" -f "$dir" 2>&1 | tee "$dir/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then exit "$rc"; fi
}

# Convenience option: the type-4 request is optimized after an exact BPW check.
run_bench adaptive-option ./prmers "$EXPONENT" -d "$DEVICE" -pfa9-type4 -proof 0

# The explicit type-4 plan must resolve to the same fast execution.
run_bench adaptive-explicit ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:4:512:9:512:202 -proof 0

# Full three-plane path, optimized with queue overlap.
run_bench full-multiq env AEVUM_TYPE4_MULTI_Q=1 ./prmers "$EXPONENT" -d "$DEVICE" -pfa9-type4-full -proof 0

# Baseline proving the effect of queue overlap.
run_bench full-singleq env AEVUM_TYPE4_MULTI_Q=0 ./prmers "$EXPONENT" -d "$DEVICE" -pfa9-type4-full -proof 0

# Direct paired-NTT reference.
run_bench type1 ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9:1:512:9:512:202 -proof 0

echo
echo '===== stabilized comparison ====='
for log in bench-*/run.log; do
  printf '%-38s\n' "$log"
  grep -E 'optimized type-4|concurrent GF61|FFT:|Progress:' "$log" | tail -8 || true
done
