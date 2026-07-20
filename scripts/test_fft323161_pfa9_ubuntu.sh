#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
SECONDS="${2:-360}"
EXPONENT="${3:-175000039}"
cd "$ROOT"

printf '\n[1/3] Building PrMers and Aevum\n'
make clean-all || true
make -j"$(nproc)" KERNEL_PATH=./kernels/

printf '\n[2/3] Word-exact type-1/type-4 GPU differential test\n'
./third_party/aevum/scripts/test_type4_pfa9_ubuntu.sh "$DEVICE" "$EXPONENT" 2

run_bench() {
  local name="$1" spec="$2" dir="$ROOT/bench-$name"
  rm -rf "$dir"
  mkdir -p "$dir"
  printf '\n[3/3] Benchmark %s for %ss\n' "$name" "$SECONDS"
  set +e
  timeout -s INT "${SECONDS}s" ./prmers "$EXPONENT" -d "$DEVICE" \
    -aevum-fft "$spec" -proof 0 -f "$dir" 2>&1 | tee "$dir/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then
    echo "benchmark $name failed with status $rc" >&2
    exit "$rc"
  fi
}

run_bench pfa9-type1 pfa9:1:512:9:512:202
run_bench pfa9-type4-adaptive pfa9:4:512:9:512:202
run_bench pfa9-type4-full pfa9full:4:512:9:512:202

printf '\n===== comparison =====\n'
for log in bench-pfa9-type1/run.log bench-pfa9-type4-adaptive/run.log bench-pfa9-type4-full/run.log; do
  echo "--- $log"
  grep -E 'FFT:|Aevum experimental type-4 PFA active|Progress:' "$log" | tail -6 || true
done
