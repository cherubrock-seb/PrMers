#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${AEVUM_DEVICES:-0}"
TARGET="${AEVUM_TARGET:-}"
JOBS="${JOBS:-4}"
if [[ "$TARGET" != "rtx3080" && "$TARGET" != "radeonvii" ]]; then
  echo "Set AEVUM_TARGET=rtx3080 or AEVUM_TARGET=radeonvii" >&2; exit 2
fi
OUT="$(mktemp -d "$ROOT/benchmark-results-pass4-${TARGET}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee "$OUT/driver.log") 2>&1
finish() {
  rc=$?
  if (( rc )); then printf 'validation failed rc=%s\n' "$rc" > "$OUT/FAILED.txt"; fi
  # Pass-5 handoff is intentionally compact: structured results, GPU identity,
  # cache records and per-case logs. Transform-sized residue blobs are hashed
  # and deleted by aevum_pass4_validate.py.
  (
    cd "$OUT"
    files=(driver.log system.txt)
    [[ -f summary.json ]] && files+=(summary.json)
    [[ -f devices.txt ]] && files+=(devices.txt)
    [[ -f nvidia.txt ]] && files+=(nvidia.txt)
    [[ -f FAILED.txt ]] && files+=(FAILED.txt)
    while IFS= read -r -d '' f; do files+=("${f#./}"); done < <(find . -maxdepth 1 -type f -name 'cache-*.tsv' -print0 | sort -z)
    while IFS= read -r -d '' f; do files+=("${f#./}"); done < <(find . -mindepth 2 -maxdepth 2 -type f -name 'run.log' -print0 | sort -z)
    zip -q tuning-output.zip "${files[@]}"
  ) || true
  echo "Return: $OUT/tuning-output.zip"
  exit "$rc"
}
trap finish EXIT

uname -a > "$OUT/system.txt"
command -v clinfo >/dev/null && clinfo > "$OUT/devices.txt" || true
command -v nvidia-smi >/dev/null && nvidia-smi -q > "$OUT/nvidia.txt" || true

# Host/source gates first. These do not need GPU execution; OpenCL headers/libs
# are still required below for the production engine build on the target host.
make -C "$ROOT/third_party/aevum" -j"$JOBS" test-host
make -C "$ROOT" test-aevum-auto
python3 "$ROOT/tests/aevum_pass4_source_test.py"
python3 "$ROOT/tests/aevum_pow2_type4_source_test.py"
python3 "$ROOT/tests/gaussian_pair_backend_policy_test.py"

# Production build and real-GPU validation.
make -C "$ROOT/third_party/aevum" -j"$JOBS" engine-lib
make -C "$ROOT" -j"$JOBS" prmers KERNEL_PATH="$ROOT/kernels/"

c++ -O3 -std=c++20 "$ROOT/scripts/aevum_engine_bench.cpp" -ldl -o "$OUT/engine-bench"
python3 "$ROOT/scripts/aevum_pass4_validate.py" \
  --bench "$OUT/engine-bench" \
  --lib "$ROOT/third_party/aevum/build-engine/libaevum_engine.so" \
  --tune-dir "$ROOT/third_party/aevum" \
  --device "$DEVICE" --target "$TARGET" --out "$OUT"
