#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
grep -q 'for (enum FFT_TYPES type : {FFT3161})' "$ROOT/src/FFTConfig.cpp"
grep -q 'json("name", "aevum")' "$ROOT/src/Task.cpp"
grep -q 'Aevum %s starting' "$ROOT/src/main.cpp"
grep -q 'Aevum auto FFT:' "$ROOT/src/FFTConfig.cpp"
! grep -q 'vector<TuneEntry> tunes = TuneEntry::readTuneFile(args)' "$ROOT/src/FFTConfig.cpp"
if grep -A5 "makeTransformBufVector" "$ROOT/src/Gpu.cpp" | grep -q "&queue, N"; then
  echo "Prepared transform buffers still use the integer buffer size" >&2
  exit 1
fi

grep -q "TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)" "$ROOT/src/Gpu.cpp"
grep -q "bits_per_word < fft.minBpw()" "$ROOT/src/FFTConfig.cpp"
grep -q "multiply_small" "$ROOT/src/EngineApi.cpp"
grep -q "small_factor_scratch_" "$ROOT/src/EngineApi.cpp"
grep -q "increment_mod_mersenne" "$ROOT/src/state.cpp"
grep -q "aevum_engine_equal" "$ROOT/src/EngineApi.cpp"
! grep -R -q "regScale\|REGSCALE" "$ROOT/src"
grep -q "294cc485ac8cf53c8b69144a3039832eda573849" "$ROOT/UPSTREAM.md"

echo "Aevum source audit passed"
