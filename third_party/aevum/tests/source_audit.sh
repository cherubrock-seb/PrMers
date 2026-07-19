#!/usr/bin/env bash
set -euo pipefail
trap 'echo "::error file=${BASH_SOURCE[0]},line=${LINENO}::${BASH_COMMAND}" >&2' ERR
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
grep -q '#include <sstream>' "$ROOT/src/fs.cpp"
! grep -q 'std::jthread' "$ROOT/src/Background.h"
grep -q 'thread.joinable()' "$ROOT/src/Background.h"
! grep -q 'namespace std::filesystem' "$ROOT/src/common.h"
grep -q 'ENGINE_LINK_FLAGS = -dynamiclib' "$ROOT/Makefile"

grep -q '#include <sstream>' "$ROOT/src/Gpu.cpp"
grep -q 'string toLiteral(double value)' "$ROOT/src/Gpu.cpp"
! grep -R -q '\bjthread\b' "$ROOT/src"
grep -q 'ThreadJoiner joiner' "$ROOT/src/main.cpp"


grep -q 'fftWGF61ApplePlaceholder' "$ROOT/src/cl/fftw.cl"
grep -q 'fftWGF61LoadScalarApple' "$ROOT/src/cl/fftw.cl"
grep -q 'fftWGF61WidthRadixApple' "$ROOT/src/cl/fftw.cl"
grep -q 'fftWGF61TwiddleShuffle1Apple' "$ROOT/src/cl/fftw.cl"
grep -q 'fftWGF61WidthFinalApple' "$ROOT/src/cl/fftw.cl"
grep -Fq 'kfftWGF61LoadScalarApple(*out, *in);' "$ROOT/src/Gpu.cpp"
! grep -q 'bufAppleFftWGF61' "$ROOT/src/Gpu.cpp"
python3 "$ROOT/tests/apple_gf61_fftw_staging_test.py"
python3 "$ROOT/tests/apple_gf61_tailmul_staging_test.py"
python3 "$ROOT/tests/apple_prepared_tailmullow_staging_test.py"
python3 "$ROOT/tests/apple_gf61_middleout_staging_test.py"
python3 "$ROOT/tests/apple_readchecked_double_sync_test.py"
python3 "$ROOT/tests/apple_queue_marker_flush_test.py"
python3 "$ROOT/tests/apple_gf61_ffthin_staging_test.py"
python3 "$ROOT/tests/apple_generic_mul_safety_test.py"


grep -q 'fftP31WeightScalarApple' "$ROOT/src/cl/fftp.cl"
grep -q 'fftP31WidthRadixApple' "$ROOT/src/cl/fftp.cl"
grep -q 'fftP31WidthFinalApple' "$ROOT/src/cl/fftp.cl"
grep -q 'runGF31();' "$ROOT/src/Gpu.cpp"
grep -q 'runGF61();' "$ROOT/src/Gpu.cpp"
! grep -q 'kfftP31Apple(\*out, in);' "$ROOT/src/Gpu.cpp"
python3 "$ROOT/tests/apple_gf31_width_staging_test.py"
python3 "$ROOT/tests/apple_fused_fftp_test.py"
python3 "$ROOT/tests/apple_plane_isolation_gate_test.py"
grep -q 'transposeOutAppleGlobal' "$ROOT/src/cl/transpose.cl"
grep -q 'transposeInAppleGlobal' "$ROOT/src/cl/transpose.cl"
grep -Fq 'K(transpOut, "transpose.cl", "transposeOutAppleGlobal", hN)' "$ROOT/src/Gpu.cpp"
python3 "$ROOT/tests/apple_global_transpose_test.py"


grep -Eq '^AEVUM_VERSION[[:space:]]+\?=[[:space:]]+v[0-9]+\.[0-9]+\.[0-9]+' "$ROOT/Makefile"
grep -q 'OpenCLSourceBuilder.cpp' "$ROOT/Makefile"
grep -q 'buildMonolithicOpenCLSource' "$ROOT/src/KernelCompiler.cpp"
grep -q 'clBuildProgram' "$ROOT/src/KernelCompiler.cpp"
grep -q 'AEVUM_APPLE_OPENCL12=1' "$ROOT/src/KernelCompiler.cpp"
grep -q 'Apple OpenCL pipeline retry' "$ROOT/src/KernelCompiler.cpp"
grep -q '__OPENCL_C_VERSION__ < 200' "$ROOT/src/cl/base.cl"
grep -Fq 'shufl32((local F2 *) lds, (F2 *) u' "$ROOT/src/cl/fftbase.cl"
python3 "$ROOT/tests/gpu_init_order_test.py"

grep -q 'AEVUM_READY_STORE' "$ROOT/src/cl/carryfused.cl"
grep -q 'atomic_xchg((volatile global uint \*) (ptr), 1u)' "$ROOT/src/cl/carryfused.cl"
grep -q 'atomic_add((volatile global uint \*) (ptr), 0u)' "$ROOT/src/cl/carryfused.cl"
grep -q 'apple_opencl12_kernel_matrix_syntax.sh' "$ROOT/Makefile"

grep -q 'fftMiddleInGF61LoadScalarApple' "$ROOT/src/cl/fftmiddlein.cl"
grep -q 'fftMiddleInGF61Mul2FactorScalarApple' "$ROOT/src/cl/fftmiddlein.cl"
grep -q 'fftMiddleInGF61ApplyScalarApple' "$ROOT/src/cl/fftmiddlein.cl"
grep -q 'fftMiddleInGF61FftApple' "$ROOT/src/cl/fftmiddlein.cl"
grep -q 'fftMiddleInGF61MulApple' "$ROOT/src/cl/fftmiddlein.cl"
grep -q 'fftMiddleInGF61TransposeApple' "$ROOT/src/cl/fftmiddlein.cl"
grep -q 'local Z61 lds' "$ROOT/src/cl/fftmiddlein.cl"
python3 "$ROOT/tests/apple_gf61_middlein_staging_test.py"
python3 "$ROOT/tests/apple_gf61_middlein_restrict_alias_test.py"

# Apple FFT3161 keeps the double-wide LDS tail algorithm but uses the existing
# two-kernel mode so the special lines are isolated from tailSquareGF61.
grep -q 'forcing TAIL_KERNELS=3 (double-wide, two kernels)' "$ROOT/src/Gpu.cpp"
grep -Fq 'config["TAIL_KERNELS"] = "3";' "$ROOT/src/Gpu.cpp"
grep -q 'tail_single_wide = false;' "$ROOT/src/Gpu.cpp"
grep -q 'tail_single_kernel = false;' "$ROOT/src/Gpu.cpp"
grep -q 'local GF61 lds\[2 \* LDS_BYTES / sizeof(GF61)\]' "$ROOT/src/cl/tailsquare.cl"
grep -q 'tailSquareZeroGF61' "$ROOT/src/cl/tailsquare.cl"
python3 "$ROOT/tests/apple_gf61_tail_two_kernel_policy_test.py"
# Apple FFT3161 stages the two exceptional GF61 tail lines through a tiny
# scratch buffer.  The stock main double-wide LDS kernel remains in source for
# non-Apple builds, while Apple stages normal lines through the existing in/out
# transform buffers without a new transform-sized allocation.
grep -q 'tailSquareZeroGF61LoadApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'tailSquareZeroGF61FftRadixApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'tailSquareZeroGF61FftTwiddleApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'tailSquareZeroGF61FftShuffleApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'tailSquareZeroGF61FftFinalApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'appleTailZeroGF61BankIndex' "$ROOT/src/cl/tailsquare.cl"
grep -q 'BUF(bufAppleTailZeroGF61, fft.NTT_GF61 ? 8 \* SMALL_H : 0)' "$ROOT/src/Gpu.cpp"
grep -q 'tailSquareZeroGF61ReverseGlobalApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'tailSquareZeroGF61PairApple(P(GF61) scratch, Trig smallTrig, u32 bank)' "$ROOT/src/cl/tailsquare.cl"
grep -Fq 'ktailSquareZeroGF61PairApple(bufAppleTailZeroGF61, bufTrigH, 1u);' "$ROOT/src/Gpu.cpp"
! grep -Fq 'ktailSquareZeroGF61PairApple.setFixedArgs(1, bufTrigH);' "$ROOT/src/Gpu.cpp"
grep -q 'tailSquareZeroGF61WriteDirectApple' "$ROOT/src/cl/tailsquare.cl"
grep -Fq 'out61[DISTGF61 + outBase + offset] = scratch[scratchBase + offset];' "$ROOT/src/cl/tailsquare.cl"
! grep -q 'writeTailFusedValue(value' "$ROOT/src/cl/tailsquare.cl"
grep -Fq 'ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, 0u, 0u);' "$ROOT/src/Gpu.cpp"
grep -Fq 'ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, SMALL_H, (fft.shape.middle / 2u) * SMALL_H);' "$ROOT/src/Gpu.cpp"
! grep -q 'KERNEL(G_H) tailSquareZeroGF61WriteApple' "$ROOT/src/cl/tailsquare.cl"
grep -q 'bufAppleTailZeroGF61' "$ROOT/src/Gpu.cpp"
python3 "$ROOT/tests/apple_gf61_tailzero_staging_test.py"
python3 "$ROOT/tests/apple_gf61_maintail_staging_test.py"


grep -Fq 'writeIn(dst, makeWords(E, value));' "$ROOT/src/Gpu.cpp"
grep -q 'set_u32 uses canonical compact-word upload' "$ROOT/src/EngineApi.cpp"

# The v0.3.56 release keeps the validated v0.3.54 tailSquare pipeline.
# The bridge kernel was referenced by C++ but its OpenCL implementation was
# not shipped, so it must remain disabled.
! grep -q 'tailSquareGF61FinalPairFirstFusedApple' "$ROOT/src/cl/tailsquare.cl"
grep -Fq 'apple_bridge_fused_tailsquare_gf61 = false;' "$ROOT/src/Gpu.cpp"
grep -q 'AEVUM_APPLE_TAILSQUARE_V54' "$ROOT/src/Gpu.cpp"
grep -q 'apple_bridge_fused_tailsquare_gf61' "$ROOT/src/Gpu.h"

echo "Aevum source audit passed"
