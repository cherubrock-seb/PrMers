#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLANG_BIN="${CLANG:-$(command -v clang || true)}"
if [[ -z "$CLANG_BIN" ]]; then
  echo 'WARNING: clang not found; power-of-two type-4 syntax test skipped.' >&2
  exit 0
fi
mkdir -p "$ROOT/build-tests"
DUMPER="$ROOT/build-tests/aevum-opencl-monolithic-source-test"
rm -f "$DUMPER"
"${CXX:-c++}" -O2 -std=c++20 "$ROOT/tests/opencl_monolithic_source_test.cpp" \
  "$ROOT/src/OpenCLSourceBuilder.cpp" -o "$DUMPER"
roots=(fftp fftmiddlein tailsquare tailmul fftmiddleout fftw carry carryfused carryb)
for f in "${roots[@]}"; do
  (cd "$ROOT" && "$DUMPER" --dump-root "$f.cl" "build-tests/pow2-type4-$f.cl") >/dev/null
done

defs=(
  -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64
  -DEXP=175000039u -DWIDTH=512u -DSMALL_HEIGHT=512u -DMIDDLE=8u
  -DCARRY_LEN=8u -DNW=8u -DNH=8u
  -DFFT_VARIANT=202u -DMAXBPW=4729u
  '-DTAILT=U2(1.0f,0.0f)'
  '-DTAILTGF31=U2(269176336u,500380354u)'
  '-DTAILTGF61=U2(807738046998073027ul,30095103403839256ul)'
  -DFFT_TYPE=4 -DWordSize=8u
  -DDISTGF31=1048576u -DDISTWTRIGGF31=131072u -DDISTMTRIGGF31=2048u -DDISTHTRIGGF31=131072u
  -DDISTGF61=2097152ul -DDISTWTRIGGF61=262144ul -DDISTMTRIGGF61=4096ul -DDISTHTRIGGF61=262144ul
  -DFRAC_BPW_HI=375134435u -DFRAC_BPW_LO=2386092941u
  -DFFT_FP64=0 -DFFT_FP32=1 -DNTT_GF31=1 -DNTT_GF61=1
  -DTAIL_KERNELS=1 -DINPLACE=0 -DPAD=0
  -DTAIL_TRIGS32=2 -DTAIL_TRIGS31=0 -DTAIL_TRIGS61=0
  -DWEIGHT_STEP=0.0f -DIWEIGHT_STEP=0.0f -DTRIG_SCALE=1.0f
  '-DTRIG_SIN={1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}'
  '-DTRIG_COS={1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}'
  '-DTRIG_W={1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f}'
  '-DTRIG_H={1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f}'
  '-DTRIG_M={1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f}'
  -DNVIDIAGPU=1 -DCC=806u -DNO_ASM=1
)
for f in "${roots[@]}"; do
  "$CLANG_BIN" -x cl -cl-std=CL1.2 -fsyntax-only \
    "$ROOT/build-tests/pow2-type4-$f.cl" "${defs[@]}"
done
echo 'Aevum power-of-two FFT323161 4:512:8:512:202 OpenCL C 1.2 syntax passed'
