#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLANG_BIN="${CLANG:-$(command -v clang || true)}"
if [[ -z "$CLANG_BIN" ]]; then
  echo 'WARNING: clang not found; optional Native PFA OpenCL C syntax test skipped.' >&2
  exit 0
fi
mkdir -p "$ROOT/build-tests"
DUMPER="$ROOT/build-tests/aevum-opencl-monolithic-source-test"
# Rebuild every time: release archives may have been unpacked on a different CPU/OS.
rm -f "$DUMPER"
"${CXX:-c++}" -O2 -std=c++20 "$ROOT/tests/opencl_monolithic_source_test.cpp" \
  "$ROOT/src/OpenCLSourceBuilder.cpp" -o "$DUMPER"
roots=(fftp fftmiddlein tailsquare tailmul fftmiddleout fftw pfaunpack carry)
for f in "${roots[@]}"; do
  (cd "$ROOT" && "$DUMPER" --dump-root "$f.cl" "build-tests/native-pfa-$f.cl") >/dev/null
done
run_case() {
  local radix="$1" exp="$2" width="$3" height="$4" nw="$5" nh="$6" binary="$7" linv="$8" dist61="$9"
  local lr31 lr61 logical_step=0 binary_step=$((binary / nw)) k candidate
  for ((k=0; k<radix; ++k)); do
    candidate=$((binary_step + binary * k))
    if (( candidate % radix == 0 )); then logical_step=$candidate; break; fi
  done
  lr31="$(python3 -c "print(pow((${radix}*${binary}) % 31, -1, 31))")"
  lr61="$(python3 -c "print(pow((${radix}*${binary}) % 61, -1, 61))")"
  local defs=(
    -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64
    -DEXP="${exp}u" -DWIDTH="${width}u" -DSMALL_HEIGHT="${height}u" -DMIDDLE="${radix}u"
    -DCARRY_LEN=8u -DNW="${nw}u" -DNH="${nh}u"
    -DPFA_RADIX="${radix}u" -DPFA_BINARY_LENGTH="${binary}u" -DPFA_L_INV="${linv}u" -DPFA_LOGICAL_STEP="${logical_step}u"
    -DPFA_LOG2_ROOT_TWO31="${lr31}u" -DPFA_LOG2_ROOT_TWO61="${lr61}u"
    -DFFT_VARIANT=101u -DMAXBPW=4000u
    '-DTAILTGF31=U2(509684486u,293249438u)'
    '-DTAILTGF61=U2(249938719029223731ul,1245372627562045535ul)'
    -DFFT_TYPE=1 -DWordSize=8u
    -DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0
    -DDISTGF61="${dist61}ul" -DDISTWTRIGGF61=0 -DDISTMTRIGGF61=0 -DDISTHTRIGGF61=0
    -DFRAC_BPW_HI=852672511u -DFRAC_BPW_LO=4294967295u
    -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=1
    -DTAIL_KERNELS=1 -DINPLACE=0 -DPAD=0 -DTAIL_TRIGS31=0 -DTAIL_TRIGS61=0
  )
  for f in "${roots[@]}"; do
    "$CLANG_BIN" -x cl -cl-std=CL1.2 -fsyntax-only \
      "$ROOT/build-tests/native-pfa-$f.cl" "${defs[@]}"
  done
  echo "Native Aevum PFA radix-$radix OpenCL C 1.2 syntax passed"
}
run_case 3 100000019 256 256 4 4 131072 2 196608
run_case 9 175000001 512 512 8 8 524288 5 2359296

# Experimental FFT323161 (type 4) + PFA9 configuration used by the RTX 3080
# benchmark.  All three planes must parse together because carry.cl reconstructs
# the canonical integer from FP32, GF31 and GF61.
run_type4_pfa9() {
  local defs=(
    -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64
    -DEXP=175000039u -DWIDTH=512u -DSMALL_HEIGHT=512u -DMIDDLE=9u
    -DCARRY_LEN=8u -DNW=8u -DNH=8u
    -DPFA_RADIX=9u -DPFA_BINARY_LENGTH=524288u -DPFA_L_INV=5u -DPFA_LOGICAL_STEP=589824u
    -DPFA_LOG2_ROOT_TWO31=14u -DPFA_LOG2_ROOT_TWO61=30u
    -DFFT_VARIANT=202u -DMAXBPW=3895u
    '-DTAILT=U2(1.0f,0.0f)'
    '-DTAILTGF31=U2(269176336u,500380354u)'
    '-DTAILTGF61=U2(807738046998073027ul,30095103403839256ul)'
    -DFFT_TYPE=4 -DWordSize=8u
    -DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0
    -DDISTGF61=1179648ul -DDISTWTRIGGF61=147776ul -DDISTMTRIGGF61=2560ul -DDISTHTRIGGF61=147776ul
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
      "$ROOT/build-tests/native-pfa-$f.cl" "${defs[@]}"
  done
  echo 'Experimental Aevum FFT323161 PFA9 OpenCL C 1.2 syntax passed'
}
run_type4_pfa9
