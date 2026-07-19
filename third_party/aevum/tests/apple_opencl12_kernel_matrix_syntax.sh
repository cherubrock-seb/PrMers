#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST="$ROOT/build-tests/aevum-opencl-monolithic-source-test"
CLANG_BIN="${CLANG:-$(command -v clang || true)}"
if [[ -z "$CLANG_BIN" ]]; then
  echo "Apple OpenCL 1.2 kernel matrix syntax test skipped: clang not found"
  exit 0
fi

COMMON=(
  -x cl -cl-std=CL1.2 -fsyntax-only
  -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64
  -DAEVUM_APPLE_OPENCL12=1
  -DPFA_RADIX=0u
  -DEXP=1362763u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u
  -DCARRY_LEN=8u -DNW=4u -DNH=4u -DFFT_VARIANT=101u -DMAXBPW=4054u
  '-DTAILTGF31=U2(509684486u,293249438u)'
  '-DTAILTGF61=U2(249938719029223731ul,1245372627562045535ul)'
  -DFFT_TYPE=1 -DWordSize=8u
  -DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0
  -DDISTGF61=65536ul -DDISTWTRIGGF61=16576ul -DDISTMTRIGGF61=384ul -DDISTHTRIGGF61=16576ul
  -DFRAC_BPW_HI=852672511u -DFRAC_BPW_LO=4294967295u
  -DIN_WG=64 -DOUT_WG=64 -DIN_SIZEX=16 -DOUT_SIZEX=16
  -DMIDDLE_IN_LDS_TRANSPOSE=1 -DMIDDLE_OUT_LDS_TRANSPOSE=1
  -DNO_ASM=1 -DPAD=0
  -DTAIL_KERNELS=3
)

mkdir -p "$ROOT/build-tests/apple-cl12-matrix"
compile_case() {
  local root_file="$1" mode="$2"
  local out="$ROOT/build-tests/apple-cl12-matrix/${root_file%.cl}-${mode}.cl"
  (cd "$ROOT" && "$TEST" --dump-root "$root_file" "$out")
  local defs=(-DFFT_FP64=0 -DFFT_FP32=0)
  case "$mode" in
    gf31) defs+=(-DNTT_GF31=1 -DNTT_GF61=0) ;;
    gf61) defs+=(-DNTT_GF31=0 -DNTT_GF61=1) ;;
    all)  defs+=(-DNTT_GF31=1 -DNTT_GF61=1) ;;
    *) echo "bad mode $mode" >&2; exit 2 ;;
  esac
  "$CLANG_BIN" "${COMMON[@]}" "${defs[@]}" "$out"
  echo "syntax pass: $root_file $mode"
}

# Every root source instantiated by Gpu.cpp is parsed in the Apple OpenCL C 1.2
# configuration. GF-specific roots are checked separately where this catches
# address-space/type specialization errors that an all-in-one parse can hide.
compile_case fftmiddlein.cl gf31
compile_case fftmiddlein.cl gf61
compile_case ffthin.cl gf31
compile_case ffthin.cl gf61
compile_case tailsquare.cl gf31
compile_case tailsquare.cl gf61
compile_case tailmul.cl gf31
compile_case tailmul.cl gf61
compile_case fftmiddleout.cl gf31
compile_case fftmiddleout.cl gf61
compile_case fftw.cl gf31
compile_case fftw.cl gf61
compile_case carry.cl all
compile_case carryfused.cl all
compile_case carryb.cl all
compile_case transpose.cl all
compile_case etc.cl all
compile_case selftest.cl all

# Large Aevum plan used by p=136279841.  MIDDLE=8 instantiates GF31
# fft8Core inside fftmiddlein.cl; Apple cl2Metal previously lost the imported
# fft4Core overload at this exact configuration.
BIG=(
  -x cl -cl-std=CL1.2 -fsyntax-only
  -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64
  -DAEVUM_APPLE_OPENCL12=1
  -DPFA_RADIX=0u
  -DEXP=136279841u -DWIDTH=1024u -DSMALL_HEIGHT=256u -DMIDDLE=8u
  -DCARRY_LEN=8u -DNW=4u -DNH=4u -DCARRY64=1 -DFFT_VARIANT=101u -DMAXBPW=3946u
  '-DTAILTGF31=U2(509684486u,293249438u)'
  '-DTAILTGF61=U2(249938719029223731ul,1245372627562045535ul)'
  -DFFT_TYPE=1 -DWordSize=8u
  -DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0
  -DDISTGF61=1048576ul -DDISTWTRIGGF61=512ul -DDISTMTRIGGF61=1536ul -DDISTHTRIGGF61=262336ul
  -DFRAC_BPW_HI=2111603711u -DFRAC_BPW_LO=4294967295u
  -DIN_WG=64 -DOUT_WG=64 -DIN_SIZEX=16 -DOUT_SIZEX=16
  -DMIDDLE_IN_LDS_TRANSPOSE=1 -DMIDDLE_OUT_LDS_TRANSPOSE=1
  -DNO_ASM=1 -DPAD=0 -DTAIL_KERNELS=3
  -DFFT_FP64=0 -DFFT_FP32=0
)
compile_big_case() {
  local root_file="$1" mode="$2"
  local out="$ROOT/build-tests/apple-cl12-matrix/${root_file%.cl}-${mode}-middle8.cl"
  (cd "$ROOT" && "$TEST" --dump-root "$root_file" "$out")
  local defs=()
  case "$mode" in
    gf31) defs=(-DNTT_GF31=1 -DNTT_GF61=0) ;;
    gf61) defs=(-DNTT_GF31=0 -DNTT_GF61=1) ;;
    all)  defs=(-DNTT_GF31=1 -DNTT_GF61=1) ;;
    *) echo "bad mode $mode" >&2; exit 2 ;;
  esac
  "$CLANG_BIN" "${BIG[@]}" "${defs[@]}" "$out"
  echo "syntax pass: $root_file $mode MIDDLE=8 WIDTH=1024 CARRY64"
}

# Parse every arithmetic root reached by large PRP/P-1, not only the first
# kernel that failed in the observed log.
compile_big_case fftmiddlein.cl gf31
compile_big_case fftmiddlein.cl gf61
compile_big_case ffthin.cl gf31
compile_big_case ffthin.cl gf61
compile_big_case tailsquare.cl gf31
compile_big_case tailsquare.cl gf61
compile_big_case tailmul.cl gf31
compile_big_case tailmul.cl gf61
compile_big_case fftmiddleout.cl gf31
compile_big_case fftmiddleout.cl gf61
compile_big_case fftw.cl gf31
compile_big_case fftw.cl gf61
compile_big_case carry.cl all
compile_big_case carryfused.cl all
compile_big_case carryb.cl all

echo "Aevum Apple OpenCL C 1.2 production kernel matrix syntax test passed"
