#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build-tests/fftp-opencl12.cl}"
TEST="$ROOT/build-tests/aevum-opencl-monolithic-source-test"

(cd "$ROOT" && "$TEST" --dump-fftp "$OUT")

CLANG_BIN="${CLANG:-}"
if [[ -z "$CLANG_BIN" ]]; then
  CLANG_BIN="$(command -v clang || true)"
fi
if [[ -z "$CLANG_BIN" ]]; then
  echo "OpenCL 1.2 syntax test skipped: clang not found"
  exit 0
fi

grep -q 'fftP31Apple' "$OUT"
grep -q 'fftP61WeightScalarApple' "$OUT"
grep -q 'fftP61WidthRadixApple' "$OUT"
for k in fftP61TwiddleShuffle1Apple fftP61TwiddleShuffle4Apple fftP61TwiddleShuffle8Apple fftP61TwiddleShuffle16Apple fftP61TwiddleShuffle64Apple fftP61WidthFinalApple; do
  grep -q "$k" "$OUT"
done

"$CLANG_BIN" -x cl -cl-std=CL1.2 -fsyntax-only "$OUT" \
  -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64 \
  -DAEVUM_APPLE_SPLIT_FFTP=1 \
  -DEXP=1362763u \
  -DWIDTH=256u \
  -DSMALL_HEIGHT=256u \
  -DMIDDLE=2u \
  -DCARRY_LEN=8u \
  -DNW=4u \
  -DNH=4u \
  -DFFT_VARIANT=101u \
  -DMAXBPW=4054u \
  '-DTAILTGF31=U2(509684486u,293249438u)' \
  '-DTAILTGF61=U2(249938719029223731ul,1245372627562045535ul)' \
  -DFFT_TYPE=1 \
  -DWordSize=8u \
  -DDISTGF31=0 \
  -DDISTWTRIGGF31=0 \
  -DDISTMTRIGGF31=0 \
  -DDISTHTRIGGF31=0 \
  -DDISTGF61=65536ul \
  -DDISTWTRIGGF61=16576ul \
  -DDISTMTRIGGF61=384ul \
  -DDISTHTRIGGF61=16576ul \
  -DFRAC_BPW_HI=852672511u \
  -DFRAC_BPW_LO=4294967295u \
  -DFFT_FP64=0 \
  -DFFT_FP32=0 \
  -DNTT_GF31=1 \
  -DNTT_GF61=1

echo "Aevum global GF61 width stages OpenCL C 1.2 syntax test passed"
