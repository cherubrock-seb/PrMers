#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../third_party/aevum" && pwd)"
CLANG_BIN="${CLANG:-$(command -v clang || true)}"
if [[ -z "$CLANG_BIN" ]]; then
  echo 'No clang: new fused PRP kernel syntax unverified. Install clang to screen fusion.' >&2
  exit 1
fi
mkdir -p "$ROOT/build-tests"
DUMPER="$ROOT/build-tests/aevum-opencl-monolithic-source-test"
"${CXX:-c++}" -O2 -std=c++20 "$ROOT/tests/opencl_monolithic_source_test.cpp" "$ROOT/src/OpenCLSourceBuilder.cpp" -o "$DUMPER"
(cd "$ROOT" && "$DUMPER" --dump-root tailsquare.cl build-tests/prp-middle1.cl) >/dev/null
# Constants below are syntax fixtures. GPU kernels use the engine's actual generated roots.
defs=(
 -Dcl_khr_fp64=1 -Xclang -cl-ext=+cl_khr_fp64 -DAEVUM_PRP_MIDDLE1=1
 -DMIDDLE=1u -DCARRY_LEN=8u -DWordSize=8u -DFFT_FP64=0 -DNTT_GF31=1 -DNTT_GF61=1
 -DTAIL_KERNELS=2 -DINPLACE=0 -DNO_ASM=1 -DAEVUM_GF61_LIMB32=0
 -DTAIL_TRIGS32=2 -DTAIL_TRIGS31=0 -DTAIL_TRIGS61=0
 -DMAXBPW=4729u -DDISTGF31=1048576u -DDISTWTRIGGF31=131072u -DDISTMTRIGGF31=2048u -DDISTHTRIGGF31=131072u
 -DDISTGF61=2097152ul -DDISTWTRIGGF61=262144ul -DDISTMTRIGGF61=4096ul -DDISTHTRIGGF61=262144ul
 -DFRAC_BPW_HI=375134435u -DFRAC_BPW_LO=2386092941u
 '-DTAILT=U2(1.0f,0.0f)' '-DTAILTGF31=U2(269176336u,500380354u)'
 '-DTAILTGF61=U2(807738046998073027ul,30095103403839256ul)'
 -DWEIGHT_STEP=0.0f -DIWEIGHT_STEP=0.0f -DTRIG_SCALE=1.0f
 '-DTRIG_SIN={1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}'
 '-DTRIG_COS={1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}'
 '-DTRIG_W={1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f}'
 '-DTRIG_H={1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f}'
 '-DTRIG_M={1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f}'
)
for arch in NVIDIA AMD; do
  if [[ "$arch" == NVIDIA ]]; then gpu=(-DNVIDIAGPU=1 -DAMDGPU=0 -DPAD=0 -DCC=806u); else gpu=(-DNVIDIAGPU=0 -DAMDGPU=1 -DPAD=256); fi
  for type in 1 4; do
    fp=0; [[ "$type" == 4 ]] && fp=1
    for shape in '256 256 4 4 101' '512 512 8 8 101' '512 512 8 8 202' '1024 256 4 4 101' '4096 512 8 8 202'; do
      read -r w h nw nh variant <<< "$shape"
      "$CLANG_BIN" -x cl -cl-std=CL2.0 -fsyntax-only "$ROOT/build-tests/prp-middle1.cl" \
        "${defs[@]}" "${gpu[@]}" -DFFT_TYPE="$type" -DFFT_FP32="$fp" -DEXP=21000029u \
        -DWIDTH="$w" -DSMALL_HEIGHT="$h" -DNW="$nw" -DNH="$nh" -DFFT_VARIANT="$variant"
    done
  done
done
echo 'PRP middle-one fusion: AMD/NVIDIA type1/type4 syntax matrix passed.'
