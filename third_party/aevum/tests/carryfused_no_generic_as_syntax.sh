#!/usr/bin/env bash
# carryFused's ready-flag atomics must compile on an OpenCL C 2.0/3.0 device without the generic address space
# (e.g. PoCL's CPU device on aarch64, or an OpenCL 3.0 driver without __opencl_c_generic_address_space).
# A cast of a global u32* to a plain atomic_uint* needs the generic address space; the cast must name global.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLANG_BIN="${CLANG:-$(command -v clang || true)}"
if [[ -z "$CLANG_BIN" ]]; then
  echo 'WARNING: clang not found; carryFused no-generic-address-space syntax test skipped.' >&2
  exit 0
fi
DUMPER="$ROOT/build-tests/aevum-opencl-monolithic-source-test"
if [[ ! -x "$DUMPER" ]]; then
  mkdir -p "$ROOT/build-tests"
  "${CXX:-c++}" -O2 -std=c++20 "$ROOT/tests/opencl_monolithic_source_test.cpp" \
    "$ROOT/src/OpenCLSourceBuilder.cpp" -o "$DUMPER"
fi
SRC="$ROOT/build-tests/carryfused-no-generic-as.cl"
(cd "$ROOT" && "$DUMPER" --dump-root carryfused.cl "$SRC") >/dev/null

defs=(
  -Xclang -cl-ext=+cl_khr_fp64,+__opencl_c_fp64,-__opencl_c_generic_address_space,-__opencl_c_device_enqueue,-__opencl_c_pipes
  # Older clang (e.g. 18) also drops these feature macros with the generic address space, and its headers then hide
  # atomic_store/atomic_load. Keep them so the only missing feature is the generic address space.
  -D__opencl_c_atomic_order_seq_cst=1 -D__opencl_c_atomic_scope_device=1
  -DEXP=1362763u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DCARRY_LEN=8u -DNW=4u -DNH=4u
  -DFFT_VARIANT=101u -DMAXBPW=4054u
  '-DTAILTGF31=U2(509684486u,293249438u)'
  '-DTAILTGF61=U2(249938719029223731ul,1245372627562045535ul)'
  -DFFT_TYPE=1 -DWordSize=8u
  -DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0
  -DDISTGF61=65536ul -DDISTWTRIGGF61=16576ul -DDISTMTRIGGF61=384ul -DDISTHTRIGGF61=16576ul
  -DFRAC_BPW_HI=852672511u -DFRAC_BPW_LO=4294967295u
  -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=1
)

# INPLACE=1 uses the lines-read flag; OLD_FENCE=0 (an AMD-only setting) uses the per-wavefront hand-off.  Together
# they reach every AEVUM_READY_STORE / AEVUM_READY_LOAD expansion.
for extra in "-DINPLACE=0" "-DINPLACE=1" "-DINPLACE=1 -DOLD_FENCE=0 -DAMDGPU=1 -DWAVEFRONT=64"; do
  # shellcheck disable=SC2086
  if ! "$CLANG_BIN" -x cl -cl-std=CL3.0 -fsyntax-only "$SRC" "${defs[@]}" $extra; then
    echo "FAIL: carryfused.cl does not compile without the generic address space ($extra)" >&2
    exit 1
  fi
done
echo "carryFused no-generic-address-space syntax test passed"
