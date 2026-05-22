# PrMers OpenCL odd9 CRT test

Experimental OpenCL implementation for Mersenne PRP/LL style squaring with a mixed CRT transform.

The current fast path targets the odd-radix layout

```text
N = 9 * 2^m
pack + odd DFT -> row half-real NTT -> odd inverse + CRT/Garner -> carry
```

The code uses two CRT fields, GF(M61^2) and GF(M31^2), then reconstructs digits with a GPU Garner/carry path.

## Build

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

## Fast benchmark

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 30000 \
  --quiet
```

For a clean speed measurement, do not use `--profile-kernels`. In normal bench mode the queues are not profiling queues and the host synchronization mode is final-only.

Expected fast-path markers:

```text
queues=async-event
profile-queue=off
host-sync=final-only
pack-tile14-lmat-61x31=on
precrt-garner64=on
precrt-garner-pair30=on
precrt-garner-pair30-smat=on
center-single-lds=61:on,31:off
stage-single-lds=61:on,31:off
```

## Validation

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-mixed-gpu-reference \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

Validation uses the generic GPU reference path, so it is not a benchmark.

## Profiling

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 3000 \
  --profile-kernels
```

Profiling enables OpenCL event timing and profile reports. The measured speed is useful for kernel comparison, not as the final throughput number.

## Main defaults

The default odd9 CRT mode now uses:

```text
fused pack GF61+GF31
fused odd9 preCRT+Garner pair30 scalar-matrix tail
two async queues
no hot-loop clFlush
no progress clFinish
profiling queue only with --profile-kernels
GF61 center/stage single-LDS enabled
GF31 center/stage single-LDS disabled
```

## Environment switches

Useful switches:

```bash
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER=0       # use split preCRT + Garner
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30=0
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30_SMAT=0
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=1     # force GF31 center one-LDS test
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=1      # force GF31 stage one-LDS test
PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH=1         # debug only
PRMERS_CRT_PROGRESS_FINISH=1                # debug/progress sync only
```

`PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31` is intentionally off by default. Previous tests showed it was not a good default on RTX 3080 for this odd9 case.

## Files

```text
prmers_opencl_prp.cpp   host code and OpenCL scheduling
prmers_opencl_prp.cl    OpenCL kernels
README.md              this file
test_mixed_row_lds_matrix.sh  helper test matrix
```

## v47 note

The default GF31 center is the normal LDS512 center. The one-LDS/regA/twinline GF31 center is not selected unless `PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=1` is set. This keeps the measured best default: GF61 single-LDS on, GF31 single-LDS off.

