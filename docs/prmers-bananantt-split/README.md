# PrMers OpenCL odd9 CRT branch

Experimental OpenCL PRP code for Mersenne numbers using a mixed CRT/PFA odd-radix path.
This release is based on the v47 baseline and keeps the fast, validated odd9 path:

- odd radix 9 with row half-real NTTs
- fused pack + odd DFT for GF(M61^2) and GF(M31^2)
- fused odd inverse + preCRT + first Garner for the tail
- two OpenCL queues by default
- no `clFlush` or `clFinish` in the hot loop
- profiling queues only when `--profile-kernels` is requested

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

## Useful environment switches

```text
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER=0|1
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30=0|1
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30_SMAT=0|1
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0|1
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0|1
PRMERS_CRT_MIXED_FUSE_PACK_BOTH=0|1
```

The default path keeps GF31 center/stage single-LDS disabled.  The v51 change is limited to the fused pack+oddDFT kernel: the odd DFT matrices are cached as scalar values in LDS instead of full GF/GF31 pairs.

## V57 experimental prepack-next

`PRMERS_CRT_MIXED_PREPACK_NEXT=1` moves the pack of iteration `n+1` to the end of
iteration `n`, immediately after carry. The next iteration skips the usual head
pack and starts from the already packed `a61/a31` buffers.

This is a safe scheduling experiment based on the existing tile28 61x31 pack
kernel. It is not yet the final carry+pack fused kernel, so speedup may be small
or neutral. Use it only for A/B testing.

## V58 experimental carry-pack-next LDS

This build adds an optional fused carry + next-pack path. It is off unless both variables are set:

```bash
PRMERS_CRT_MIXED_PREPACK_NEXT=1
PRMERS_CRT_MIXED_CARRY_PACK_NEXT_LDS=1
```

The path tries to avoid the separate next-iteration pack kernel by building the next GF61/GF31 packed rows inside the carry pass using LDS. Validate it before long runs. See `V58_CARRY_PACK_NEXT_LDS.md`.

