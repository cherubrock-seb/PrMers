# PrMers OpenCL PRP

Experimental OpenCL code for Mersenne PRP tests and benchmarks.

The current tuned path is the mixed CRT/PFA half-real transform:

```text
GF(M61^2) x GF(M31^2)
N = odd * 2^m
pack digits + odd DFT -> half-real 2^m row transforms -> odd inverse DFT -> CRT/Garner -> carry
```

For the current GPU code, `odd=9` is the main target. The odd part is handled as a small real-scalar transform, currently using the faster `3 x 3` DFT9 decomposition. The power-of-two axis keeps the half-real GF(p^2) layout. This follows the CPU prototype idea in `docs/mersenne2_mixed_crt_2d_half_fast`: split the transform as `odd x 2^m`, keep the half-real transform on the `2^m` rows, then reconstruct through CRT/Garner.

## Build

```bash
g++ -O3 -DNDEBUG -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

The program now adds these OpenCL build options by default:

```text
-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math
```

They can be disabled or extended:

```bash
PRMERS_OCL_FAST_BUILD_OPTS=0 ./prmers_opencl_prp ...
PRMERS_OCL_FLAGS="-cl-opt-disable" ./prmers_opencl_prp ...
```

`PRMERS_OCL_FLAGS` is appended after the default build options.

## Quick validation

The fast half-real convention is `48` by default, so `--crt-halfreal-flags 48` is no longer needed in normal commands.

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
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

Expected validation line:

```text
CRT halfreal validator: OK
```

## Benchmark

Use `--quiet` without `--profile-kernels` for throughput. This avoids profiling queues and avoids host synchronization in the hot loop.

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 30000 \
  --quiet
```

With profiling:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-fuse-both off \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

The progress label for this path is now:

```text
[mixed CRT/PFA half-real]
```

## Main modes

```text
--modulus gf61       Use only GF(M61^2).
--modulus gf31       Use only GF(M31^2), only when the exponent fits the safety limits.
--modulus crt        Use GF(M61^2) x GF(M31^2), then reconstruct with CRT/Garner.
--modulus best       Let the planner choose the available mode.
```

For large exponents, the intended fast path is usually:

```text
--modulus crt --crt-odd-radix 9 --crt-center-mode halfreal
```

## Mixed CRT/PFA options

```text
--crt-odd-radix auto|1|3|9
    Odd axis. 9 is the tuned path. 1 disables the mixed odd split.

--crt-center-mode normal|halfreal
    Half-real is the current mixed odd fast path.

--crt-halfreal-no-autoprobe
    Skip convention probing. Use this with the fixed fast convention.

--crt-halfreal-flags N
    Half-real convention flags. Default is 48. It is kept for debugging and A/B tests.

--crt-mixed-gpu-reference
    Validate the optimized mixed path against the generic mixed GPU reference.

--crt-mixed-row-core auto|lds|lds512|lds1024|generic
    Row-transform implementation. `lds` / `lds512` is the normal fast path.

--crt-mixed-row-stage 512
    Intermediate LDS stage size. 512 is the tuned setting here.

--crt-mixed-row-center 512
    LDS center square size. 512 is the tuned setting here.

--crt-mixed-row-fuse-both off|auto|center|stage|all|force
    Optional GF61/GF31 row-fusion experiments. Current default remains off.
```

## CRT/NTT options

```text
--crt-defused-ntt / --crt-no-defused-ntt
    Defused NTT kernels. This is the fast default.

--crt-edge-radix 2|4|8|16
    Edge radix for weighted/unweighted stages. Default is 4.

--crt-edge-mode auto|legacy|generic
    Edge kernel family.

--crt-local-square 512
    Local center square size.

--crt-lds-stage 512
    Intermediate LDS stage size.

--crt-async-queues / --crt-two-queues
    Two-queue CRT scheduling. This remains the normal fast mode.

--crt-single-queue / --crt-shared-queue
    Single/shared queue tests. Useful for debugging, usually slower here.
```

## Fast-path defaults

The odd-9 mixed path defaults to the current best settings from the 142606357 RTX 3080 tests:

```text
PRMERS_CRT_MIXED_PACK_TILE14_LMAT_61X31=1
PRMERS_CRT_MIXED_PACK_TILE28_61X31=1
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30=1
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30_SMAT=1
PRMERS_CRT_MIXED_PREPACK_NEXT=0
PRMERS_CRT_MIXED_CARRY_PACK_NEXT_LDS=0
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0
```

Notes:

```text
DFT9 is implemented as two DFT3 stages in the compact odd-pack path.
GF31 single-LDS center is off by default; the normal GF31 center was faster in the tested path.
Next-iteration prepack/carry-pack is still available, but it is off by default because the long run was better without it.
```

To test next-iteration packing:

```bash
PRMERS_CRT_MIXED_PREPACK_NEXT=1 \
PRMERS_CRT_MIXED_CARRY_PACK_NEXT_LDS=1 \
./prmers_opencl_prp 142606357 --modulus crt --crt-odd-radix 9 --crt-center-mode halfreal ...
```

## Synchronization and profiling

The default benchmark path avoids `clFlush` and `clFinish` inside the iteration loop.

Default synchronization switches:

```text
PRMERS_CRT_PERIODIC_FLUSH=0
PRMERS_CRT_PROGRESS_FINISH=0
PRMERS_CRT_ALLOW_HOST_FLUSH=0
PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH=0
```

A final `clFinish` is still used at the end of a run to measure completed work and read the result.

`--profile-kernels` enables profiling queues and host synchronization for timing reports. Use it only for kernel breakdowns, not for final speed numbers.

## Current tuned kernel sequence for odd=9

Typical hot-loop sequence:

```text
crt_mixed_pack_weight_odd_fwd_tile28_shift_lmat_61x31
crt_mixed_lds512_forward_1lds_61
crt_mixed_lds512_forward_31
crt_mixed_lds512_center_1lds_rega_twinline_f48_61
crt_mixed_lds512_center_31
crt_mixed_lds512_inverse_1lds_61
crt_mixed_lds512_inverse_31
crt_mixed_odd9_precrt_garner9seg30_pair_smat
crt_carry_pass_oneout
crt_cleanup_serial_oneout
```

The largest remaining costs are the GF61 center and row stages, plus the fused preCRT/Garner tail. The GF31 pair2/stage2 experiments are not enabled by default because they validated worse or ran slower on the tested configuration.
