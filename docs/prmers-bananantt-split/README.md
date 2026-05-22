# PrMers BananaNTT odd9 CRT/PFA branch

GPU test branch for Mersenne PRP squaring with OpenCL.

The main path in this archive is the mixed CRT/PFA odd-radix path:

```text
N = odd * 2^m
```

The odd axis is handled with CRT/PFA indexing.  The power-of-two axis uses the half-real packing used by the current row NTT code.

This branch is mainly for correctness checks and performance experiments around:

```text
GF((2^61 - 1)^2) x GF((2^31 - 1)^2)
odd radix 9
half-real row NTT
GPU Garner/carry
```

It is not a packaged end-user PRP client.

## Build

```bash
g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

`-lgmp` is needed for the CPU/reference validation code.

## Fast odd9 run

Use this for timing.  Do not add `--profile-kernels` for a final speed number.

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
  --iters 10000 \
  --quiet
```

Expected fast-path log lines:

```text
queues=async-event
profile-queue=off
host-sync=final-only
pack-tile14-lmat-61x31=on
fuse-pack-both=on
center-single-lds=61:on,31:off
stage-single-lds=61:on,31:off
```

`host-sync=final-only` means the hot loop does not use host-side `clFinish` for progress.  A final synchronization is still needed to measure completed GPU work.

## Validation

This compares the selected LDS row path against the generic mixed GPU reference path.  Profiling is enabled here only to show which kernels ran.

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

Expected validation line:

```text
CRT halfreal validator: OK
```

Expected fused pack kernel:

```text
crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61x31
```

The old separate pack kernels should not appear in the normal fast path:

```text
crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61
crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_31
```

## Profiling

Use this only to inspect bottlenecks.  It creates profiled OpenCL queues and reports kernel timings, so it is slower than the real benchmark mode.

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
  --iters 5000 \
  --profile-kernels
```

With profiling enabled the log should show:

```text
profile-queue=on
host-sync=profile-reports
```

Do not compare `--profile-kernels` speed with `--quiet` speed directly.

## Main modes

### CRT normal power-of-two path

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --device 1 \
  --iters 3000 \
  --profile-kernels
```

### CRT mixed odd radix 9

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

### Single field tests

```bash
./prmers_opencl_prp 216091 --modulus gf61 --device 1 --iters 1000 --profile-kernels
./prmers_opencl_prp 216091 --modulus gf31 --device 1 --iters 1000 --profile-kernels
```

## Useful options

| option | purpose |
|---|---|
| `--modulus gf61` | use only GF((2^61 - 1)^2) |
| `--modulus gf31` | use only GF((2^31 - 1)^2) |
| `--modulus crt` | use the CRT path GF61 x GF31 |
| `--crt-odd-radix off` | normal power-of-two CRT transform |
| `--crt-odd-radix 3` | mixed CRT/PFA radix 3 path |
| `--crt-odd-radix 9` | mixed CRT/PFA radix 9 path |
| `--crt-odd-radix auto` | planner choice if supported by the branch |
| `--crt-center-mode halfreal` | use half-real center mode |
| `--crt-halfreal-flags 48` | current fast half-real flag set |
| `--crt-mixed-row-core lds` | force the LDS mixed row path |
| `--crt-mixed-row-stage 512` | LDS stage before the center |
| `--crt-mixed-row-center 512` | LDS center size |
| `--crt-mixed-row-fuse-both off` | keep GF61/GF31 row stages separate |
| `--crt-single-queue` | diagnostic mode; usually slower here |
| `--profile-kernels` | enable OpenCL queue profiling and print kernel times |
| `--quiet` | suppress progress output during timing |

## Runtime switches

The fast defaults can still be overridden from the environment.

| variable | default | purpose |
|---|---:|---|
| `PRMERS_CRT_MIXED_FUSE_PACK_BOTH` | `1` | fused GF61/GF31 pack + oddDFT kernel |
| `PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61` | `1` | GF61 optimized single-LDS center path |
| `PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61` | `1` | GF61 optimized single-LDS stage path |
| `PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31` | `0` | GF31 single-LDS test path; slower in current benches |
| `PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31` | `0` | GF31 single-LDS stage test path |
| `PRMERS_CRT_MIXED_CENTER_REGA_31` | `1` | GF31 center regA test path |
| `PRMERS_CRT_MIXED_CENTER_TWINLINE_31` | `1` | GF31 center twinline test path |
| `PRMERS_GF31_4MUL` | `0` | compare old 4-mul GF31 complex multiply against the default 3-mul form |
| `PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER` | `0` | experimental fused preCRT + Garner tail |
| `PRMERS_CRT_PROGRESS_FINISH` | `0` | force completed progress timing with `clFinish`; slower |
| `PRMERS_CRT_PERIODIC_FLUSH` | `0` | driver debugging only |
| `PRMERS_CRT_ALLOW_HOST_FLUSH` | `0` | allow legacy host flush points |
| `PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH` | `0` | allow legacy mixed-path host flush points |

Keep all flush/finish switches at `0` for performance runs.

## Experimental fused preCRT + Garner tail

This archive includes a fused tail experiment.  It tries to replace:

```text
crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat
crt_garner_first_coeffhi_mask32_anybase_x2
```

with:

```text
crt_mixed_odd9_precrt_garner64_lmat
```

Enable it with:

```bash
PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER=1 \
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-flags 48 \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

It is off by default.  Use it only for A/B tests until the profile proves it is faster than the separate preCRT and Garner kernels.

## Notes on timing

For a real throughput number:

```text
no --profile-kernels
use --quiet
profile-queue=off
host-sync=final-only
```

For kernel diagnosis:

```text
use --profile-kernels
expect profile-queue=on
expect host-sync=profile-reports
```

The two modes are not equivalent.  Profile runs are useful for relative kernel cost, not for final speed.

## Files

```text
prmers_opencl_prp.cpp      host code
prmers_opencl_prp.cl       OpenCL kernels
README.md                  this file
test_mixed_row_lds_matrix.sh optional path test script
V*.md                      short notes for recent experiments
```
