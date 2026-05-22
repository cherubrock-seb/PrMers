
## v41 GF31 Karatsuba fast path

Default fast mode is still the v40 mode: fused `pack61x31`, two async queues, no hot-loop `clFlush`, no progress `clFinish`, and queue profiling only with `--profile-kernels`.

New in v41: GF31 complex multiplication uses a Karatsuba formula by default. This reduces a GF31 complex multiply from 4 scalar GF31 multiplications to 3 scalar GF31 multiplications plus cheap 32-bit add/sub operations. This targets the GF31 row path: forward LDS512, center LDS512, and inverse LDS512.

To compare against the old 4-mul GF31 formula, rebuild/run with:

```bash
PRMERS_GF31_4MUL=1 ./prmers_opencl_prp ...
```

# odd9 release v40 - mixed pack61x31 fast defaults, no hot-loop host sync

This release keeps the v37/v38/v39 mixed odd-radix method, but makes the fast path the default:

- `odd=9` mixed CRT/PFA row half-real path.
- `pack+oddDFT` fused for GF61 and GF31 in one kernel by default.
- GF61 center/stage single-LDS enabled by default.
- GF31 center/stage single-LDS disabled by default because it was slower in benchmarks.
- No `clFlush` in the hot CRT loop by default.
- No `clFinish` for progress printing by default.
- OpenCL queue profiling is enabled only when `--profile-kernels` is passed.
- Two async queues remain the default. `--crt-single-queue` is available for comparison, but it is slower on the RTX 3080 test.

The important log lines for a fast benchmark are:

```text
queues=async-event
profile-queue=off
host-sync=final-only
pack-tile14-lmat-61x31=on
fuse-pack-both=on
center-single-lds=61:on,31:off
stage-single-lds=61:on,31:off
```

`host-sync=final-only` means the benchmark loop does not force host/GPU synchronization. The final `clFinish` is still required to measure the completed GPU work.

## Build

```bash
cd ~/mgpu
rm -rf odd9_release_v40_mixed_pack61x31_default_fast_nosync
unzip odd9_release_v40_mixed_pack61x31_default_fast_nosync.zip
cd odd9_release_v40_mixed_pack61x31_default_fast_nosync
chmod +x test_mixed_row_lds_matrix.sh

g++ -O3 -std=c++20 -march=native prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

## Correctness validation

Validation uses `--profile-kernels` here intentionally, so it will create profiled queues and report kernel timings. Do not use this command for speed measurement.

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

Expected profile must show:

```text
crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61x31
```

and should not show separate pack kernels:

```text
crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61
crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_31
```

## Fast benchmark, no profiling, no hot-loop host sync

This is the command to measure speed. No environment variables are needed for the default fast mode.

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

Without `--quiet`, progress is printed from the host enqueue loop. It does not call `clFinish` unless profiling is active or `PRMERS_CRT_PROGRESS_FINISH=1` is explicitly set.

## Profile benchmark

This command is for locating bottlenecks. It is slower because profiled queues and profile report syncs are enabled.

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

## Single queue comparison

This is only for diagnosis. It was slower on the uploaded RTX 3080 log.

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
  --crt-single-queue \
  --device 1 \
  --iters 5000 \
  --quiet
```

## Runtime knobs

Fast defaults:

```bash
PRMERS_CRT_MIXED_FUSE_PACK_BOTH=1
PRMERS_CRT_MIXED_FINISH_AFTER_SQUARE=0
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=0
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=1
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=0
PRMERS_CRT_PROGRESS_FINISH=0
PRMERS_CRT_PERIODIC_FLUSH=0
PRMERS_CRT_ALLOW_HOST_FLUSH=0
PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH=0
```

You can still override them explicitly:

```bash
PRMERS_CRT_MIXED_FUSE_PACK_BOTH=0        # old separate GF61/GF31 pack kernels
PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH=1      # driver debugging only
PRMERS_CRT_PROGRESS_FINISH=1             # accurate completed progress, slower
PRMERS_CRT_PERIODIC_FLUSH=1              # driver debugging only
PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=1  # test only; slower in uploaded bench
PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=1   # test only; slower in uploaded bench
```

## What changed compared to v39

- `PRMERS_CRT_MIXED_FUSE_PACK_BOTH` now defaults to `1`.
- The preferred mixed path now automatically selects `crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61x31` when available.
- No extra host flush/finish was added. Hot-loop host sync remains disabled by default.
- README and test commands were cleaned so the default command is the fast command.

## Current interpretation of the benchmark

On the uploaded test, v39 with fused pack61x31 and no profiling reached about 438-441 it/s for 5000-10000 iterations, while the profile run is around 429 it/s and single queue is around 426-427 it/s. That supports the current default: two async queues, no profiled queues unless requested, and fused pack61x31 enabled.


## V42B compile fix

This archive fixes the v42 diagnostic-print compile error (`flags31` out of scope). Kernel changes are otherwise unchanged.
