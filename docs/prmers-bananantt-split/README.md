# BananaNTT 0.75.00-alpha

Experimental OpenCL PRP code for Mersenne numbers.

The current tuned path is the mixed CRT/PFA half-real transform:

```text
GF(M61^2) x GF(M31^2)
N = odd * 2^m
pack digits + odd DFT -> half-real 2^m row transforms -> odd inverse DFT -> CRT/Garner -> carry
```

For the current GPU code, `odd=9` is the main tuned path. The odd part uses a compact `3 x 3` DFT9 decomposition. The power-of-two axis keeps the half-real GF(p^2) layout, then CRT/Garner reconstructs the integer digits.

BananaNTT is still experimental. Keep PrMers as the compatible/reference program for PrimeNet-style production submission.

## Build

```bash
cd ~/mgpu
rm -rf odd9_release_v75_bananantt_readme_stageauto
unzip odd9_release_v75_bananantt_readme_stageauto.zip
cd odd9_release_v75_bananantt_readme_stageauto

g++ -O3 -std=c++20 -march=native -pthread prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

Default OpenCL build options:

```text
-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math
```

Controls:

```text
PRMERS_OCL_FAST_BUILD_OPTS=0   disable the default fast OpenCL options
PRMERS_OCL_FLAGS="..."         append extra OpenCL build flags
```

## OpenCL binary cache

The first run compiles the OpenCL program and saves a binary in `.ocl_cache/`. Later runs with the same GPU, driver, kernel source, build options and BananaNTT version load the binary cache.

```bash
./prmers_opencl_prp 10000019 --device 0 --iters 1000 --no-resume
./prmers_opencl_prp 10000019 --device 0 --iters 1000 --no-resume
```

Expected second run:

```text
OpenCL build: loaded binary cache for GF(M61^2)
OpenCL build: loaded binary cache for GF(M31^2)
```

Controls:

```text
PRMERS_OCL_BINARY_CACHE=0       disable binary cache
PRMERS_OCL_CACHE_DIR=/path      choose cache directory
PRMERS_SHOW_OCL_BUILD=0         hide build/cache messages
PRMERS_OCL_BUILD_SPINNER=0      use simple build messages
```

Reset cache:

```bash
rm -rf .ocl_cache
```

## Default run

```bash
./prmers_opencl_prp 142606357 --device 1
```

Default behavior:

```text
modulus planner selects CRT when appropriate
odd-radix planner enables odd=9 only when the row plan is expected to be useful
half-real row core is used for the mixed odd path
Gerbicz-Li is enabled
backup/resume is enabled
JSON output is enabled
one JSON result is appended to ./results.txt
adaptive queue guard is enabled for Ctrl-C responsiveness
OpenCL binary cache is enabled
```

## Quick validation

The fast half-real convention is `48` by default, so `--crt-halfreal-flags 48` is not needed for normal commands.

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-mixed-gpu-reference \
  --device 1 \
  --iters 1 \
  --profile-kernels
```

Expected validation line:

```text
CRT halfreal validator: OK
```

## Benchmark

Use `--quiet` without `--profile-kernels` for throughput. Profiling queues and `--res64-every` force synchronization and should not be used for final speed numbers.

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --iters 30000 \
  --quiet \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --queue-guard 0
```

With profiling:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --iters 5000 \
  --profile-kernels \
  --no-resume
```

Progress label for the mixed path:

```text
[mixed CRT/PFA half-real]
```

## Main modes

```text
--modulus gf61       use only GF(M61^2)
--modulus gf31       use only GF(M31^2), when the exponent fits the safety limits
--modulus crt        use GF(M61^2) x GF(M31^2), then CRT/Garner
--modulus best       let the planner choose
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
    Skip convention probing.

--crt-halfreal-flags N
    Half-real convention flags. Default is 48.

--crt-mixed-gpu-reference
    Validate the optimized mixed path against the generic mixed GPU reference.

--crt-mixed-row-core auto|lds|lds512|lds1024|generic
    Row-transform implementation. `auto` is the normal default.

--crt-mixed-row-stage 8|16|32|64|128|256|512|1024
    Intermediate LDS row stage cap.

--crt-mixed-row-center 512|1024
    LDS center square size. 512 is the main tuned setting.

--crt-mixed-row-fuse-both off|auto|center|stage|all|force
    Optional GF61/GF31 row-fusion experiments.
```

## Odd-radix auto planner

The odd9 path is excellent when the row stage lands on the tuned 512-style path. Some medium sizes can instead fall into small row stages such as LDS32, which may be slower than the normal CRT path.

Version 0.75 avoids automatic odd9 selection when the predicted row reduction is too small. A case like `p=9437189` should therefore default away from the bad `row-stage-plan=fwd=32` unless you force odd9 manually.

Controls:

```text
PRMERS_CRT_MIXED_AUTO_MIN_STAGE_RADIX=64   default minimum auto row-stage factor
--crt-odd-radix 9                          force odd9 anyway
--crt-odd-radix 1                          force normal non-mixed CRT
```

Useful A/B commands:

```bash
./prmers_opencl_prp 9437189 --device 1 --profile-kernels --no-resume
./prmers_opencl_prp 9437189 --device 1 --crt-odd-radix 9 --profile-kernels --no-resume
./prmers_opencl_prp 9437189 --device 1 --crt-odd-radix 1 --profile-kernels --no-resume
```

If you want the old aggressive auto behavior:

```bash
PRMERS_CRT_MIXED_AUTO_MIN_STAGE_RADIX=1 ./prmers_opencl_prp 9437189 --device 1
```

## CRT/NTT options

```text
--crt-defused-ntt / --crt-no-defused-ntt
    Defused NTT kernels. Fast default.

--crt-edge-radix 2|4|8|16
    Edge radix for weighted/unweighted stages. Default is 4.

--crt-edge-mode auto|legacy|generic
    Edge kernel family.

--crt-local-square 512
    Local center square size.

--crt-lds-stage 512
    Intermediate LDS stage size.

--crt-async-queues / --crt-two-queues
    Two-queue CRT scheduling.

--crt-single-queue / --crt-shared-queue
    Single/shared queue tests. Usually slower here.
```

## Fast-path defaults

The odd9 mixed path defaults to the current best settings from the RTX 3080 tests:

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
GF31 single-LDS center is off by default because the normal GF31 center was faster in the tested path.
Next-iteration prepack/carry-pack remains available but is off by default.
```

## Gerbicz-Li

Gerbicz-Li is enabled by default. The default mode is quiet: setup line, full checks, failures, restore, backup and final status only.

```bash
./prmers_opencl_prp 216091 --device 1 --iters 50000 --no-resume
```

Fast injected-error test on M10000019:

```bash
./prmers_opencl_prp 10000019 --device 0 \
  --iters 8000 \
  --no-resume \
  --no-backup \
  --gerbicz-b 1024 \
  --gerbicz-checklevel 1 \
  --error-iter 2000 \
  --error-limb 0 \
  --error-delta 1
```

More visible diagnostics:

```bash
./prmers_opencl_prp 10000019 --device 0 \
  --iters 8000 \
  --no-resume \
  --no-backup \
  --gerbicz-b 1024 \
  --gerbicz-checklevel 1 \
  --error-iter 2000 \
  --error-limb 0 \
  --error-delta 1 \
  --gerbicz-progress
```

Controls:

```text
--gerbicz-seconds S            target full Li check spacing
--gerbicz-boundary-seconds S   target D update spacing when B is automatic
--gerbicz-b B                  force Li block size
--gerbicz-checklevel N         force full check every N boundaries
--gerbicz-verbose              print every boundary update
--gerbicz-progress             print full-check leg progress
--gerbicz-host                 use host GMP checker
--no-gerbicz                   disable Gerbicz-Li
```

## Backup and resume

Resume is automatic from `save/M<p>.bananantt.chk`.

```bash
./prmers_opencl_prp 142606357 --device 1 --backup-seconds 120
```

Disable resume for clean tests:

```bash
./prmers_opencl_prp 216091 --device 1 --no-resume
```

Disable backup and resume for pure benchmarks:

```bash
./prmers_opencl_prp 142606357 --device 1 --iters 30000 --quiet --no-backup --no-resume
```

## Queue guard and Ctrl-C

Default:

```text
--queue-guard auto --queue-guard-seconds 2
```

This limits Ctrl-C latency without forcing a fixed `clFinish` every 256 iterations. For pure benchmark mode:

```bash
./prmers_opencl_prp 142606357 --device 1 --iters 30000 --quiet --no-gerbicz --no-backup --no-resume --queue-guard 0
```

## Output files

Default files:

```text
./prmers_bananantt_M<p>.json
./results.txt
./save/M<p>.bananantt.chk
./<p>/proof/
```

Path controls:

```text
--output-dir DIR
--json-file FILE
--results-file FILE
--backup-dir DIR
--save-file FILE
--resume-file FILE
--proof-dir DIR
```

The JSON follows the PrMers-style result shape with `program.name = "banana"` and `program.version = "0.75.00-alpha"`.

## Current tuned kernel sequence for odd=9

Typical hot-loop sequence when odd9 uses the LDS512 row path:

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

For medium sizes, watch the profile. If you see `crt_mixed_lds32_forward_*` and `crt_mixed_lds32_inverse_*` dominating, compare against `--crt-odd-radix 1` or force a different row plan.
