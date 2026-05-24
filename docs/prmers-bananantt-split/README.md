# PrMers BananaNTT Split

GPU PRP prototype for Mersenne numbers using a mixed CRT/PFA odd-radix layout.

This branch is part of PrMers. It is not meant to replace the stable PrMers/PrimeNet-compatible path yet. The goal is to test a GPU implementation of the mixed-radix idea described in:

```text
docs/mersenne2_mixed_crt_2d_half_fast
docs/prmers-bananantt-split
```

Main repository:

```text
https://github.com/cherubrock-seb/PrMers
```

## Idea

The tuned path uses two NTT fields and CRT:

```text
GF(M61^2) x GF(M31^2)
M61 = 2^61 - 1
M31 = 2^31 - 1
```

Instead of always using a pure power-of-two transform, BananaNTT can use:

```text
N = odd * 2^m
```

The current GPU path targets:

```text
odd = 3
odd = 9
```

The hot path is usually:

```text
pack digits + odd DFT
-> half-real row transforms on the 2^m axis
-> odd inverse DFT + unweight
-> CRT/Garner
-> carry
```

For `odd=9`, the odd DFT is implemented as a compact `3 x 3` decomposition.

## About the half-real point

This is not Yves Gallot's newer Four-Step method.

Yves' Four-Step method keeps the odd radix in the center of the transform. That is mathematically cleaner and can preserve the half-real symmetry better across the rows.

BananaNTT currently uses a direct 2D CRT/PFA split:

```text
odd axis     : small DFT / inverse DFT
power2 axis  : half-real NTT rows
```

So the half-real symmetry is used on the `2^m` axis only. The code does not assume that odd rows are simple conjugate copies. For radix 9 this means the path may compute all odd rows instead of using a `9 -> 5` symmetry shortcut.

This is still useful because the total transform can be smaller in radix-9 windows. In some exponent ranges, this can already beat the pure power-of-two CRT path even before the Four-Step version is implemented.

## Status

Current status:

```text
experimental
OpenCL GPU path
CRT GF(M61^2) x GF(M31^2)
odd radix 3 and 9
half-real rows
Gerbicz-Li enabled by default
backup / resume
PrMers-style JSON result
results.txt append
intermediate res64 readback
OpenCL binary cache
startup autotune
```

Recent RTX 3080 tests were around:

```text
M142606357 odd9 path: about 600 it/s class, depending on options and checks
```

Use `--no-gerbicz --no-backup --no-resume --quiet` for pure speed tests.

## Build

```bash
g++ -O3 -DNDEBUG -std=c++20 -march=native -pthread prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

`-lgmp` is needed for CPU reference validation and host-side debug checks.

Default OpenCL build flags:

```text
-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math
```

Optional controls:

```bash
PRMERS_OCL_FAST_BUILD_OPTS=0 ./prmers_opencl_prp ...
PRMERS_OCL_FLAGS="-cl-opt-disable" ./prmers_opencl_prp ...
```

## Default run

```bash
./prmers_opencl_prp 142606357 --device 1
```

Default behavior:

```text
--modulus crt
--crt-odd-radix auto
--crt-center-mode halfreal
Gerbicz-Li on
backup/resume on
JSON output on
results.txt append on
OpenCL binary cache on
adaptive queue guard on
startup autotune on for full runs
```

For a clean benchmark:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --iters 30000 \
  --quiet \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --queue-guard 0 \
  --no-startup-autotune
```

## Main modes

```text
--modulus best      planner chooses the available mode
--modulus crt       GF(M61^2) x GF(M31^2), CRT/Garner
--modulus gf61      GF(M61^2) only
--modulus gf31      GF(M31^2) only, when safe for the exponent
```

For the current BananaNTT path:

```text
--modulus crt --crt-odd-radix 9 --crt-center-mode halfreal
```

## Odd-radix controls

```text
--crt-odd-radix auto      automatic planner
--crt-odd-radix off       pure power-of-two CRT path
--crt-odd-radix 1         alias for no odd split
--crt-odd-radix 3         force mixed radix 3
--crt-odd-radix 9         force mixed radix 9
```

Recommended explicit odd9 test:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --device 1 \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume
```

Pure power-of-two comparison:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix off \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --device 1 \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume
```

## Row core and LDS controls

The mixed odd path has two axes:

```text
odd axis    : radix 3 or 9
row axis    : power-of-two half-real NTT
```

The important controls are:

```text
--crt-mixed-row-core auto|lds|lds512|lds1024|generic
--crt-mixed-row-stage N
--crt-mixed-row-center N
--crt-local-square N
--crt-lds-stage N
```

Aliases:

```text
--crt-local-square N       same role as --crt-mixed-row-center N
--crt-lds-stage N          same role as --crt-mixed-row-stage N
```

Stable RTX 3080 setting:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 512 \
  --crt-mixed-row-center 512 \
  --device 1 \
  --iters 5000 \
  --profile-kernels
```

Test 1024 center or stage:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-mixed-row-core lds \
  --crt-mixed-row-stage 1024 \
  --crt-mixed-row-center 1024 \
  --device 1 \
  --iters 2000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume
```

## Startup autotune

Startup autotune is intended to make the default run pick a good row center/stage on different GPUs.

Typical full run:

```bash
./prmers_opencl_prp 142606357 --device 1
```

Force autotune even for a short test:

```bash
./prmers_opencl_prp 9437189 \
  --device 1 \
  --iters 5000 \
  --startup-autotune \
  --crt-autotune-iters 1200 \
  --no-gerbicz \
  --no-backup \
  --no-resume
```

Disable autotune:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-startup-autotune
```

or:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-autotune
```

## Small row stages

The 512 row kernels are the most tuned path.

For smaller row stages such as 8, 16, 32, 64, 128 and 256, the code has batch variants to reduce overhead. The user-visible label is:

```text
small32-batch8=on
```

This means several radix-32 style row transforms are batched together. It does not mean the row stage became radix 8.

Useful A/B test:

```bash
./prmers_opencl_prp 9437189 \
  --device 1 \
  --crt-odd-radix 9 \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --no-startup-autotune
```

Disable the batch8 path:

```bash
PRMERS_CRT_MIXED_SMALL32_X8=0 \
./prmers_opencl_prp 9437189 \
  --device 1 \
  --crt-odd-radix 9 \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --no-startup-autotune
```

## Validation

Quick odd9 GPU reference validation:

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
  --profile-kernels \
  --no-backup \
  --no-resume
```

Expected line:

```text
CRT halfreal validator: OK
```

Small radix 3 validation:

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=4 \
./prmers_opencl_prp 11213 \
  --modulus crt \
  --crt-odd-radix 3 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --crt-halfreal-dump hr11213_odd3 \
  --crt-halfreal-dump-count 2048 \
  --device 1 \
  --iters 1
```

Small radix 9 validation:

```bash
PRMERS_CRT_HALFREAL_DUMP_ALWAYS=1 \
PRMERS_CRT_HALFREAL_CPU_REF_ITERS=1 \
./prmers_opencl_prp 3021377 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-halfreal-dump hr3021377_odd9 \
  --crt-halfreal-dump-count 4096 \
  --device 1 \
  --iters 1
```

## Intermediate residue readback

To print a residue during a run:

```bash
./prmers_opencl_prp 10000019 \
  --device 0 \
  --res64-every 1000
```

Note:

```text
--res64-every N
```

forces regular readback and synchronization. It is useful for debugging, but it slows benchmarks.

## Gerbicz-Li

Gerbicz-Li is enabled by default.

Default behavior:

```text
quiet boundary updates
GPU full-check path
GPU D update
full checks printed
errors printed
```

Disable for benchmarks:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-gerbicz
```

Main controls:

```text
--gerbicz-seconds S            target full Li check spacing
--gerbicz-boundary-seconds S   target D update spacing when B is automatic
--gerbicz-b B                  force Li block size
--gerbicz-checklevel N         force full check every N boundaries
--gerbicz-verbose              print every boundary update
--gerbicz-progress             print full-check progress
--gerbicz-host                 use host GMP full-check
--no-gerbicz                   disable Gerbicz-Li
```

Fast injected-error test:

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

## Backup and resume

Default backup file:

```text
save/M<p>.bananantt.chk
```

Resume is automatic.

Disable resume for clean tests:

```bash
./prmers_opencl_prp 216091 --device 1 --no-resume
```

Disable backup and resume for benchmarks:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --iters 30000 \
  --quiet \
  --no-backup \
  --no-resume
```

Backup controls:

```text
--backup-seconds S
--backup-dir DIR
--save-file FILE
--resume-file FILE
--no-backup
--no-resume
```

Ctrl-C should save a backup before exit when backup is enabled.

## OpenCL binary cache

The first run compiles OpenCL and writes a binary cache in:

```text
.ocl_cache/
```

Next runs with the same GPU, driver, kernel source and build options should load the cache.

Controls:

```text
PRMERS_OCL_BINARY_CACHE=0       disable binary cache
PRMERS_OCL_CACHE_DIR=/path      choose cache directory
PRMERS_SHOW_OCL_BUILD=0         hide build/cache messages
PRMERS_OCL_BUILD_SPINNER=0      simple build messages
```

Reset cache:

```bash
rm -rf .ocl_cache
```

## Queue guard and Ctrl-C

Default queue guard is adaptive:

```text
--queue-guard auto
--queue-guard-seconds 2
```

It limits Ctrl-C latency without forcing a fixed `clFinish` every few iterations.

Pure benchmark:

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

## Output files

Default output:

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

The JSON is intentionally close to the PrMers result shape, with:

```json
"program": {
  "name": "banana",
  "version": "0.79.00-alpha"
}
```

The version should be kept in one global program-version variable and bumped for each release.

## Profiling

Use profiling only for kernel breakdowns:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --crt-odd-radix 9 \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume
```

Do not use `--profile-kernels` for final throughput numbers.

Expected odd9 labels include:

```text
crt_mixed_pack_weight_odd_fwd...
crt_mixed_lds512_forward...
crt_mixed_lds512_center...
crt_mixed_lds512_inverse...
crt_mixed_odd...precrt...
crt_garner...
crt_carry...
```

For small row stages, labels may include:

```text
crt_mixed_lds32_forward_batch8...
crt_mixed_lds32_inverse_batch8...
```

## Useful quick commands

Compile:

```bash
g++ -O3 -DNDEBUG -std=c++20 -march=native -pthread prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

Warm OpenCL cache:

```bash
./prmers_opencl_prp 216091 --device 1 --iters 1 --no-gerbicz --no-backup --no-resume
```

Fast benchmark:

```bash
./prmers_opencl_prp 142606357 --device 1 --iters 30000 --quiet --no-gerbicz --no-backup --no-resume --queue-guard 0
```

Odd9 profile:

```bash
./prmers_opencl_prp 142606357 --device 1 --crt-odd-radix 9 --iters 5000 --profile-kernels --no-gerbicz --no-backup --no-resume
```

Normal CRT comparison:

```bash
./prmers_opencl_prp 142606357 --device 1 --crt-odd-radix off --iters 5000 --profile-kernels --no-gerbicz --no-backup --no-resume
```

Full run:

```bash
./prmers_opencl_prp 142606357 --device 1
```

## Notes for contributors

This is a research branch.

Important points:

```text
The current path works as a direct 2D odd x 2^m GPU transform.
It is not the same as Yves' Four-Step method.
Half-real symmetry is used on the power-of-two axis.
Odd-axis row conjugate symmetry is not assumed.
Radix 9 can still win because it reduces transform size.
The 512 LDS row path is the most optimized path today.
Small row stages still need more work.
```

Testing on more GPUs is useful, especially RTX 40/50, RDNA, MI-series and Intel Arc.
