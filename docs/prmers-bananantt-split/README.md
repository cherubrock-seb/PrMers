# PrMers BananaNTT Split

GPU PRP prototype for Mersenne numbers using the PrMers OpenCL CRT engine and a mixed odd-radix BananaNTT path.

Repository: [https://github.com/cherubrock-seb/PrMers/](https://github.com/cherubrock-seb/PrMers/)

CPU prototype / original mixed 2D idea: [https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast](https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast)

BananaNTT split notes: [https://github.com/cherubrock-seb/PrMers/tree/main/docs/prmers-bananantt-split](https://github.com/cherubrock-seb/PrMers/tree/main/docs/prmers-bananantt-split)

## Goal

This branch tests a GPU implementation of the mixed-radix idea:

```text
N = odd * 2^m
```

The odd axis is handled by a small radix `3` or `9` transform.  
The `2^m` axis keeps the half-real NTT layout.  
The result is reconstructed through CRT / Garner / carry.

Current main path:

```text
GF(M61^2) x GF(M31^2)
odd radix 9
half-real rows
OpenCL GPU kernels
CRT Garner bridge
Gerbicz-Li checks
PrMers-style result files
```

## Relation to Yves Gallot's Four-Step method

This is not the same method as Yves Gallot's new `3*2^n` half-size NTT in `mersenne2`.

Yves' method keeps the odd radix in the center of a Four-Step transform. This preserves a cleaner half-real structure and can reduce the number of independent odd rows.

BananaNTT Split uses a direct 2D layout:

```text
odd axis first
half-real transform on the 2^m axis
odd inverse axis
CRT/Garner/carry
```

So it does not assume that the odd rows are simple conjugate copies. For radix 9 it computes the odd rows directly. Even without the extra odd-axis symmetry shortcut, the transform can still be useful because the global transform size is smaller in some exponent windows.

## Build

```bash
g++ -O3 -DNDEBUG -std=c++20 -march=native -pthread prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

`-lgmp` is used by the exact CPU reference checks and some validation paths.

## Quick run

```bash
./prmers_opencl_prp 142606357 --device 1
```

Default behavior:

```text
--modulus crt
--crt-odd-radix auto
--crt-center-mode halfreal
Gerbicz-Li enabled
backup/resume enabled
JSON result enabled
append to results.txt enabled
OpenCL binary cache enabled
adaptive queue guard enabled
startup autotune enabled for full runs
```

For a pure benchmark, disable safety and output overhead:

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
--modulus gf61       use only GF(M61^2)
--modulus gf31       use only GF(M31^2)
--modulus crt        use GF(M61^2) x GF(M31^2)
--modulus best       planner choice
```

Odd-radix controls:

```text
--crt-odd-radix auto   automatic planner
--crt-odd-radix off    normal power-of-two CRT path
--crt-odd-radix 1      same idea as off for the mixed planner
--crt-odd-radix 3      force mixed radix 3
--crt-odd-radix 9      force mixed radix 9
```

The main tuned path is usually:

```bash
./prmers_opencl_prp 142606357 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --device 1
```

## Mixed CRT/PFA odd-radix path

Hot-loop shape:

```text
pack digits + odd DFT
2^m half-real row NTT
row center square
inverse 2^m half-real row NTT
odd inverse DFT + unweight
CRT/Garner/carry
```

The current radix 9 path uses a compact `3 x 3` decomposition for the odd transform.

Useful row controls:

```text
--crt-mixed-row-core auto|lds|lds512|lds1024|generic
--crt-mixed-row-stage N
--crt-mixed-row-center N
--crt-local-square N
--crt-lds-stage N
```

The `512` row center is the most optimized path today.  
Small stages such as `32` are now batched, but still need more tuning.

## Startup autotune

Startup autotune can test a few row-center / row-stage combinations at launch and keep the fastest one for the current GPU.

```bash
./prmers_opencl_prp 142606357 --device 1 --startup-autotune
```

Disable it:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-startup-autotune
```

Tune the amount of work:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --startup-autotune \
  --crt-autotune-iters 1200
```

Useful when comparing RTX 30/40/50, RDNA, MI-series or Intel Arc GPUs.

## Validation

Fast validation against the mixed GPU reference:

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

Small radix 3 test:

```bash
./prmers_opencl_prp 11213 \
  --modulus crt \
  --crt-odd-radix 3 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --device 1 \
  --iters 1
```

Small radix 9 test:

```bash
./prmers_opencl_prp 216091 \
  --modulus crt \
  --crt-odd-radix 9 \
  --crt-center-mode halfreal \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 4 \
  --device 1 \
  --iters 1
```

## Benchmark examples

Radix 9 profile:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --crt-odd-radix 9 \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --no-startup-autotune
```

Normal CRT comparison:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --crt-odd-radix off \
  --iters 5000 \
  --profile-kernels \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --no-startup-autotune
```

Small-stage case:

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

Expected labels for the batched small row path:

```text
crt_mixed_lds32_forward_batch8_61
crt_mixed_lds32_inverse_batch8_61
crt_mixed_lds32_forward_batch8_31
crt_mixed_lds32_inverse_batch8_31
```

`batch8` means eight radix-32 row transforms per workgroup. It is still a radix/LDS32 stage.

## Gerbicz-Li

Gerbicz-Li is enabled by default for full runs.

Default backend:

```text
gpu-fullcheck + gpu-D-update
```

Disable it for benchmarks:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-gerbicz
```

Main controls:

```text
--gerbicz-seconds S
--gerbicz-boundary-seconds S
--gerbicz-b B
--gerbicz-checklevel N
--gerbicz-verbose
--gerbicz-progress
--gerbicz-host
--no-gerbicz
```

Fast error-injection test:

```bash
./prmers_opencl_prp 10000019 \
  --device 0 \
  --iters 8000 \
  --no-resume \
  --no-backup \
  --gerbicz-b 1024 \
  --gerbicz-checklevel 1 \
  --error-iter 2000 \
  --error-limb 0 \
  --error-delta 1
```

More visible check progress:

```bash
./prmers_opencl_prp 10000019 \
  --device 0 \
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

## Intermediate residues

Print a residue regularly:

```bash
./prmers_opencl_prp 10000019 \
  --device 0 \
  --res64-every 1000
```

This is useful for debugging and comparing runs, but it forces readbacks and synchronization. Do not use it for final speed measurements.

## Backup and restore

Backup and resume are enabled by default.

Default backup file:

```text
save/M<p>.bananantt.chk
```

Run with backup every 120 seconds:

```bash
./prmers_opencl_prp 142606357 --device 1 --backup-seconds 120
```

Disable resume for clean tests:

```bash
./prmers_opencl_prp 216091 --device 1 --no-resume
```

Disable backup and resume for benchmark:

```bash
./prmers_opencl_prp 142606357 \
  --device 1 \
  --iters 30000 \
  --quiet \
  --no-backup \
  --no-resume
```

Output path controls:

```text
--backup-dir DIR
--save-file FILE
--resume-file FILE
```

## Ctrl-C and queue guard

The queue guard limits how much OpenCL work can be queued ahead, so Ctrl-C does not wait forever for a huge queue to drain.

Default:

```text
--queue-guard auto
--queue-guard-seconds 2
```

Disable for pure benchmark:

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

## OpenCL binary cache

The first run compiles OpenCL kernels and stores binaries in `.ocl_cache/`. Later runs with the same GPU, driver, source, build options and program version load the cached binaries.

Controls:

```text
PRMERS_OCL_BINARY_CACHE=0
PRMERS_OCL_CACHE_DIR=/path/to/cache
PRMERS_SHOW_OCL_BUILD=0
PRMERS_OCL_BUILD_SPINNER=0
```

Clear cache:

```bash
rm -rf .ocl_cache
```

## Output files

Default files:

```text
prmers_bananantt_M<p>.json
results.txt
save/M<p>.bananantt.chk
<p>/proof/
```

Path controls:

```text
--output-dir DIR
--json-file FILE
--results-file FILE
--proof-dir DIR
--no-json
--no-results
```

The JSON is PrMers-style and includes:

```text
status
exponent
worktype
res64
res2048
residue-type
errors
shift-count
fft-length
proof
program
os
timestamp
checksum
```

`program.name` is `banana`.

## Worktodo

If no exponent is given, the program can read a PrMers-style `worktodo.txt` from the current directory.

Example:

```bash
./prmers_opencl_prp --device 1
```

## Current status

This is still experimental code.

Working features:

```text
CRT GF(M61^2) x GF(M31^2)
odd radix 3/9
half-real row path
OpenCL binary cache
startup autotune
Gerbicz-Li checks
backup/restore
Ctrl-C backup
PrMers-style JSON/results
intermediate res64 output
error injection
```

Known optimization work:

```text
small row stages below 512
better radix32 leaf kernels
async Gerbicz full-check with separate buffers
more GPU-specific tuning
more testing outside RTX 3080
```

## Credits

Original `mersenne2` ideas and reference work by Yves Gallot:

[https://github.com/galloty/mersenne2](https://github.com/galloty/mersenne2)

PrMers / BananaNTT GPU experiments by cherubrock:

[https://github.com/cherubrock-seb/PrMers/](https://github.com/cherubrock-seb/PrMers/)
