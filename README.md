# PrMers

GPU-accelerated PRP, Lucas-Lehmer, P-1 and ECM testing for Mersenne numbers.

https://github.com/cherubrock-seb/PrMers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherubrock-seb/PrMers/blob/main/prmers.ipynb)

Releases for Linux, macOS and Windows are available here:

[https://github.com/cherubrock-seb/PrMers/releases](https://github.com/cherubrock-seb/PrMers/tags)

PrMers is an OpenCL GPU program focused on long modular arithmetic runs for numbers of the form `2^p - 1`. It supports PRP, Lucas-Lehmer, P-1 and ECM workflows, with checkpointing, result JSON output, Prime95 compatible handoff files, worktodo parsing, and an optional web GUI.

The default backend policy is automatic. PrMers selects between the Marin backend by Yves Gallot and the optional Aevum backend from the workload, register count and transform sizes. Marin uses an integer IBDWT-style transform modulo `2^64 - 2^32 + 1`. Aevum exposes GPUOwl/PRPLL paired integer NTT arithmetic over `GF(M31^2)` and `GF(M61^2)` through a PrMers `engine::Reg` adapter. The v99.7 policy uses measured transform-size thresholds per workload. The adapter retains GPU-side register equality for Gerbicz checks and bounded Mersenne carry canonicalization for residue export.

Use `-engine-marin` to force Marin, `-aevum` to force Aevum, or `-aevum-auto` to request the default policy explicitly. The historical `-marin` option keeps its previous meaning and selects the internal PrMers NTT path. The standalone engine is intended for publication at https://github.com/cherubrock-seb/aevum-engine.
## Contents

- [What PrMers can do](#what-prmers-can-do)
- [Quick start](#quick-start)
- [Build from source](#build-from-source)
- [Command line options](#command-line-options)
- [PRP and proof generation](#prp-and-proof-generation)
- [Lucas-Lehmer modes](#lucas-lehmer-modes)
- [P-1 factoring](#p-1-factoring)
- [ECM factoring](#ecm-factoring)
- [worktodo.txt and AutoPrimeNet](#worktodotxt-and-autoprimenet)
- [Prime95 and mprime interop](#prime95-and-mprime-interop)
- [Web GUI](#web-gui)
- [GPU memory test](#gpu-memory-test)
- [NTT and IBDWT transform sizes](#ntt-and-ibdwt-transform-sizes)
- [Benchmarks](#benchmarks)
- [Backend and code](#backend-and-code)
- [Related inspiration](#related-inspiration)
- [Must read papers](#must-read-papers)
- [License and upstream attribution](#license-and-upstream-attribution)

## What PrMers can do

| Area | Status | Notes |
|---|---|---|
| Mersenne PRP | Supported | Default mode, automatic Marin/Aevum backend selection |
| Mersenne Lucas-Lehmer | Supported | Safe GL mode, classic unsafe mode, doubling safe mode |
| P-1 factoring | Supported | Stage 1, default V-trace Stage 2, classic Stage 2 fallback, resume export, Prime95 handoff |
| P-1 ultra-low-memory mode | Supported | 1-register Stage 1 and 1-register Stage 2 product-exponent path |
| ECM factoring | Supported | Edwards and Montgomery variants, optional Prime95 Stage 2 handoff |
| Wagstaff PRP | Supported | `W = (2^p + 1) / 3` with `-wagstaff` |
| Mersenne cofactors | Supported | PRP with known factors using `-factors` |
| worktodo.txt | Supported | PRP and Pminus1 parsing, including Prime95 compatible Pminus1 metadata |
| Web GUI | Supported | Local browser interface for monitoring and worktodo editing |
| GPU memory test | Supported | OpenCL VRAM and stability test |

## Quick start

Run a PRP test on a Mersenne number:

```bash
./prmers 136279841
```

The command above uses automatic backend selection. Force a backend only for testing or benchmarking:

```bash
./prmers 136279841 -engine-marin
./prmers 136279841 -aevum
```

Run a safe Lucas-Lehmer test:

```bash
./prmers 127 -ll
```

Run P-1 factoring with Stage 1 and Stage 2. Normal-memory Stage 2 uses the V-trace path by default, with conservative automatic `D` selection:

```bash
./prmers 367 -pm1 -b1 11981 -b2 38971
```

Force the previous classic Stage 2 path when comparing or debugging:

```bash
./prmers 367 -pm1 -b1 11981 -b2 38971 -pm1-vtrace-off
```

Run the MM31 ultra-low-memory P-1 example that uses a 1-register GPU Stage 2:

```bash
./prmers 2147483647 -pm1 -b1 100 -b2 5000 -pm1-ultralowmem -nogcd-stage1
```

Expected test factor for that example:

```text
295257526626031
```

Run ECM:

```bash
./prmers 701 -ecm -b1 6000 -b2 33333 -K 8
```

Use a worktodo file:

```bash
./prmers -worktodo ./worktodo.txt
```

Start the web GUI:

```bash
./prmers -gui -http 3131
```

Then open the URL printed by the program.

## Build from source

### Requirements

- C++20 compiler
- OpenCL runtime and headers
- GMP development library
- Make or CMake

Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y g++ make ocl-icd-opencl-dev opencl-headers libgmp-dev
```

### Linux and macOS with Makefile

```bash
git clone https://github.com/cherubrock-seb/PrMers.git
cd PrMers
make -j"$(nproc)"
```

Build with the optional Aevum shared engine when its source is present under `third_party/aevum`:

```bash
./build_with_aevum_engine.sh
```

For the public repositories, the recommended layout is to keep Aevum in its own GPLv3 repository and attach it to PrMers as a Git submodule or external shared-library dependency:

```bash
git submodule add https://github.com/cherubrock-seb/aevum-engine third_party/aevum
git submodule update --init --recursive
./build_with_aevum_engine.sh
```

Install system-wide:

```bash
sudo make install
```

Installed paths:

```text
/usr/local/bin/prmers
/usr/local/share/prmers/
```

When building a local zip or test copy, pass the kernel path explicitly:

```bash
make clean
make -j"$(nproc)" KERNEL_PATH=./kernels/
```

### Windows with CMake and vcpkg

```powershell
git clone https://github.com/cherubrock-seb/PrMers.git
cd PrMers
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
cd ..

cmake -S . -B build ^
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

Copy required DLLs from `vcpkg\installed\x64-windows\bin` next to `prmers.exe`, or add that directory to `PATH`.

### Windows with MSYS2 UCRT64

```bash
pacman -Syu
pacman -S --noconfirm make \
  mingw-w64-ucrt-x86_64-gcc \
  mingw-w64-ucrt-x86_64-opencl-headers \
  mingw-w64-ucrt-x86_64-opencl-icd-loader \
  mingw-w64-ucrt-x86_64-gmp

make -j"$(nproc)"
```



### Strict backend requests in v99.9

`-aevum` is now strict: when the requested exponent has no admissible FFT3161
plan, PrMers exits with code 2 instead of silently running Marin. Automatic mode
continues to choose Marin normally in that situation. For example, `M216091` is
below the native FFT3161 range, while `M1362763` can be used to exercise the
forced Aevum LL paths.

The legacy internal NTT option `-marin` is rejected with `-llunsafe`, because
that historical Lucas-Lehmer path is not validated. Use automatic mode,
`-engine-marin`, or `-aevum`. The one-register `-pm1-ultralowmem` path remains
Marin-only and an explicit `-aevum` request is rejected before OpenCL and
transform allocation.

### Backend validation matrix (v99.9)

The automatic policy remains active for PRP, all Lucas-Lehmer engine paths, normal P-1, 3-register P-1 low-memory, and ECM. Aevum is selected when its FFT3161 transform meets the workload ratio threshold. The historical one-register `-pm1-ultralowmem` algorithm is the only explicit Marin-only compatibility case because it encodes multiply-by-3 through Marin `fast3`; forced `-aevum` is rejected cleanly before plugin allocation.

The tiny Marin radix-4 kernel used by transforms of 16 to 80 words is fixed in v99.8. This repairs small P-1 low-memory and ECM examples such as `M367` and `M701`.

Run the exhaustive cross-backend validation and create a single report archive:

```bash
PRMERS_TEST_DEVICE=0 PRMERS_MATRIX_PROFILE=standard make test-backend-matrix
```

Profiles are `quick`, `standard`, and `full`. The script covers Auto, forced Aevum, forced Marin, and the internal NTT path across PRP, LL-safe, LL-unsafe, LL-safe2, normal/low/ultra-low-memory P-1, and Edwards/Montgomery ECM. ECM uses a fixed seed so Aevum and Marin execute the same curve. The `full` profile adds complete medium-size Aevum/Marin pairs and compares their final JSON status, residues or factors as well as measured throughput.

```bash
PRMERS_TEST_DEVICE=0 PRMERS_MATRIX_PROFILE=full \
PRMERS_MATRIX_COMPLETE_SECONDS=1200 make test-backend-matrix
```

To rerun only selected cases, use a regular-expression filter:

```bash
PRMERS_MATRIX_CASE_FILTER='ll-safe|pm1-lowmem|ecm-medium' \
PRMERS_MATRIX_PROFILE=full ./tests/run_backend_validation_matrix.sh 0 full
```

The script writes `tests/backend-validation-<timestamp>.tar.gz` and a matching SHA-256 file. The archive contains every command and log plus `summary.tsv`, `comparisons.tsv`, `errors.tsv`, `system.txt`, `combined.log`, and `report-manifest.txt`.

## Command line options

Run the built-in help for the exact option list supported by your binary:

```bash
./prmers -h
```

### General options

| Option | Meaning |
|---|---|
| `<p>` | Exponent for `2^p - 1`, unless `-worktodo` supplies it |
| `-d <id>` | OpenCL device id, default `0` |
| `-c <depth>` | Local carry propagation depth |
| `-t <seconds>` | Checkpoint interval |
| `-f <path>` | Checkpoint and state directory |
| `-config <path>` | Read options from a config file |
| `-worktodo [path]` | Read assignment from worktodo.txt |
| `-profile` | Enable kernel profiling |
| `-debug` | Verbose debug output |
| `-bench` | Run benchmark over supported transform sizes |
| `-v`, `--version` | Print version |
| `-h`, `--help` | Print help |

### Device and stability options

| Option | Meaning |
|---|---|
| `-iterforce <n>` | Force GPU queue synchronization every `n` iterations |
| `-iterforce2 <n>` | Force queue synchronization in P-1 Stage 2 |
| `-memtest` | Run GPU memory and stability test |
| `-memlim <percent>` | Limit memory used by some precompute paths |
| `-maxe <MiB>` | Maximum P-1 exponent chunk size in MiB |
| `-res64_display_interval <n>` | Print Res64 every `n` iterations in Marin mode |

### Mode selection

| Option | Meaning |
|---|---|
| `-prp` | PRP mode, default |
| `-ll` | Lucas-Lehmer safe mode with Gerbicz-Li style checks |
| `-llunsafe` | Classic Lucas-Lehmer mode without error checking |
| `-llsafe2` | Lucas-Lehmer block-doubling safe mode |
| `-wagstaff` | Test `W = (2^p + 1) / 3` |
| `-pm1` | P-1 factoring mode |
| `-ecm` | ECM factoring mode |
| `-aevum` | Force the Aevum `engine::Reg` backend |
| `-engine-marin` | Force the Marin `engine::Reg` backend |
| `-aevum-auto` | Explicitly request automatic Marin/Aevum selection |
| `-marin` | Legacy option: use the internal PrMers NTT path |

Automatic Marin/Aevum selection is the default when no backend option is supplied.


Automatic selection compares the native transform sizes for every engine creation. PRP/LL and multi-register P-1 accept Aevum when its transform is no larger than Marin. P-1 Stage 1 and ECM require a clearer advantage (`Aevum / Marin <= 0.75` by default). Thus `M136279841` P-1 Stage 1 selects Aevum automatically: its FFT3161 transform is 4M words versus 8M for Marin, matching the measured Radeon VII advantage. A forced Aevum Stage 1 uses the generic `square` plus prepared base-3 multiplication path instead of the Marin-specific `fast3` shortcut. `-pm1-ultralowmem` remains Marin-only because its one-register algorithm depends on `fast3`. Stage 1 checkpoints are tagged with their arithmetic backend; incompatible checkpoints are ignored safely.

### PRP and proof options

| Option | Meaning |
|---|---|
| `-proof <level>` | Proof power, `1` to `12`, or `0` to disable proof generation |
| `-noverify` | Skip verification of generated PRP proof |
| `-factors <f1,f2,...>` | Test the remaining cofactor after known Mersenne factors |
| `-gerbiczli` | Disable Gerbicz-Li checks, mainly for benchmarking |
| `-checklevel <k>` | Tune Gerbicz-Li check frequency |
| `-erroriter <i>` | Inject an error at iteration `i` to test recovery |

### Lucas-Lehmer options

| Option | Meaning |
|---|---|
| `-ll` | Safe LL mode |
| `-llunsafe` | Fast classic LL mode, no error checking |
| `-llsafe2` | Block-doubling safe LL mode |
| `-llsafeb <B>` | Override block size for `-llsafe2` |

### P-1 options

| Option | Meaning |
|---|---|
| `-pm1` | Enable P-1 factoring |
| `-b1 <B1>` | Stage 1 bound |
| `-b2 <B2>` | Stage 2 bound |
| `-pm1-vtrace` | Use scalar-trace P-1 Stage 2. This is the default for normal-memory Stage 2, with conservative auto-D |
| `-pm1-vtrace-off` | Disable V-trace and use the previous classic Stage 2 BSGS path |
| `-pm1-vtrace-pair95` | Enable the Pair95 prime-pairing planner for V-trace Stage 2 when supported by the build |
| `-pm1-vtrace-pair95-off` | Disable Pair95 and use the ordinary V-trace prime-pairing planner |
| `-pm1-vtrace-pair95-l <L>` | Set the Pair95 irregular-unit count, for example `2` or `3` |
| `-pm1-vtrace-d <D>` | Force the V-trace giant-step parameter `D`, for example `4620`, `13860`, or `30030` |
| `-pm1-vtrace-auto-d` | Explicitly auto-select `D` for V-trace under a conservative register cap |
| `-pm1-vtrace-auto-d-aggressive` | Auto-select `D` with a larger register cap for normal-size Mersennes |
| `-pm1-vtrace-max-regs <N>` | Register cap used by V-trace auto-D |
| `-pm1-vtrace-auto-batch` | Explicitly enable integrated V-trace `D` plus baby-batch selection. This is the default in current normal-memory V-trace builds |
| `-pm1-vtrace-no-auto-batch` | Disable automatic V-trace baby batching and keep the full baby table for the selected `D`, using segmented GPU memory if needed |
| `-pm1-vtrace-baby-batch <N>` | Force the number of active baby traces kept per Stage 2 pass. If `N` covers all baby residues, no batching is used |
| `-pm1-vtrace-max-batches <N>` | Maximum number of baby-window passes considered by V-trace auto-batching |
| `-pm1-vtrace-negadd-off` | Disable the negative-baby trace optimization and use the older copy/subtract path for comparison |
| `-b1old <B1old>` | Extend Stage 1 from an existing `.save` or `.p95` file |
| `-resume` | Write GMP-ECM `.save` and Prime95 `.p95` resume files |
| `-p95` | Write Prime95 `.p95` resume file |
| `-p95path <path>` | Delegate P-1 Stage 2 to Prime95 or mprime |
| `-nop95stage2` | Disable Prime95 Stage 2 handoff even if `-p95path` is set |
| `-filemers <path>` | Convert a `.mers` state to GMP-ECM `.save` |
| `-K <K>` | Enable the `n^K` Stage 2 variant |
| `-nmax <n>` | Upper bound for the `n^K` variant |
| `-pm1-lowmem` | P-1 low-memory mode, fewer GPU registers |
| `-pm1-ultralowmem` | P-1 ultra-low-memory mode, 1-register Stage 1 and 1-register product-exponent Stage 2 |
| `-nogcd-stage1` | Skip ordinary Stage 1 GCD after writing resume data, useful before Stage 2 |

### ECM options

| Option | Meaning |
|---|---|
| `-ecm` | Enable ECM mode |
| `-b1 <B1>` | ECM Stage 1 bound |
| `-b2 <B2>` | ECM Stage 2 bound |
| `-K <curves>` | Number of curves |
| `-montgomery` | Use Montgomery curve model |
| `-edwards` | Use Edwards curve setup |
| `-ced` | Compute directly in Twisted Edwards coordinates |
| `-cmont` | Compute in Montgomery coordinates |
| `-torsion8` | Use torsion-8 family |
| `-torsion16` | Use torsion-16 family |
| `-notorsion` | Disable torsion family |
| `-iv163` | Use family IV 163 curves |
| `-seed <value>` | Force curve seed |
| `-sigma <value>` | Force Montgomery sigma |
| `-ecm_check_interval <seconds>` | ECM error-check interval |
| `-ecm_progress_ms <ms>` | ECM progress update interval |
| `-p95path <path>` | Delegate ECM Stage 2 to Prime95 or mprime |

### Web GUI options

| Option | Meaning |
|---|---|
| `-gui` | Enable embedded web GUI |
| `-http <port>` | HTTP port, default `3131` |
| `-host <ip>` | HTTP host, default `localhost` |

## PRP and proof generation

Default PRP mode tests `2^p - 1` using GPU modular exponentiation and Gerbicz-Li style checking. Results are written to `results.txt` and to a JSON result file.

Example:

```bash
./prmers 136279841
```

Useful files include:

| File | Purpose |
|---|---|
| `results.txt` | Human-readable result history |
| `<p>_prp_result.json` | JSON result for automation |
| proof files | Optional PRP proof output depending on proof settings |

Cofactor PRP:

```bash
./prmers 10449497 -factors 62696983
```

## Lucas-Lehmer modes

PrMers implements three GPU LL variants.

| Mode | Option | Safety | Notes |
|---|---|---|---|
| LL safe | `-ll` | Gerbicz-Li style checking | Recommended safe LL mode |
| LL classic | `-llunsafe` | No checking | Fast, for quick checks and debugging |
| LL safe2 | `-llsafe2` | Block-doubling checks | Lighter safe mode with block verification |

Safe LL uses the split representation `s = a + b*sqrt(3)` and checks progress periodically. `-llunsafe` uses the classic recurrence `S_{i+1} = S_i^2 - 2` and should not be used for production runs on unstable hardware.

## P-1 factoring

For a factor `q` of `M_p = 2^p - 1`, factors have the form:

```text
q = 2*k*p + 1
```

### Stage 1

Stage 1 computes:

```text
x = 3^(E(B1)*2*p) mod M_p
```

where `E(B1)` is the product of prime powers up to `B1`. Then it tests:

```text
gcd(x - 1, M_p)
```

Example:

```bash
./prmers 541 -pm1 -b1 8099
```

Write resume files:

```bash
./prmers 541 -pm1 -b1 8099 -resume
./prmers 541 -pm1 -b1 8099 -p95
```

Extend from an older Stage 1 bound:

```bash
./prmers 541 -pm1 -b1 20000 -b1old 8099
```

### Stage 2

Standard Stage 2 extends the search from `B1` to `B2`.

```bash
./prmers 367 -pm1 -b1 11981 -b2 38971
```

For normal-memory P-1, Stage 2 uses the V-trace path by default. Let `H` be the Stage 1 result. Instead of accumulating one term per prime with powers of `H`, V-trace works with the scalar trace:

```text
V_n = H^n + H^(-n) mod M_p
```

For a giant step `kD` and baby step `j`, the difference

```text
V_(kD) - V_j
```

covers both `kD - j` and `kD + j`, because it contains the factors `(H^(kD-j)-1)` and `(H^(kD+j)-1)` up to multiplication by an invertible term. This gives a compact baby/giant Stage 2 using one scalar residue per baby trace. By default PrMers uses automatic `D` selection and, when useful, automatic baby batching. `D` can also be forced explicitly with `-pm1-vtrace-d`.

Examples:

```bash
./prmers 1362763 -pm1 -b1 29 -b2 6910159 -nogcd-stage1 \
  -factors 46333943,282345414919

./prmers 1362763 -pm1 -b1 29 -b2 6910159 -nogcd-stage1 \
  -factors 46333943,282345414919 -pm1-vtrace-d 30030
```

Expected factor for this regression test:

```text
28401397572100073
```

#### Pair95 prime pairing

Some builds include a Pair95 planner for V-trace Stage 2, inspired by the prime-pairing method described by Atnashev and Woltman. It extends the ordinary `kD ± j` pairing by allowing irregular precomputed units such as `unit`, `unit + D`, `unit + 3D`, `unit + 7D`, and so on. This is a Stage 2 planning and coverage option; it does not change the underlying Marin modular arithmetic.

Pair95 can be selected explicitly with `-pm1-vtrace-pair95`, disabled with `-pm1-vtrace-pair95-off`, and tuned with `-pm1-vtrace-pair95-l <L>`. When Pair95 is enabled, `D` and the baby-window size can still be forced with `-pm1-vtrace-d <D>` and `-pm1-vtrace-baby-batch <N>`.

#### V-trace `D`, baby residues and batching

For a selected giant-step parameter `D`, the V-trace path stores one baby trace for each odd residue `j <= D/2` with `gcd(j,D)=1`. The number of baby traces is therefore:

```text
m(D) = phi(D) / 2
```

If all `m(D)` baby traces fit efficiently, Stage 2 can run in one pass. If only `b` active baby traces are kept in GPU memory, PrMers splits the baby table into:

```text
P = ceil(m(D) / b)
```

baby-window passes.

A useful first-order cost model is:

```text
Cost(D,b) ~= A * (P * (B2-B1) / D)
           + C * prime_terms
           + E * baby_precompute
           + batch_overhead
```

The important term for the giant recurrence is `P/D`, not just `D`. This is why two different settings can have similar runtime. For example, if `B = B2-B1`:

```text
D = 630:   phi(D)/2 = 72,  b = 72, P = 1  -> P/D = 1/630
D = 1260:  phi(D)/2 = 144, b = 72, P = 2  -> P/D = 2/1260 = 1/630
D = 2310:  phi(D)/2 = 240, b = 72, P = 4  -> P/D = 4/2310
```

`D=2310` may show a much higher `p/s` because each pass covers only a subset of residues and skips many primes quickly. The more useful comparison is usually `term/s` and the final ETA.

Examples:

```bash
# Let PrMers choose D and the baby-window batch size.
./prmers 205271257 -pm1 -b1 1301000 -b2 250000000

# Force D and keep all 72 baby residues for D=630.
./prmers 205271257 -pm1 -b1 1301000 -b2 250000000 \
  -pm1-vtrace-d 630 -pm1-vtrace-baby-batch 72

# Force D=2310 and keep 72 active baby traces per pass.
./prmers 205271257 -pm1 -b1 1301000 -b2 250000000 \
  -pm1-vtrace-d 2310 -pm1-vtrace-baby-batch 72
```

Use `-pm1-vtrace-no-auto-batch` when you want to force a full baby table for the selected `D` and compare it against the automatic batching choice.

#### Segmented GPU register space

On some OpenCL drivers, especially NVIDIA, a single `cl_mem` buffer can be limited by `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, often much smaller than total VRAM. Recent PrMers builds can split the Marin register slab into several GPU-only OpenCL buffers when the logical register space is larger than one OpenCL allocation but still fits in total VRAM.

This segmented register space is used automatically by normal-memory V-trace Stage 2 when needed. It is not host-backed swapping: the hot loop remains GPU-only. With `PRMERS_GPU_ALLOC_DIAG=1`, the log reports both the logical register slab and the segmented allocation plan, for example:

```text
[MARIN-SEGMENTED] logical regs=86 reg slab=6.72 GiB exceeds OpenCL max single allocation=2.41 GiB.
[MARIN-SEGMENTED] using 4 GPU-only cl_mem segment(s), ... no host backing in hot loop.
```

Useful environment variables for debugging:

| Variable | Meaning |
|---|---|
| `PRMERS_GPU_ALLOC_DIAG=1` | Print GPU memory planning and allocation diagnostics |
| `PRMERS_PM1_VTRACE_BABY_BATCH=<N>` | Force V-trace active baby traces per pass, same purpose as `-pm1-vtrace-baby-batch <N>` |
| `PRMERS_PM1_VTRACE_PAIRING95=1` | Enable Pair95 prime-pairing in builds where it is available |
| `PRMERS_PM1_VTRACE_PAIRING95_DISABLE=1` | Disable Pair95 and use the ordinary V-trace planner |
| `PRMERS_PM1_VTRACE_PAIRING95_L=<L>` | Set the Pair95 irregular-unit count |
| `PRMERS_PM1_VTRACE_NO_AUTO_BATCH=1` | Disable V-trace auto-batching |
| `PRMERS_MARIN_SEGMENTED_DISABLE=1` | Disable segmented register space and require a single flat register buffer |
| `PRMERS_MARIN_SEGMENTED_MAXALLOC_FRAC=<f>` | Fraction of `CL_DEVICE_MAX_MEM_ALLOC_SIZE` used when sizing each segment |
| `PRMERS_MARIN_SEGMENTED_SCRATCH_REGS=<N>` | Scratch registers reserved per segment for cross-segment operations |

When benchmarking the internal V-trace path, do not set `-p95path`: that option delegates P-1 Stage 2 to Prime95 or mprime.

V-trace Stage 2 checkpoints are compact: they save the accumulator and the current giant recurrence state, and rebuild deterministic baby tables on resume instead of writing the full baby table to disk.

To force the previous classic Stage 2 path, add:

```bash
-pm1-vtrace-off
```

The classic Stage 2 path uses GPU precomputation and prime sweeps. The `n^K` variant can be enabled with:

```bash
./prmers 367 -pm1 -b1 11981 -b2 38971 -K 8 -nmax 200000
```

### Ultra-low-memory P-1

`-pm1-ultralowmem` is designed for huge transforms where a normal multi-register Stage 2 does not fit in VRAM.

Current behavior:

| Stage | Method |
|---|---|
| Stage 1 | 1 GPU register, fast3 path, Gerbicz-Li disabled |
| Stage 2 | 1 GPU register, product-exponent path |

The Stage 2 ultra-low-memory path computes the product of Stage 2 primes into the exponent and runs one direct GPU exponentiation:

```text
3^(E(B1)*2*p*product_primes(B1,B2]) mod M_p
```

It then tests `gcd(x - 1, M_p)`. This is slower than a full-memory BSGS-style Stage 2, but it fits on GPUs where a 2-register or table-based Stage 2 does not fit.

MM31 validation example:

```bash
./prmers 2147483647 -pm1 -b1 100 -b2 5000 -pm1-ultralowmem -nogcd-stage1
```

Expected known factor:

```text
295257526626031
```

## ECM factoring

Basic ECM:

```bash
./prmers p -ecm -b1 B1 -b2 B2 -K curves
```

Examples:

```bash
./prmers 701 -ecm -b1 6000 -K 8
./prmers 701 -ecm -b1 6000 -b2 33333 -K 8
```

Curve and arithmetic options include Montgomery, Edwards, torsion variants, seeds and sigma values. Use `./prmers -h` for the exact list supported by your build.

## worktodo.txt and AutoPrimeNet

PrMers can read GIMPS-style `worktodo.txt` assignments.

```bash
./prmers -worktodo
./prmers -worktodo ./worktodo.txt
```

### PRP format

```text
PRP=AID,k,b,n,c,tf_bits,tests_saved
```

Example:

```text
PRP=DEADBEEFCAFEBABEDEADBEEFCAFEBABE,1,2,197493337,-1,76,0
```

### Pminus1 format

Prime95-compatible P-1 format:

```text
Pminus1=k,b,n,c,B1,B2[,how_far_factored][,B2_start][,"factors"]
Pminus1=AID,k,b,n,c,B1,B2[,how_far_factored][,B2_start][,"factors"]
```

Examples:

```text
Pminus1=AID,1,2,160575647,-1,900000,32000000
Pminus1=AID,1,2,160575647,-1,900000,32000000,79
Pminus1=AID,1,2,160575647,-1,900000,32000000,79,5000000
Pminus1=AID,1,2,11,-1,100,200,79,"23"
```

The parser keeps these fields separate:

| Field | Meaning |
|---|---|
| `how_far_factored` | Trial factoring depth, for example `79` means TF completed to `2^79` |
| `B2_start` | Optional Stage 2 start bound |
| `"factors"` | Quoted known-factor list |

So the trailing `79` in this line is not a known factor:

```text
Pminus1=AID,1,2,160575647,-1,900000,32000000,79
```

It is interpreted as trial factoring completed to `2^79`.

### AutoPrimeNet

AutoPrimeNet can fetch assignments, monitor progress and submit results.

Project:

https://github.com/tdulcet/AutoPrimeNet

Windows:

```powershell
autoprimenet.exe --setup
autoprimenet.exe
prmers.exe -worktodo worktodo.txt
```

Linux or macOS:

```bash
python3 autoprimenet.py --setup
python3 -OO autoprimenet.py
./prmers -worktodo worktodo.txt
```

For multiple GPUs or workers, use one working directory per worker.

## Prime95 and mprime interop

### P-1 Stage 2 handoff

PrMers can run P-1 Stage 1 and let Prime95 or mprime run Stage 2.

```bash
./prmers 75931 -pm1 -b1 100 -b2 200000000 -p95path /home/sebastien/gimps/v31_31.04_b05c
```

Windows:

```powershell
prmers.exe 75931 -pm1 -b1 100 -b2 200000000 -p95path C:\gimps\v31_31.04_b05c
```

What happens:

1. PrMers runs P-1 Stage 1.
2. PrMers writes a Prime95 Stage 1 state file.
3. PrMers copies it as `mXXXXXXX` in the Prime95 directory.
4. PrMers writes a `Pminus1` line to Prime95 `worktodo.txt`.
5. Prime95 or mprime runs Stage 2.
6. PrMers reads `results.json.txt` and reports `NF` or `F`.

Example Prime95 line:

```text
Pminus1=1,2,75931,-1,100,200000000,68
```

With known factors:

```text
Pminus1=1,2,10449497,-1,1440000,1440000,68,"62696983"
```

### ECM Stage 2 handoff

PrMers can run ECM Stage 1 and let Prime95 or mprime run ECM Stage 2.

```bash
./prmers 757 -ecm -b1 97 -b2 9500 -K 15 -p95path /home/sebastien/gimps/v31_31.04_b05c
```

Example Prime95 line:

```text
ECMSTAGE2=N/A,1,2,757,-1,"resume_p757_ECM_TE_B1_97_c000006.p95",9500
```

## Web GUI

Start the GUI:

```bash
./prmers -gui -http 3131
```

Common options:

```bash
./prmers -gui -host 127.0.0.1 -http 3131
./prmers -gui -host 0.0.0.0 -http 3131
```

The GUI can monitor progress, show logs, inspect results and help edit `worktodo.txt`. It now also displays the configured backend mode, the active backend, workload, Marin/Aevum transform sizes, the FFT3161 plan and the reason for the automatic decision. ECM initialization, Stage 1 and Stage 2 update the main progress bar. The settings editor can select Auto, forced Aevum, forced Marin or the internal PrMers NTT path, and can generate ECM work entries.

Example with backend telemetry:

```bash
./prmers 136279841 -pm1 -b1 1000000 -gui -http 3131 -d 0 --noask
```

Open the printed URL and look at the **Computation engine** card. For this case the expected decision is `Auto -> Aevum`, with Aevum 4M words versus Marin 8M words.

## GPU memory test

```bash
./prmers -memtest
./prmers -memtest -d 1
```

The memory test scans GPU VRAM with several patterns and reports bandwidth, coverage and errors.

## NTT and IBDWT transform sizes

For a given exponent p, PrMers chooses an NTT/IBDWT size N:

| Exponent p range | N | Structure |
|---|---:|---|
| 3-113 | 4 | 2^2 |
| 127-239 | 8 | 2^3 |
| 241-463 | 16 | 2^4 |
| 467-919 | 32 | 2^5 |
| 929-1153 | 40 | 5*2^3 |
| 1163-1789 | 64 | 2^6 |
| 1801-2239 | 80 | 5*2^4 |
| 2243-3583 | 128 | 2^7 |
| 3593-4463 | 160 | 5*2^5 |
| 4481-6911 | 256 | 2^8 |
| 6917-8629 | 320 | 5*2^6 |
| 8641-13807 | 512 | 2^9 |
| 13829-17257 | 640 | 5*2^7 |
| 17291-26597 | 1024 | 2^10 |
| 26627-33247 | 1280 | 5*2^8 |
| 33287-53239 | 2048 | 2^11 |
| 53267-66553 | 2560 | 5*2^9 |
| 66569-102397 | 4096 | 2^12 |
| 102407-127997 | 5120 | 5*2^10 |
| 128021-204797 | 8192 | 2^13 |
| 204803-255989 | 10240 | 5*2^11 |
| 256019-393209 | 16384 | 2^14 |
| 393241-491503 | 20480 | 5*2^12 |
| 491527-786431 | 32768 | 2^15 |
| 786433-982981 | 40960 | 5*2^13 |
| 983063-1507321 | 65536 | 2^16 |
| 1507369-1884133 | 81920 | 5*2^14 |
| 1884193-3014653 | 131072 | 2^17 |
| 3014659-3768311 | 163840 | 5*2^15 |
| 3768341-5767129 | 262144 | 2^18 |
| 5767169-7208951 | 327680 | 5*2^16 |
| 7208977-11534329 | 524288 | 2^19 |
| 11534351-14417881 | 655360 | 5*2^17 |
| 14417927-22020091 | 1048576 | 2^20 |
| 22020127-27525109 | 1310720 | 5*2^18 |
| 27525131-44040187 | 2097152 | 2^21 |
| 44040253-55050217 | 2621440 | 5*2^19 |
| 55050253-83886053 | 4194304 | 2^22 |
| 83886091-104857589 | 5242880 | 5*2^20 |
| 104857601-167772107 | 8388608 | 2^23 |
| 167772161-209715199 | 10485760 | 5*2^21 |
| 209715263-318767093 | 16777216 | 2^24 |
| 318767107-398458859 | 20971520 | 5*2^22 |
| 398458889-637534199 | 33554432 | 2^25 |
| 637534277-796917757 | 41943040 | 5*2^23 |
| 796917763-1207959503 | 67108864 | 2^26 |
| 1207959559-1509949421 | 83886080 | 5*2^24 |
| 1509949440 and above, including MM31 | 167772160 | 5*2^25 |


Note: for MM31 (`p = 2147483647`), the valid Marin transform size is `167772160 = 5*2^25`. Pure `2^27` is not valid for the Goldilocks root layout used here.

## Benchmarks

Performance depends on GPU, clocks, power limits, OpenCL driver, thermal behavior and PrMers version. Treat the following values as rough guidance.

Mersenne Forum discussion:

https://www.mersenneforum.org/node/1086124/page3

### Quick PRP overview, Marin backend

PRP throughput for `p` near `136279841`.

| GPU | User or system | PRMERS_SCORE | Iter/s | Approx PRP ETA | Notes |
|---|---:|---:|---:|---:|---|
| NVIDIA GeForce RTX 5090 | Resolver, vast.ai | n/a | about 2230 | about 17 h | High-end NVIDIA |
| NVIDIA GeForce RTX 4090 | Resolver | 100.00/100 | about 1225 | about 31 h | Reference score |
| NVIDIA GeForce RTX 5070 Laptop | beepthebee | 62.69/100 | about 356 | about 4.5 d | OC reported |
| NVIDIA GeForce RTX 4060 Ti | Lorenzo | 69.14/100 | about 318 | about 5 d | Desktop midrange |
| NVIDIA GeForce RTX 4070 Laptop | Phantomas | 52.24/100 | about 255 | about 6 d | Laptop GPU |
| NVIDIA GeForce RTX 2060 | hwt, Artoria2e5 | 45.76/100 | about 240-259 | about 6 d | Some undervolt or power cap runs |
| NVIDIA GeForce GTX 1660 Ti | Phantomas | n/a | about 234 | about 6.8 d | Older Turing GPU |
| AMD Radeon VII | cherubrock | 50.57/100 | about 350 | about 4.5 d | Development card |
| Apple M4 Pro | wigglefruit | 30.29/100 | about 164 | about 9.6 d | Apple silicon |
| Apple M2 | cherubrock | n/a | about 25 | about 62 d | MacBook Air 8 GB |

### Detailed examples

| GPU | p = 57885161 | p = 74207281 | p = 82589933 | p = 136279841 |
|---|---:|---:|---:|---:|
| RTX 5090 | about 2350 iter/s | about 2230 iter/s | about 1970 iter/s | about 2230 iter/s |
| Radeon VII | about 510 iter/s | about 436 iter/s | about 402 iter/s | about 350 iter/s |
| RTX 4090 | about 1030 iter/s | about 910 iter/s | about 840 iter/s | about 1225 iter/s |
| RTX 4060 Ti | about 420 iter/s | about 366 iter/s | about 337 iter/s | about 318 iter/s |
| RTX 4070 Laptop | about 370 iter/s | about 320 iter/s | about 283 iter/s | about 255 iter/s |
| GTX 1660 Ti | about 330 iter/s | about 288 iter/s | about 262 iter/s | about 234 iter/s |
| RTX 5070 Laptop | about 858 iter/s | about 882 iter/s | about 875 iter/s | about 356 iter/s |
| Apple M4 Pro | about 264 iter/s | about 231 iter/s | about 213 iter/s | about 164 iter/s |
| Apple M2 | about 42 iter/s | about 38 iter/s | about 32 iter/s | about 25 iter/s |

## Backend and code

- PrMers and its integrated algorithms
  - https://github.com/cherubrock-seb/PrMers
  - developed by cherubrock-seb
- Marin register engine and integer IBDWT backend
  - https://github.com/galloty/marin
  - written by Yves Gallot
  - MIT licensed
- Aevum register engine
  - intended repository: https://github.com/cherubrock-seb/aevum-engine
  - modified derivative of GPUOwl/PRPLL
  - exposes `GF(M31^2) x GF(M61^2)` arithmetic through a shared C API and a PrMers `engine::Reg` adapter
- GPUOwl/PRPLL lineage
  - original GPUOwl: https://github.com/preda/gpuowl by Mihai Preda
  - imported fork: https://github.com/gwoltman/gpuowl by George Woltman
  - imported commit: `294cc485ac8cf53c8b69144a3039832eda573849`
- Gerbicz-Li proof scheme
  - used for PRP error checking and safe long exponentiation workflows

## Related inspiration

- GPUOwl by Mihai Preda
  - https://github.com/preda/gpuowl
- GPUOwl/PRPLL fork and NTT work by George Woltman
  - https://github.com/gwoltman/gpuowl
- Genefer22 by Yves Gallot
  - https://github.com/galloty/genefer22
- Marin by Yves Gallot
  - https://github.com/galloty/marin
- GIMPS and the Mersenne Forum community
  - https://www.mersenne.org/
  - https://www.mersenneforum.org/
- GMP-ECM
  - https://gitlab.inria.fr/zimmerma/ecm
- Yves Gallot repositories
  - https://github.com/galloty
  - https://github.com/galloty/f12ecm
  - https://github.com/galloty/FastMultiplication
- Nick Craig-Wood work
  - IOCCC 2012 entry: https://github.com/ncw/ioccc2012
  - GitHub: https://github.com/ncw/
  - ARM Prime Math: https://www.craig-wood.com/nick/armprime/math/

## Must read papers

### Multiplication by FFT and weighted transforms

Discrete Weighted Transforms and Large Integer Arithmetic  
Richard Crandall and Barry Fagin, 1994  
https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/S0025-5718-1994-1185244-1.pdf

Rapid Multiplication Modulo the Sum And Difference of Highly Composite Numbers  
Colin Percival, 2002  
https://www.daemonology.net/papers/fft.pdf

### P-1 factoring

An FFT Extension to the P-1 Factoring Algorithm  
Peter L. Montgomery and Robert D. Silverman, 1990  
https://www.ams.org/journals/mcom/1990-54-190/S0025-5718-1990-1011444-3/S0025-5718-1990-1011444-3.pdf

Improved Stage 2 to P+/-1 Factoring Algorithms  
Peter L. Montgomery and Alexander Kruppa, 2008  
https://inria.hal.science/inria-00188192v3/document

Prime pairing in algorithms searching for smooth group order  
Pavel Atnashev and George Woltman, 2021  
https://eprint.iacr.org/2021/1462.pdf

### Proof schemes

An Efficient Modular Exponentiation Proof Scheme  
Darren Li and Yves Gallot, 2022-2023  
https://arxiv.org/abs/2209.15623

The paper describes a proof scheme for left-to-right modular exponentiation, generalizing the Gerbicz-Pietrzak approach to arbitrary exponents. It is relevant to long PRP runs and validation of large modular exponentiations.

## Aevum arithmetic backend

PrMers can load Aevum as an in-process arithmetic plugin for PRP, LL, P-1 and ECM:

```bash
./build_with_aevum_engine.sh
./prmers 136279841 -prp -proof 0 -d 0 --noask
```

The command above uses automatic backend selection. Explicit choices are available with `-aevum`, `-engine-marin` and `-aevum-auto`.

Default workload thresholds:

| Workload | Default maximum `Aevum / Marin` transform ratio |
|---|---:|
| PRP / LL | `1.00` |
| P-1 Stage 1 (up to 16 registers) | `0.75` |
| P-1 Stage 2 / multi-register | `1.00` |
| ECM (51 registers) | `0.75` |

The global threshold can be changed with `AEVUM_AUTO_MAX_RATIO`. Workload-specific variables are `AEVUM_AUTO_PRP_MAX_RATIO`, `AEVUM_AUTO_LL_MAX_RATIO`, `AEVUM_AUTO_PM1_STAGE1_MAX_RATIO`, `AEVUM_AUTO_PM1_STAGE2_MAX_RATIO`, `AEVUM_AUTO_ECM_MAX_RATIO` and `AEVUM_AUTO_GENERIC_MAX_RATIO`.

Run the host and routing checks with:

```bash
make test-aevum-host test-aevum-reg test-aevum-auto test-aevum-default test-gui-state test-aevum-source
```

Run the GPU backend matrix with:

```bash
AEVUM_TEST_DEVICE=0 make test-aevum-auto-gpu
```

Aevum is a customized GPLv3 derivative of GPUOwl/PRPLL, adapted by cherubrock-seb into a reusable register engine. Its external interface is modeled after the kind of opaque register operations used by Marin, while the Aevum arithmetic implementation remains derived from GPUOwl/PRPLL. Aevum is not an official upstream release.

The standalone source includes:

- a main Aevum README
- the original upstream README renamed to `README_GPUOWL.md`
- `UPSTREAM.md` with the exact imported commit
- `MODIFICATIONS.md`
- GPU register examples and CPU-verified arithmetic chains

See `third_party/aevum/README.md`, `README_AEVUM_REG.md`, `README_V99_7_AEVUM_AUTO_GUI.md` and `NOTICE.md`.

## License and upstream attribution

The root `LICENSE` is the MIT License and applies to PrMers-authored files unless a file or subdirectory states another license.

The optional engine in `third_party/aevum/` is licensed under GNU GPL version 3 because it is a modified derivative of GPUOwl/PRPLL. Its own `LICENSE`, copyright headers, upstream README and attribution files are preserved in that directory.

Recommended public distribution:

1. publish `cherubrock-seb/aevum-engine` as a fork of `gwoltman/gpuowl`, preserving history and GPLv3
2. keep the main PrMers repository under MIT for PrMers-authored files
3. reference Aevum as an optional submodule or external plugin
4. when distributing a bundle or binary containing Aevum, provide the corresponding Aevum source and satisfy GPLv3 requirements

Whether a particular binary/plugin arrangement is treated as one combined work can depend on facts and jurisdiction. The conservative release approach is to publish complete source for both projects and avoid imposing terms incompatible with GPLv3. This is not legal advice.

## Cleaning and uninstall

```bash
make clean
sudo make uninstall
```

## Contributing and issues

Bug reports, feature requests and pull requests are welcome:

https://github.com/cherubrock-seb/PrMers/issues

When reporting a problem, include:

- OS and GPU
- OpenCL driver version
- Full command line
- Relevant `worktodo.txt` line, if any
- Last lines of terminal output and `prmers.log`

## Author

PrMers is developed by cherubrock-seb, with feedback and contributions from users on GitHub and mersenneforum.org.


### Backend compatibility details

- PRP, LL-safe, LL-unsafe, LL-safe2, normal P-1, three-register P-1 low-memory and ECM all use the same Auto/Aevum/Marin selection policy.
- A forced Aevum request falls back to Marin only when no admissible FFT3161 plan exists, and the log states the reason explicitly.
- The one-register P-1 ultra-low-memory algorithm is the sole Marin-only path because it depends on Marin `fast3`; Auto selects Marin and forced Aevum exits cleanly before GPU allocation.
- PRP/LL checkpoints now include backend and mode metadata. Legacy untagged checkpoints are not loaded into Aevum, and LL-safe, LL-safe2 and LL-unsafe use distinct filenames.
