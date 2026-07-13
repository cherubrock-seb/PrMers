# Aevum engine backend v99.6

Aevum is an in-process `engine::Reg` backend for PrMers. It uses the paired integer transform path `GF(M31^2) x GF(M61^2)` and can replace Marin in the existing PRP, LL, P-1 and ECM code.

Automatic Marin/Aevum selection is the default. Use `-aevum` to force Aevum, `-engine-marin` to force Marin, and `-aevum-auto` to request automatic selection explicitly.

## Build on Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential g++ make ocl-icd-opencl-dev opencl-headers libgmp-dev

./build_with_aevum_engine.sh
make test-aevum-host
make test-aevum-reg
```

The build creates:

```text
./prmers
./third_party/aevum/build-engine/libaevum_engine.so
```

The adapter loads the shared library in the same process. It does not launch a second executable.

## Commands

Automatic selection:

```bash
./prmers 136279841 -prp -proof 0 -d 0 --noask
```

Forced Aevum PRP:

```bash
./prmers 1362763 -prp -proof 0 -aevum -d 0 --noask -f ./run-aevum-prp
```

Lucas-Lehmer:

```bash
./prmers 1362763 -llunsafe -aevum -d 0 --noask -f ./run-aevum-ll
```

P-1:

```bash
./prmers 1362763 -pm1 -b1 29 -b2 6910159 -aevum -d 0 --noask -f ./run-aevum-pm1
```

ECM:

```bash
./prmers 9815459 -ecm -b1 10000 -b2 1000000 -K 1 -aevum -d 0 --noask -f ./run-aevum-ecm
```

Force an FFT3161 shape:

```bash
./prmers 1362763 -prp -proof 0 -aevum -aevum-fft 1:256:2:256 -d 0 --noask
```

The selected transform must be type `1`.

## Backend behavior

The factory behind `engine::create_gpu()` selects Marin or Aevum. The algorithms continue to use the same register interface:

```text
set
copy
square_mul
set_multiplicand
mul
add
sub
sub_reg
get_data and set_data
checkpoint save and restore
set_mpz and get_mpz
```

`set_multiplicand()` prepares the Aevum transform in advance. Consecutive multiplications reuse it while it remains valid. Register values stay in compact Mersenne residue form. Stage 1 checkpoints carry a backend sidecar marker so Marin register images are never loaded into Aevum buffers, or conversely. Legacy unmarked checkpoints are treated as Marin and ignored by forced Aevum runs.

Small factors used by P-1 and PRP are applied with a normal modular multiplication by a prepared Aevum constant. The unsafe raw `regScale` word kernel has been removed.

When Aevum is selected, PrMers does not allocate its separate internal NTT buffers. PRP proof checkpoints use the Marin-compatible host proof path, so proof generation does not require the internal PrMers NTT allocation.

## Transform selection and paths

Without `-aevum-fft`, the engine selects a valid FFT3161 plan directly from the NTT shape table. The generic GPUOwl `tune.txt` is not used for this automatic selection because it contains FP64 shapes with non-power-of-two middle dimensions.

Typical automatic plans are:

```text
216091       -> Marin fallback, below native FFT3161 range
1362763      -> 1:256:2:256:101
136279841    -> 1:1K:8:256:101
2147483647   -> 1:4K:16:512:101
```

Overrides:

```bash
export AEVUM_ENGINE_LIB=/path/to/libaevum_engine.so
export AEVUM_PREPARED_CACHE=2
```

`AEVUM_PREPARED_CACHE` controls the number of transformed multiplicands retained by the adapter. The default is two, capped by the register count. A value from 4 to 6 can help ECM or normal-memory P-1 when VRAM permits; `0` disables the transformed cache without changing results.

System installation:

```bash
sudo make install
```

This installs:

```text
/usr/local/bin/prmers
/usr/local/lib/prmers/libaevum_engine.so
/usr/local/share/prmers/aevum/tune.txt
```

## Comparison

```bash
./scripts/benchmark_aevum_vs_marin.sh 1362763 0
```

The script runs the same proof-disabled PRP once with Marin and once with Aevum. Kernel compilation on the first Aevum run is cached in `.aevum-kernel-cache`, so repeat the Aevum run before judging steady-state speed.

## License and upstream

Aevum is a customized GNU GPLv3 derivative of PRPLL/GPUOwl. It is not an official upstream release. See `third_party/aevum/UPSTREAM.md`, `third_party/aevum/MODIFICATIONS.md` and `NOTICE.md`.

## Standalone repository

The reusable engine is intended for publication at https://github.com/cherubrock-seb/aevum-engine as a GPLv3 fork-derived project with GPUOwl history preserved.


## v99.8 compatibility routing

LL-safe, LL-unsafe, LL-safe2, normal P-1, three-register P-1 low-memory, and ECM all retain automatic/forced Aevum support. The one-register P-1 ultra-low-memory fast3 algorithm remains Marin-only by design. Use `tests/run_backend_validation_matrix.sh` to validate every routing combination and produce a shareable report archive. The `full` profile uses deterministic ECM seeds and complete medium-size Aevum/Marin pairs; `PRMERS_MATRIX_CASE_FILTER` can restrict a rerun to one family of failures.
