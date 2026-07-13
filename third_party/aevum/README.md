# Aevum Engine

Aevum is an experimental OpenCL arithmetic engine for modular calculations modulo a Mersenne number `M_p = 2^p - 1`.

It is a modified derivative of GPUOwl/PRPLL. The main change is architectural: the original application-oriented GPU arithmetic has been exposed as a reusable register engine with a small C ABI. The API is designed so programs such as PrMers can adapt it to an interface similar in spirit to Marin's `engine::Reg` abstraction.

Aevum is not an official GPUOwl, PRPLL or Marin release.

## Project idea

The project combines two useful ideas:

- GPUOwl/PRPLL arithmetic and the paired integer NTT path over `GF(M31^2) x GF(M61^2)`
- a register-oriented engine model inspired by Marin, where algorithms manipulate opaque arithmetic registers through operations such as set, copy, square, multiply, add, subtract, import and export

This is not a source-level merge of GPUOwl and Marin. Aevum is derived from GPUOwl/PRPLL. Marin is credited as the design inspiration for the external register-engine shape used by the adapter.

## Authors and upstream projects

- GPUOwl was originally written by Mihai Preda: https://github.com/preda/gpuowl
- the imported upstream is George Woltman's GPUOwl/PRPLL fork: https://github.com/gwoltman/gpuowl
- the imported base commit is `294cc485ac8cf53c8b69144a3039832eda573849`
- Marin and its register engine were written by Yves Gallot: https://github.com/galloty/marin
- the Aevum engine API, register adaptation and PrMers integration were added by cherubrock-seb: https://github.com/cherubrock-seb

The original upstream README is preserved as `README_GPUOWL.md`. Exact provenance is recorded in `UPSTREAM.md`, and Aevum-specific changes are listed in `MODIFICATIONS.md`.

## Arithmetic backend

The reusable engine path selects GPUOwl's paired integer NTT configuration:

```text
GF(M31^2) x GF(M61^2)
```

The public API is declared in:

```text
src/EngineApi.h
```

The shared library is:

```text
build-engine/libaevum_engine.so
```

## Register API

The ABI exposes opaque engines and indexed registers. Important operations include:

```c
aevum_engine_create
aevum_engine_destroy
aevum_engine_set_u32
aevum_engine_set_words
aevum_engine_get_words
aevum_engine_copy
aevum_engine_prepare
aevum_engine_square_mul
aevum_engine_mul
aevum_engine_add
aevum_engine_sub_reg
aevum_engine_sub_u32
aevum_engine_equal
```

`aevum_engine_prepare` stores a transformed multiplicand for repeated multiplication. `aevum_engine_equal` compares two normalized register buffers directly on the GPU and avoids exporting very large residues for routine error checks. Register import and export use little-endian 32-bit words representing an integer modulo `2^p - 1`.

Small factors used by `square_mul` and `mul` are applied in the integer register domain with a scratch register. Residue export canonicalizes the final signed carry modulo `2^p - 1` with bounded word access, including positive and negative cyclic carry cases.

## Build

Ubuntu or Debian dependencies:

```bash
sudo apt update
sudo apt install -y g++ make ocl-icd-opencl-dev opencl-headers
```

Build the shared engine:

```bash
make engine-lib -j"$(nproc)"
```

Build the original command-line application as well:

```bash
make -j"$(nproc)"
```

Build the examples:

```bash
make examples
```

## Tests

Host arithmetic and source audit:

```bash
make test-host
```

Load the shared API and test automatic FFT3161 plan selection:

```bash
make test-engine-api
```

Run register examples on OpenCL device 0:

```bash
AEVUM_TEST_DEVICE=0 make test-examples-gpu
```

Run the focused small-factor GPU test:

```bash
AEVUM_TEST_DEVICE=0 make test-small-factor-gpu
```

## Examples

### FFT plan selection

```bash
./build-examples/fft_plans
```

This prints the selected FFT3161 plan for several exponents without starting a long PRP run.

### Register operations

```bash
./build-examples/register_ops 0 .
```

The sample checks:

- `3^2 * 3 = 27`
- prepared multiplication `7 * 5 * 2 = 70`
- register add and subtract
- register copy
- word import and export

### Verified power chain

```bash
./build-examples/power_chain 0 .
```

The sample executes repeated `square_mul` operations on the GPU and compares the complete exported integer with a CPU `boost::multiprecision::cpp_int` result. It also checks prepared multiplication followed by a small factor.

## Use from PrMers

PrMers loads Aevum as an optional in-process shared library through a small adapter derived from its existing `engine` interface.

## License

Aevum is licensed under GNU GPL version 3 because it is a modified derivative of GPUOwl/PRPLL. See `LICENSE`.

The GPL permits modification and redistribution, provided its conditions are followed. In particular, modified versions must be identified, copyright and license notices must be preserved, and corresponding source must be made available when binaries are conveyed.

This README is a technical packaging recommendation, not legal advice.

## Status

Aevum is experimental. It has produced matching PRP residues and correct P-1 results in PrMers testing, but performance depends strongly on exponent size, transform choice, GPU architecture and workload. Automatic Marin/Aevum selection is therefore recommended in PrMers.

## 0.3.4 build identity and portability update

When Aevum is embedded as `third_party/aevum` inside another Git repository, the build now reports `v0.3.4` instead of accidentally using the parent repository commit hash. When built from the standalone Aevum repository, a matching tag or the Aevum repository commit identity is used.

The same update keeps the portable macOS shared-library path: AppleClang/libc++ uses the real `std::filesystem` declaration, the linker uses `-dynamiclib`, and the API loader tests do not require `libdl` on macOS. No arithmetic API or GPU kernel semantics changed in 0.3.4.
