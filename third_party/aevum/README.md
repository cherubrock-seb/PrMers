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

## Native PFA plans

The engine also supports Good-Thomas mixed lengths `3 * 2^m` and `9 * 2^m`
for the paired `GF(M31^2) x GF(M61^2)` path. The power-of-two rows keep the
existing half-real transform and the odd axis is handled by small radix-3 or
radix-9 butterflies. The inverse width stage scatters directly into the normal
Aevum carry order.

`pfa:auto` selects radix 3 or radix 9 when the real Aevum stock/PFA
transform ratio reaches the validated gate. `pfa:3` and `pfa:9` remain
available for validation or benchmarking. The same OpenCL path is available on Linux, Windows and macOS;
Apple remains an explicit Aevum opt-in in PrMers.

<a id="native-pfa-automatic-selection-ranges"></a>
### Native PFA automatic selection ranges

The table below is the current default `pfa:auto` policy with the normal
`fftOverdrive = 1.0`. The limits are exponent values `p` for `M_p = 2^p - 1`.
They are checked by the plan-policy test at the exact boundaries.

| Automatic plan | Exponent range `p` | Selected Aevum plan | Transform words | Stock/PFA ratio |
|---|---:|---|---:|---:|
| Radix 3 | 10,627,319–15,724,707 | `pfa3:1:256:3:256:101` | 393,216 | 1.333x |
| Radix 3 | 21,071,135–31,284,264 | `pfa3:1:256:3:512:101` | 786,432 | 1.333x |
| Radix 9 | 41,922,069–46,560,704 | `pfa9:1:256:9:256:202` | 1,179,648 | 1.778x |
| Radix 3 | 46,560,705–62,080,936 | `pfa3:1:512:3:512:101` | 1,572,864 | 1.333x |
| Radix 9 | 83,194,017–92,625,960 | `pfa9:1:256:9:512:202` | 2,359,296 | 1.778x |
| Radix 3 | 92,625,961–123,343,992 | `pfa3:1:512:3:1K:101` | 3,145,728 | 1.333x |
| Radix 9 | 165,507,233–183,789,168 | `pfa9:1:512:9:512:202` | 4,718,592 | 1.778x |
| Radix 3 | 183,789,169–244,737,648 | `pfa3:1:1K:3:1K:101` | 6,291,456 | 1.333x |
| Radix 9 | 328,414,017–365,879,616 | `pfa9:1:512:9:1K:202` | 9,437,184 | 1.778x |
| Radix 3 | 365,879,617–487,210,368 | `pfa3:1:4K:3:512:101` | 12,582,912 | 1.333x |
| Radix 9 | 653,808,129–725,153,152 | `pfa9:1:1K:9:1K:202` | 18,874,368 | 1.778x |
| Radix 3 | 725,153,153–965,612,672 | `pfa3:1:4K:3:1K:101` | 25,165,824 | 1.333x |
| Radix 9 | 1,295,872,129–1,440,869,120 | `pfa9:1:4K:9:512:202` | 37,748,736 | 1.778x |
| Radix 9 | 2,574,967,041–2,862,863,872 | `pfa9:1:4K:9:1K:202` | 75,497,472 | 1.778x |

All other admissible exponent ranges use the normal power-of-two Aevum plan.
The automatic gates are `1.30x` for radix 3 and `1.60x` for radix 9.
The explicit radix options remain available for validation and benchmarking.

### Experimental FFT type 4 + PFA9

Version `v0.3.66-pfa9-force-adaptive-exp11` adds an explicit hybrid plan that
combines the radix-9 Good-Thomas map with the upstream `FFT323161` arithmetic:

```text
FP32 x GF(M31^2) x GF(M61^2)
```

The first plan field is the FFT arithmetic type, so the complete RTX 3080 plan
is:

```text
pfa9:4:512:9:512:202
```

In exp11 this ordinary type-4 request is capacity-adaptive.  At 175M it
resolves to the exact two-plane `pfa9:1:512:9:512:202` execution because
GF31+GF61 still has sufficient range.  Use
`pfa9full:4:512:9:512:202` only to benchmark the true three-plane path.

It is deliberately explicit and does not replace the validated type-1 automatic
policy.  The FP32, GF31 and GF61 planes use the same logical digit gather,
radix-9 middle transform, tail pairing and canonical scatter.  Type 4 PFA is
currently restricted to radix 9, non-in-place transforms and single-wide
two-kernel tails.

Build and run the exact GPU differential check against type-1 PFA9:

```bash
./scripts/test_type4_pfa9_ubuntu.sh 0 175000039 2
```

The test performs square/small-multiply and prepared multiplication through both
`pfa9:1:512:9:512:202` and `pfa9full:4:512:9:512:202`, then compares every exported
32-bit residue word.

Background and development notes:

- https://www.mersenneforum.org/node/1110517/page4
- https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast
- https://github.com/cherubrock-seb/PrMers/tree/main/docs/prmers-bananantt-split

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
