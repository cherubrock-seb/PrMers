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

When Aevum is embedded as `third_party/aevum` inside another Git repository, the build now reports `v0.3.6` instead of accidentally using the parent repository commit hash. When built from the standalone Aevum repository, a matching tag or the Aevum repository commit identity is used.

The same update keeps the portable macOS shared-library path: AppleClang/libc++ uses the real `std::filesystem` declaration, the linker uses `-dynamiclib`, and the API loader tests do not require `libdl` on macOS. No arithmetic API or GPU kernel semantics changed in 0.3.4.


## v0.3.6 Apple OpenCL compatibility

Kernel compilation now selects OpenCL C 1.2 for devices reporting OpenCL 1.x instead of unconditionally passing `-cl-std=CL2.0`. This addresses an immediate abort seen on Apple M-series OpenCL 1.2 devices when the register API compiled its first arithmetic/read kernel.


## Portable macOS build

The build defaults to `MACOSX_DEPLOYMENT_TARGET=12.0`. Override it only when intentionally requiring a newer macOS.

## Apple GF61 middle-in staging (v0.3.17)

The macOS OpenCL 1.2 path splits only `fftMiddleInGF61` into arithmetic stages
plus the original LDS transpose. Non-Apple paths remain upstream-compatible.

## Apple GF61 scalar middle-in front end (v0.3.18)

The Apple OpenCL 1.2 path uses scalar load, factor-generation and apply kernels
before the stock middle FFT and LDS transpose. This avoids a rejected Metal
pipeline without changing non-Apple execution paths or adding a transform-sized
buffer.

## Apple GF61 tailSquare two-kernel LDS (v0.3.19)

On Apple FFT3161, `TAIL_KERNELS=3` is forced so the two exceptional tail lines
are handled by `tailSquareZeroGF61`.  The main tail kernel remains double-wide
and both kernels retain the upstream LDS FFT/reverse implementation.  Other
platforms retain their requested/default tail policy.


## Apple GF61 special-tail scalar staged LDS (v0.3.20)

Apple FFT3161 keeps upstream `TAIL_KERNELS=3`.  The two exceptional lines are
now staged through a small dedicated scratch buffer so Metal creates six
compact pipelines instead of the monolithic `tailSquareZeroGF61`.  Both height
FFTs and both reverse operations retain their LDS implementation, and the main
double-wide `tailSquareGF61` kernel is unchanged.  Non-Apple dispatch is
unchanged.


## Apple GF61 special-tail radix-stage FFT (v0.3.21)
See `README_V0.3.21_APPLE_GF61_TAILZERO_RADIX_STAGED_LDS.md`.


## Apple GF61 special-tail global-shuffle staging (v0.3.22)
See `README_V0.3.22_APPLE_GF61_TAILZERO_GLOBAL_SHUFFLE_STAGED.md`.

## Apple GF61 special-tail global reverse staging (v0.3.23)

On Apple OpenCL 1.2, the two exceptional GF61 tail lines now perform their
second-half reverse as an exact global-memory permutation between the existing
tiny scratch banks.  This removes only the Metal-rejected LDS reverse pipeline;
the main double-wide tail and the rest of Aevum retain their upstream LDS
algorithms.


## Apple GF61 special-tail PairApple argument fix (v0.3.24)

The staged Apple `tailSquareZeroGF61PairApple` dispatch now binds all three kernel arguments explicitly. The previous fixed middle argument was overwritten by the dynamic bank argument. Kernel source and non-Apple paths are unchanged.


## v0.3.25 Apple GF61 special-tail scalar final write

The Apple path now writes each special-tail GF61 value with one scalar work-item. The output address is exactly the stock writeTailFusedLine mapping. No new allocation is introduced and all non-Apple paths retain the upstream vector writer.


## v0.3.26 Apple GF61 special-tail direct final copy

The final special-line writer is now a branch-free direct GF61 copy. The host launches one line at a time and supplies resolved scratch/output bases, removing transPos, T2/GF61 casts and helper overloads from the Apple Metal pipeline.


## v0.3.27 Apple GF61 normal-line main-tail global staging

Apple Metal accepts the complete exceptional-line path but rejects the stock double-wide `tailSquareGF61` compute pipeline. The Apple-only normal-line path now reuses the existing output and old input GF61 planes as ping-pong banks for scalar load, staged height FFT, cross-line reverse, scalar pair-square, reverse, and the second height FFT. No new transform-sized allocation is introduced, and the stock LDS kernel remains the non-Apple path.


## v0.3.28 Apple GF61 fftW global staging

The M2 smoke for v0.3.27 accepted the complete special and normal-line GF61 tail pipelines and `fftMiddleOutGF61`, then rejected only the stock monolithic `fftWGF61` pipeline. On Apple FFT3161, v0.3.28 scalar-loads the exact `readCarryFusedLine` layout and reuses the caller output and consumed middle-out input buffers as global width-FFT ping-pong banks. The stock LDS `fftWGF61` remains unchanged for non-Apple platforms and no transform-sized buffer is added.


## v0.3.29 Apple GF61 tailMul global staging

The M2 v0.3.28 smoke passed LL-UNSAFE and loaded every staged square/fftW kernel. LL-SAFE reached prepared multiplication and rejected the stock `tailMulGF61` pipeline. v0.3.29 stages the full all-line GF61 multiplication using the two consumed transform buffers plus one GF61-only scratch plane, preserving the prepared multiplicand. It also makes the one-word GPU checksum read synchronous on Apple after an explicit queue finish, eliminating a transient false read mismatch. Linux and Windows retain the original monolithic tailMul kernel.

## v0.3.30 Apple deterministic checked reads

The v0.3.29 M2 smoke showed that an explicit queue finish did not make the atomic `sum64` checksum reliable on Apple OpenCL. Apple `readChecked` now verifies host transfers using two independent synchronous full reads compared word-for-word. Non-Apple platforms retain the original GPU checksum path. No arithmetic kernel is changed.


## v0.3.31 Apple queue-marker flush
Adds `clFlush` before polling queue markers; no arithmetic changes.

## v0.3.32 Apple-only queue-marker flush

The periodic marker `clFlush` is now compiled only when `__APPLE__` is defined. Non-Apple OpenCL/CUDA builds retain their original queue submission behavior.

## v0.3.33 Apple MIDDLE=8 and stock-reverse GF61 tailMul

Fixes the large Apple `MIDDLE=8` GF31 compilation path and changes the Apple
staged GF61 generic multiply to the exact upstream `reverse -> pairMul ->
reverse` ordering. The complete large-plan production kernel matrix is parsed
as OpenCL C 1.2. Non-Apple paths are unchanged.


## v0.3.34 Apple GF61 middle-out and exact local-pair tailMul

Large Apple plans stage `fftMiddleOutGF61`; Apple generic multiplication uses the upstream local-memory reverse and pair routines. No non-Apple path changes.

## v0.3.35 Apple scalar GF61 pair multiply and middle-out

Replaces the rejected local-memory generic multiply with scalar special/normal pair kernels and splits large-plan middle-out into scalar load/multiply/write stages plus one isolated middle FFT. No non-Apple path changes.

## v0.3.36 — Apple GF61 special-line trig fix

Corrects the H/2 special-line trig in the staged Apple generic multiply.

## v0.3.37 — Apple generic multiplication safety

Apple generic register multiplication now fails closed pending arithmetic
validation.  Native `square_mul(reg, factor)` remains available and avoids the
generic tail multiplication pipeline.  Diagnostic override:
`AEVUM_APPLE_UNSAFE_GENERIC_MUL=1`.


## v0.3.47 — Apple GF61 middle-in restrict alias fix

The Apple FFT3161 middle-in apply and post-multiply stages now use one in-place
data pointer instead of aliased `restrict` input/output pointers. This fixes an
undefined kernel contract exposed by the standalone `5^2` arithmetic probe.
Non-Apple paths are unchanged.


## v0.3.53 — Apple nonblocking queue pacing

Apple OpenCL now submits each periodic queue batch with nonblocking `clFlush`
instead of enqueueing a marker and polling it to completion. Blocking reads,
explicit `finish()` calls and cross-queue synchronization remain completion
barriers. Set `AEVUM_APPLE_QUEUE_MARKER_WAIT=1` to restore the previous marker
policy for A/B diagnostics. Non-Apple queue behavior is unchanged.
