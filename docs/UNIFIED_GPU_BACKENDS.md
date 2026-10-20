# Unified Aevum GPU backends (CUDA / OpenCL)

PrMers always retains its existing **Marin ↔ Aevum** backend policy. When Aevum is selected and its OpenCL and native CUDA plugins are present, PrMers can compare both implementations on the selected physical NVIDIA GPU.

## Automatic selection

1. Determine the physical OpenCL device selected with `-d N`. On NVIDIA, use the driver PCI bus/slot identity to map it to the corresponding CUDA ordinal. With only one exposed NVIDIA device, CUDA ordinal zero is a fallback if PCI data are absent. If mapping is ambiguous, continue with OpenCL.
2. Build workload-appropriate Aevum engine instances for OpenCL and CUDA. **Aevum's existing native FFT policy** chooses each engine's legal FFT plan (including explicit `-aevum-fft`, tune.txt, device policy and performance tuning). There is no fixed FFT plan imposed by the CUDA/OpenCL selector.
3. Run a bounded probe and compare complete, canonical residues. PRP probes square and factor-3 arithmetic; LL includes square-minus-two; P-1 and ECM use prepared multiplication (ECM also includes add/sub). Generic workloads use a square probe. The probe is a *proxy* for production performance, not a complete benchmark of each algorithm.
4. Select CUDA only if the tested canonical residues match and the measured throughput improves by **more than 3%**. Otherwise retain OpenCL. Missing CUDA runtime/NVRTC/plugin, errors, or unsupported multi-GPU mapping are *non-fatal*. Cache the decision for the same exponent, workload, GPU and FFT request within the running process.

**Exceptions:** The legacy P-1 ultra-low-memory engine requires Marin `fast3` and is not compatible with Aevum. On macOS, Aevum currently supports PRP/LL stock FFT3161 via OpenCL; P-1/ECM use Marin by existing Apple safety policy. Both restrictions remain in effect.

The selector does **not** compare Aevum versus Marin universally; that is handled by the existing automatic engine policy.

## Environment

| Setting | Effect |
|---|---|
| Unset / `AEVUM_BACKEND=auto` | Benchmark available CUDA and OpenCL and select the safe winner |
| `AEVUM_BACKEND=opencl` | Bypass the CUDA probe and use OpenCL |
| `AEVUM_BACKEND=cuda` | Diagnostic preference for CUDA, still requiring an identical residue and safe GPU mapping |
| `AEVUM_ENGINE_LIB=/path/to/plugin` | Explicit plugin path: takes precedence over automatic selection |
| `AEVUM_TUNE_DIR=/path/to/tune` | Existing Aevum tune-file path |

The PrMers log displays both the active FFT `plan=` and `backend=CUDA` or `backend=OpenCL`. The initial selection probe reports workload, physical device and microseconds per operation.

## Compiling from sources

Run `make` for the OpenCL default. CUDA is **never** required for this build. To build both engines from the same source tree, install CUDA Toolkit headers/driver stubs/NVRTC and run:

```sh
make
make aevum-dual
make install PREFIX=/usr/local
```

The dual build produces `libaevum_engine.so` and `libaevum_engine_cuda.so` (Linux) or `aevum_engine_cuda.dll` (Windows). The Makefile performs separate clean builds so CUDA object files do not contaminate the OpenCL library. On Windows, build with MSYS2/UCRT64 and a compatible CUDA SDK; set `AEVUM_CUDA_ROOT` if needed. On macOS, only the OpenCL engine is compiled/installed.

## Binaries

Linux and Windows unified release archives ship the ordinary OpenCL engine and optional CUDA sibling in one package. macOS arm64, x86_64 and universal2 ship OpenCL-only packages from the **same PrMers release commit**.

Extract the package intact; `-aevum` forces the Aevum engine while keeping its FFT plan automatic. Without `-aevum`, the ordinary Marin/Aevum backend selection policy still applies.

## Kaggle T4 ×2 smoke validation

`notebooks/kaggle_dual_backend_validation.ipynb` and `scripts/kaggle_dual_backend_smoke.py` install native dependencies, build both engines on Kaggle for glibc compatibility, then exercise AUTO and forced OpenCL on GPU indices 0 and 1 across PRP, LL, P-1, ECM and Gaussian P-1.

These validation files contain **no database credentials or production campaign tasks**. They must not be confused with complete end-to-end mathematical correctness certification, multi-day benchmark results or successful operation of any external database.

## Release validation

CI builds/link-checks and packages Linux, Windows, macOS Intel, macOS arm64 and macOS universal2. GPU-less runners do not validate hardware-runtime correctness. Real CUDA/OpenCL runs on independent NVIDIA hardware and AMD/Intel OpenCL regression checks are necessary before a broad rollout.
