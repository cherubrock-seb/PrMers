## v0.3.99 NVIDIA 1K policy, scoped LDS padding and cumulative correctness fixes

- NVIDIA 1K automatic plans now default to radix-8 when no explicit
  `AEVUM_RADIX1K` override is present. Explicit overrides still win, while
  non-NVIDIA and Apple paths retain their validated radix-4 policy.
- RTX 3080 fixed-plan validation for the NVIDIA 1K radix-8 policy was exact
  and improved sustained PRP throughput by about +4.48% at p=210000017 and
  +4.56% at p=219999919 versus the validated radix-4 fallback.
- Height LDS padding is disabled only for the validated
  `1:1K:8:512:202` geometry. Final paired validation remained exact on RTX
  3080 and Radeon VII; measured median gains were about +0.34% at p210 and
  +0.80% at p219999919 on RTX, and +3.29% at p210 on Radeon VII.
- Cumulative post-v0.3.98 synchronization includes explicit-plan capacity
  checks, tail-trig/config consistency, active-plan API support, GF61/OpenCL
  correctness fixes, and Apple/OpenCL portability and resource-limit fixes.
- The standalone GPU smoke harness now uses the valid `-proof 1` argument.
  The corrected committed smoke passed on RTX 3080 and Radeon VII.
- Exact source SHA `105ff83d2f1c5f65cdbdbdf47243760b80c26d1a`
  passed standalone exact-SHA CI and the Linux/Windows/macOS release-package
  preflight before this release-only version/documentation bump.

## v0.3.98 NVIDIA PFA7 GF61 middle-out normalization

- NVIDIA PFA7 GF61 middle-out now folds the inverse-radix normalization
  into the left operand of the single shared middle twiddle before `cmul`,
  replacing seven per-output scalar normalizations with one prescale.
- Scope is restricted to NVIDIA, PFA7 and GF61 middle-out. The existing
  NVIDIA PFA9 middle-out normalization optimization remains unchanged, and
  Radeon compilation remains byte-identical to the previous stock path.
- PrMers fixed-plan validation: central 12-pair full-PRP geomean +0.0213%
  with 10/12 wins; low/high range aggregate +0.0255% with 10/12 wins.
- RTX JIT resources improved from 72 to 70 registers and from 1381 to 1225
  PTX instructions, with zero local-memory spill and unchanged 7-block/SM
  occupancy.
- Standalone Aevum validation passed host/API/CLI gates on RTX and API gates
  on Radeon VII; PFA7 exact 10000 and PFA9 regression exact 5000 passed on
  both GPUs.
- The exact source-only candidate passed the standalone preflight CI matrix
  on Linux, Windows and macOS (11/11 jobs) before fast-forward promotion to
  main. The promoted source SHA then passed the same main CI matrix.

## v0.3.97 NVIDIA PFA9 GF61 middle-out normalization

- NVIDIA PFA9 GF61 middle-out now folds the inverse-radix normalization into
  the single shared middle twiddle instead of applying nine per-output scalar
  normalizations.
- Scope is deliberately restricted to NVIDIA, PFA9 and GF61 middle-out.
  PFA3 and PFA7 retain their previous arithmetic, and Radeon compilation
  retains the stock path.
- PrMers validation: central fixed-plan confirmation 20/20 wins at about
  +0.059%, range validation 12/12 wins at about +0.058%.
- RTX JIT resources improved from 70 to 66 registers and from 1498 to 1290
  PTX instructions, with zero local-memory spill and unchanged 7-block/SM
  occupancy.
- Standalone Aevum validation passed host/API gates on RTX and Radeon,
  PFA9 exact 10000 on both GPUs, PFA7 stock-path exact 5000 on both GPUs,
  and standalone RTX PFA9 confirmation 4/4 positive.

## Word-exact three-plane FFT323161 PFA9

- The FP32 plane of the `pfa9full:4:...` / non-elided `pfa9:4:...` plan now
  computes the convolution on the radix-9 Good-Thomas axis.  Its complex 9th
  root makes rows k and 9-k conjugate partners, so `tailSquare`, `tailMul` and
  `tailSquareZero` pair bin l of row k with bin -l of row 9-k (the GF31/GF61
  planes keep in-row pairing: their odd roots lie in the base field).  The
  odd-axis inverse is the forward `fft9` on the component-swapped tail output
  plus 1/9, the pair twiddle is indexed by `PFA_RADIX * binary_line`, and the
  middle kernels apply the WIDTH x SMALL_HEIGHT Cooley-Tukey twiddle
  (`pfaMiddleTwiddle(F2)`, table from `genMiddleTrigFP32Pfa`) that the binary
  axis still needs.
- Validated word-for-word against GMP and against power-of-two FFT323161 /
  FFT3161 plans with dense random residues at 42.4 bits/word (256:9:256 and
  512:9:512).  `tests/type4_pfa9_engine_compare.cpp` now feeds seeded dense
  residues instead of squaring 3, which never let the FP32 plane reach the CRT.

## v0.3.68 throughput-cost auto selection + experimental PFA9 lead bridge

- adds `throughput:auto` and `pow2:auto` candidate scoring;
- calibrates M175 RTX 3080 default selection to `4:512:8:512:202`;
- adds the opt-in `AEVUM_PFA_LEAD_BRIDGE=1` path;
- adds exact bridge/canonical GPU comparison and OpenCL/source audits.

## v0.3.67 power-of-two FFT323161 + register lead cache

- enables explicit power-of-two FFT323161 plans such as `4:512:8:512:202`;
- restores the upstream PRPLL `LEAD_WIDTH`/`carryFused` chain across register API squarings;
- uses a one-pending-square scheduler so all observable register operations remain canonical;
- enables type-4 multi-queue overlap for both power-of-two and PFA9 plans;
- keeps PFA and Apple on their validated canonical paths;
- adds host source/model coverage and a cached-versus-canonical GPU differential test.

## v0.3.66 force-adaptive FFT323161 PFA9

- Ordinary `pfa9:4:...` requests are now capacity-adaptive.  When the exact
  GF31+GF61 PFA plan is safe, Aevum executes the same shape and variant without
  the redundant FP32 plane.
- `pfa9full:4:...` is the explicit diagnostic spelling for the real
  FP32+GF31+GF61 three-plane path.
- Automatic radix-9 selection now uses the measured faster `202` variant;
  radix-3 remains on `101`.
- The normal automatic policy still never selects FFT type 4.
- Host CI now verifies explicit type-4 elision, full-path preservation and the
  radix-9 `202` policy.

## v0.3.65 optimized adaptive FFT323161 PFA9

- Added `pfa9fast:4:...` as the first adaptive alias.
- Added safe two-queue overlap for the true full type-4 path.

## v0.3.64 native PFA9 FFT323161 experiment

- Added the word-exact FP32 + GF31 + GF61 Good-Thomas radix-9 path.
