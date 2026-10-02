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
  512:9:512).

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
