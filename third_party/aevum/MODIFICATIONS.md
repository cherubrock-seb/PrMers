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
