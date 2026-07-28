# PrMers v4.20.77-alpha-v99.88 — Gaussian-Mersenne P-1/ECM lift

## Added

- `-gm-pm1`: exact P-1 Stage 1 and low-memory product-exponent Stage 2 for
  `G_p = 2^p - (2/p)2^((p+1)/2) + 1`.
- Structural `4p` multiplier in the Stage 1 exponent.
- `-gm-ecm`: Suyama Montgomery ECM with GPU projective x/z ladders.
- CPU projection, GCD and inversion strictly modulo `G_p` at stage/chunk
  boundaries.
- `-gm-factor-chunk-bits` for P-1 and ECM Stage 2 batching.
- Atomic CRC32 Stage 1 and Stage 2 checkpoints.
- `-gm-safe` independent P-1 block/chunk replay and ECM Stage 1/chunk replay.
- Deterministic mathematical tests and four small GPU validation cases.

## Isolation

The new implementation is contained in
`src/modes/RunGaussianMersenneFactor.cpp` plus CLI/dispatch declarations.  No
file under `third_party/aevum`, `kernels/marin.cl` or `kernels/prmers.cl` was
modified relative to the working v99.87 Gaussian-Mersenne bundle.

## Deliberate limitation

Stage 2 is a safe product-exponent continuation rather than the ordinary
Mersenne V-trace/BSGS implementation.  Those implementations assume arithmetic
and inversions directly modulo the tested Mersenne number; applying them
unchanged to `2^(4p)-1` could be obstructed by the unrelated lift cofactor.
