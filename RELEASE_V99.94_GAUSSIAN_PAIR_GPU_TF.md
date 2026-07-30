# PrMers v99.94 — Gaussian pair GPU trial factoring

PrMers version:

    4.20.83-alpha-v99.94-gaussian-pair-gpu-tf

Aevum remains unchanged:

    v0.3.78-workload-plan-policy-audit-fix

## New GPU TF mode

PrMers now accepts direct Gaussian pair trial factoring:

    ./prmers P -gm-tf FROM_BITS TO_BITS -gm-family BOTH -d DEVICE

and native worktodo entries:

    GMTF=P,FROM_BITS,TO_BITS,BOTH,4194304,65536

Families may be GM, GQ, or BOTH.

For odd prime p, one modular exponentiation computes t=2^((p+1)/2) mod q.
PrMers derives 2^p from t, then tests both complementary residues:

    GM(p)   = 2^p - (2/p) 2^((p+1)/2) + 1
    5 GQ(p) = 2^p + (2/p) 2^((p+1)/2) + 1

Thus GM and GQ classification share the expensive exponentiation.

## Fast path

The new path runs before the large PrMers NTT/FFT application is built. It uses:

- q = 4kp + 1 candidate generation;
- a host segmented small-prime sieve;
- an OpenCL 64-bit Montgomery kernel;
- one exponentiation per surviving q for both GM and GQ;
- checkpoints at every k chunk;
- schema v2 JSON with zero, one, or several classified factors.

## Compatibility

All existing Gaussian P-1, ECM, PRP, Proth and ordinary Mersenne modes remain
unchanged. Existing result schema v1 remains valid on GMNet.
