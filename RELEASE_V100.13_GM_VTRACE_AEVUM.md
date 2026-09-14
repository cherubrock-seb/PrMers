# PrMers v100.13 RC2 — GM/GQ V-trace + factoring Aevum gate

Status: **release candidate. Host algebra/source validation is complete. RTX 3080 real-GPU correctness/performance validation is complete; Radeon VII remains the final hardware promotion gate.**

## Goal

Remove the algorithmic bottleneck in Gaussian-Mersenne P-1 Stage 2 and make
factoring backend selection consistent with the PRP/LL Aevum policy.

### Before

GM/GQ P-1 Stage 2 multiplied all primes in `(B1,B2]` into dense big exponents
and computed `H^Q`.  For `B2 ~= 2*B1`, the exponent has approximately the same
bit length as Stage 1, so Stage 2 costs almost another Stage 1.

### RC2 default

Fresh `-gm-pm1` / GQ jobs with `B2>B1` use a denominator-free V-trace in the
exact `2^(4p)-1` lift:

- compute Stage-1 `H`;
- project `h=H mod N`, where `N` is the exact GM/GQ target;
- compute `h^-1 mod N` on CPU and inject that representative into the lift;
- construct `V1=H+h^-1`;
- use `V(n+1)=V1*V(n)-V(n-1)`;
- normalize Stage-2 primes as `q=kD +/- j`;
- one `V(kD)-V(j)` term covers the symmetric pair;
- accumulate terms on GPU and project/GCD only at bounded batch boundaries.

Default `D=630` for normal GMNet-size runs.  Override with
`-pm1-vtrace-d` or `PRMERS_GM_VTRACE_D`.

## Mathematical safety

For GM and GQ the selected target `N` divides `L=2^(4p)-1`.  Projection
`Z/LZ -> Z/NZ` therefore preserves addition and multiplication.  The injected
inverse only has to be an inverse modulo `N`; no inversion in the lifted ring is
required.  The entire GPU trace recurrence is polynomial/denominator-free.

`tests/gaussian_pm1_vtrace_math_test.py` checks both Gaussian branches, multiple
`p mod 8` classes, lift divisibility, projected inverse injection and trace
identities without GPU hardware.

## Non-regression/fallbacks

The v100.12 implementation is retained byte-for-byte as
`runGaussianMersennePM1Legacy()`.

The old product-exponent path is selected automatically for:

- Stage1-only jobs (`B2<=B1`);
- a real legacy Stage-1/Stage-2 checkpoint in the save directory;
- `-gm-safe`;
- `-pm1-vtrace-off`;
- `PRMERS_GM_PM1_PRODUCT_STAGE2=1`;
- unsupported V-trace setup/memory conditions.

This lets every real-GPU A/B comparison use the same source tree.

RC2 does not yet add a compact V-trace resume format.  `CliOptions::resume` is true by default in PrMers and is therefore not used as a fresh-vs-resume discriminator. Interrupted fast V-trace jobs should be rerun with `PRMERS_GM_PM1_PRODUCT_STAGE2=1` (or `-gm-safe`) until compact V-trace resume is implemented.

## Aevum / Marin policy

Ordinary Mersenne V-trace already contains Aevum-specific prepared-multiplicand
cache refresh logic.  RC2 makes the auto-admission policy less artificially
conservative on non-Apple systems:

- P-1 Stage 1: max Aevum/Marin transform ratio `0.75 -> 1.00`;
- P-1 normal Stage 2: already `1.00`, retained;
- P-1 low-memory: `0.75 -> 1.00`;
- ECM: `0.75 -> 1.00`.

Unsupported Aevum FFT plans still fall back to Marin.  `-engine-marin`,
`-aevum`, workload FFT overrides and ratio environment variables keep their
manual precedence.

macOS intentionally retains its existing runtime safety rule: Aevum is still
allowed by default only for PRP/LL there until mixed/prepared P-1 and ECM
arithmetic is validated on real Apple GPU hardware.  macOS CI nevertheless
builds the same source and runs the host/source regression matrix.

## RTX 3080 validation completed

On cherubrock1 / RTX 3080, `p=21000041`, `B1=20000`, `B2=40000`:

- OLD product-exponent median: **72.274 s**;
- NEW Aevum V-trace median: **50.751 s**;
- total speedup: **1.424x**;
- wall reduction: **29.8%**;
- Stage 2 fell from about **31.0 s** to about **5.67 s** (~**5.5x** Stage-2 acceleration).

Positive GMNET oracle on the same RTX 3080:

- `p=20400059`;
- `B1=20000`, `B2=710573`;
- expected factor `305072808512304979901`;
- NEW Aevum V-trace recovered the exact factor in Stage 2;
- result JSON: `outcome=factor`, `stage=2`, exact factor, `backend=Aevum`;
- V-trace Stage 2 completed in **138.54 s** over **45381** unique terms.

## Required real-GPU gate

RTX 3080 is already validated. Run the same branch on Radeon VII before tagging:

```bash
DEVICE=0 TARGET=radeonVII BENCH_MERSENNE=1 BENCH_ECM=1 \
  bash scripts/bench_v10013_factoring.sh
```

The primary GM test uses `p=21000041`, `B1=20000`, `B2=40000` and alternates
`OLD NEW NEW OLD` after warmup to suppress compile/autotune/order bias.

Promotion requirements:

1. NEW and OLD give compatible factor/no-factor outcomes.
2. No Aevum invariant / prepared-multiply / residue mismatch.
3. GM Stage 2 V-trace materially beats product-exponent on both GPUs.
4. Ordinary Mersenne P-1 Aevum is exact versus Marin.
5. ECM Aevum is exact versus Marin; if it is slower on either hardware family,
   restore/tighten only the ECM auto ratio before tagging.
6. Existing `make test-gm`, Aevum host/source/auto and backend compatibility
   suites remain green.
7. Linux and macOS RC CI are green.

Only after these gates should a `v100.13...` tag trigger the existing release
workflows.
