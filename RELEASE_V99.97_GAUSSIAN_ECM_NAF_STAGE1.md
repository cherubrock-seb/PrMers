# PrMers v99.97 — Gaussian ECM signed-NAF Stage 1

This release adds an **opt-in Gaussian-Mersenne ECM Stage-1 accelerator** while keeping the v99.96 implementation available unchanged as the legacy fallback.

## What changes

- Regular Mersenne ECM is untouched.
- Gaussian P-1, PRP, LL and TF are untouched.
- The existing v99.96 `RunGaussianMersenneFactor.cpp` is compiled under the legacy symbol `runGaussianMersenneECMLegacy()`.
- `-gm-ecm` without `-edwards` therefore keeps the exact v99.96 Montgomery ladder behavior.
- `-gm-ecm -edwards` enables the new Stage-1 path:
  - same Suyama sigma/curve stream as legacy GM-ECM;
  - CPU conversion of that same Montgomery curve/point to twisted Edwards modulo the selected GM/GQ norm;
  - GPU scalar multiplication by `K=lcm(1..B1)` with signed NAF;
  - all GPU polynomial arithmetic remains in the exact `2^(4p)-1` lift;
  - dedicated resumable NAF checkpoints;
  - `-gm-safe` replay support.

The accelerator is intentionally **Stage1-only in v99.97**. If `B2>B1`, or a torsion family is explicitly requested, PrMers falls back to the unchanged v99.96 implementation. This keeps the first performance release narrow and regression-safe before a later optimized Stage 2 / PRAC release.

## Golden real-world regression

The new path is anchored to the real GM factor discovered after the v99.96 seed fix:

- `p = 21403643`
- `B1 = B2 = 2000`
- seed series: `84782075184`
- curve 7 sigma: `3059155915320676093`
- expected Stage-1 factor bundle: `482978801775374901713`
- bundle factors: `3253353737 * 148455667849`

Both factors divide the selected Gaussian-Mersenne norm and `[K]P` is the Edwards identity modulo both factors for this exact Suyama curve. The CPU regression test checks the conversion and this mathematical expectation independently of the GPU implementation.

Direct GPU golden test:

```bash
scripts/test_gm_ecm_naf_golden.sh ./prmers
```

Equivalent direct command:

```bash
./prmers 21403643 \
  -gm-ecm -gm-family GM \
  -b1 2000 -b2 2000 -K 1 \
  -sigma 3059155915320676093 \
  -gm-sieve 0 -edwards -aevum-auto -d 0 \
  -f ./gm-ecm-naf-golden
```

Expected:

```text
>>> Gaussian pair ECM Stage 1 factor: 482978801775374901713
```

## Production selection

Legacy v99.96 path:

```bash
./prmers P -gm-ecm -gm-family GM -b1 50000 -b2 50000 -K 1 ...
```

v99.97 NAF Stage-1 accelerator:

```bash
./prmers P -gm-ecm -gm-family GM -b1 50000 -b2 50000 -K 1 -edwards ...
```

Use identical exponent, seed, device and bounds to benchmark ladder vs NAF on the same Suyama curve.
