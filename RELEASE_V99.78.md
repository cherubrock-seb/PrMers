# PrMers v99.78 stable backend/policy candidate

This candidate deliberately excludes the experimental PFA3-M19 work.

## Correctness and control-flow fixes

- macOS classic P-1 BSGS uses a flat-slab auto-D correctness guard. V-trace and all non-Apple classic paths are unchanged.
- ECM stops on the first newly discovered factor by default. `-ecm-continue-after-factor` keeps running later curves.
- P-1 skips requested Stage 2 after a newly discovered Stage-1 factor by default. `-pm1-continue-stage2-after-factor` overrides this.
- Result JSON for Aevum P-1 stages reports the real Aevum transform size rather than the Marin planning size.

## Apple Aevum fixes and policy

- An evicted prepared multiplicand is rebuilt from its canonical register instead of entering the unsupported generic Apple tailMul path.
- GF61 middle-out always uses the five-stage Apple pipeline for FFT3161, including the ECM/P-1 `MIDDLE=4` plan that previously failed with `INVALID_KERNEL`.
- macOS keeps Marin as the default backend.
- Forced Aevum PRP/LL without an explicit plan uses staged stock Type1 FFT3161 (`pow2:auto`).
- Explicit Type4 or native-PFA plans are rejected cleanly on Apple OpenCL 1.2 before kernel creation. They are not silently downgraded.

## Backend-plan visibility

Automatic backend messages now include a plan family (`Type1 FFT3161`, `Type4 FFT323161`, `PFA3`, or `PFA9`). The Aevum adapter also prints its requested plan and actual transform size.

PRP/LL retain the throughput selector on non-Apple platforms. ECM and P-1 retain the conservative Type1 selector because their mixed-operation and high-register workloads have not yet received a bit-exact Type4 performance qualification.

## RTX 5090 / GB202 report

The contributor's measured profile is documented in `third_party/aevum/README_RTX5090_GB202_TUNE.md`, but is not installed as a universal default without a complete device-scoped tune file and boundary validation.
