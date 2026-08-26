# PrMers v99.98 — Gaussian ECM fused Montgomery Stage 1 + BSGS Stage 2

This release adds an opt-in high-performance Gaussian-Mersenne ECM path.

Version:
`4.20.87-alpha-v99.98-gaussian-ecm-fused-bsgs`

## Isolation

Nothing changes unless `-gm-ecm -bsgs` is selected.

- `-gm-ecm` keeps the v99.96/v99.97 legacy Suyama Montgomery implementation.
- `-gm-ecm -edwards` keeps the v99.97 twisted-Edwards/NAF experiment.
- `-gm-ecm -bsgs` selects the v99.98 optimized Montgomery implementation.
- regular Mersenne `-ecm`, P-1, PRP, LL and TF are untouched.

Safety/replay, explicit torsion, and non-zero GM sieve settings fall back to legacy until separately validated.

## Stage 1

The Suyama curve stream and mathematical ladder are unchanged.

The new path uses existing Aevum/Marin fused engine primitives:
- `addsub_copy`
- `mul_pair_prepared`
- `xdbl_tail_uv`

This reduces GPU launch/transform overhead while preserving the same sigma and `[K]P`.

Golden:
- p = 21403643
- B1 = B2 = 2000
- sigma = 3059155915320676093
- expected factor bundle = 482978801775374901713
- factorization = 3253353737 * 148455667849

## Stage 2

The old GM-ECM Stage 2 multiplied the Stage-1 point by huge products of all primes in each chunk.

v99.98 replaces that in the opt-in path with a real Montgomery differential baby-step/giant-step search.

For each Stage-2 prime q:
`q = kD +/- d`

and if q kills the Stage-1 point Q:
`x([kD]Q) = x([d]Q)`.

In projective coordinates the GPU accumulates:
`X_g * Z_b - Z_g * X_b`

and performs batched GCDs against the selected GM/GQ norm.

Default D:
- 30 for tiny bounds
- 210 for normal production bounds

Environment tuning:
- `PRMERS_GM_ECM_BSGS_D`
- `PRMERS_GM_ECM_BSGS_GCD_BATCH`

Stage-2 golden:
- p = 89
- GM = 2^89 - 2^45 + 1
- factorization includes 1069
- sigma = 6
- B1 = 20
- B2 = 50
- Stage 1: no factor
- Stage-2 prime 43: factor 1069
- with D=30, 43 = 1*30 + 13 and the baby/giant collision is exact modulo 1069.

## Checkpoints

v99.98 uses separate `*_stage1_fused.ckpt` and `*_stage2_bsgs.ckpt` files.
Legacy checkpoints are not consumed by the optimized engine.

## Validation policy

Before production:
1. legacy Stage1 golden
2. optimized Stage1 golden
3. legacy Stage2 golden
4. optimized Stage2 golden
5. same-curve B1=50000 timed A/B benchmark on Radeon VII

Only if the optimized path is both correct and faster should production workers be switched to `-bsgs`.
