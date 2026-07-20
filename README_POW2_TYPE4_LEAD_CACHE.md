# PrMers v99.74 — power-of-two FFT323161 + register lead cache

This release enables the exact upstream-style power-of-two plan:

```text
4:512:8:512:202
```

for `M175000039`.  It is the same 4M-word FFT323161 family used by PRPLL:
FP32 + GF31 + GF61, variant 202.  Previous PrMers/Aevum builds rejected this
non-PFA type-4 plan and therefore compared an 8M FFT3161 plan against PRPLL's
4M FFT323161 plan.

The register adapter now also preserves upstream PRPLL's hot squaring chain.
One logical `square_mul(reg,1)` is held pending; consecutive calls execute the
previous square with `LEAD_WIDTH`, and a read/copy/checkpoint/sync emits the
final square with a canonical boundary.  This restores `carryFused` between
ordinary PRP iterations without changing the public engine API.

Safety rules:

- enabled only for non-PFA, short-carry plans;
- disabled on Apple, retaining the validated canonical path;
- PFA3/PFA9 remain canonical because native-PFA fused carry is not implemented;
- `AEVUM_REG_LEAD_CACHE=0` restores the previous per-iteration canonical path;
- `AEVUM_TYPE4_MULTI_Q=0` disables type-4 plane overlap for A/B measurement.

Recommended RTX 3080 test:

```bash
./prmers 175000039 -d 1 \
  -aevum-fft 4:512:8:512:202 \
  -proof 0 -f bench-pow2-type4
```

Expected startup lines include:

```text
FFT: 4M 4:512:8:512:202 (41.72 bpw)
Aevum register lead cache enabled
```

Use `AEVUM_REG_LEAD_CACHE=0` with the same command for the exact canonical
baseline.  The package test script also performs a word-exact GPU differential
between the cached and canonical paths before benchmarking.
