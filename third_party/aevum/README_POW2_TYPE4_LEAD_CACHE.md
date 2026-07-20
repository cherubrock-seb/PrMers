# Aevum v0.3.67 — power-of-two FFT323161 + register lead cache

Aevum now accepts upstream power-of-two FFT323161 plans in addition to the
existing FFT3161 and PFA9 modes.  The primary RTX 3080 comparison plan is:

```text
4:512:8:512:202
```

This is a 4,194,304-word FP32 + GF31 + GF61 transform.  Previous Aevum engine
builds rejected non-PFA type 4, causing PrMers `-pfa-off` to compare an 8M
FFT3161 transform with PRPLL's 4M FFT323161 transform.

## PRPLL-style lead caching

The engine register API receives one `square_mul` call per PrMers iteration.
The old adapter forced every call through `LEAD_NONE -> LEAD_NONE`, paying
`fftW + carryA + carryB + fftP` at each register boundary.  PRPLL instead uses
`LEAD_WIDTH` and `carryFused` between consecutive squarings.

The adapter now keeps one logical square pending.  On the next square it emits
the previous one with `leadOut=LEAD_WIDTH`; before read/copy/checkpoint/sync it
emits the final pending square with `leadOut=LEAD_NONE`.  The public API and all
externally visible register values remain canonical.

The optimization is enabled only for non-PFA short-carry plans and is disabled
on Apple.  Native PFA remains unchanged until a validated PFA carryFused path
exists.

Environment controls:

```text
AEVUM_REG_LEAD_CACHE=0   canonical register boundary after every square
AEVUM_TYPE4_MULTI_Q=0    disable FP32+GF31 / GF61 queue overlap
```

GPU differential test:

```bash
./scripts/test_pow2_type4_lead_cache_ubuntu.sh 1 175000039
```

It compares several square chains, explicit sync, copy flush and a factor-3
square word-for-word with lead caching enabled and disabled.
