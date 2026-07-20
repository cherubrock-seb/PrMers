# PrMers v99.75 — throughput auto selector and experimental PFA9 lead bridge

## Default PRP/LL selection

An empty Aevum FFT request now resolves through `throughput:auto` instead of
selecting by transform size alone.  The score accounts for the measured cost
of each arithmetic family.  At exponent 175000039 the default result is:

```
4:512:8:512:202
```

This is the 4M FFT323161 power-of-two plan with the validated register
LEAD_WIDTH cache.  `-pfa-off` uses `pow2:auto`, which excludes PFA candidates
but keeps the same cost model.

Device-specific overrides are available:

```
AEVUM_AUTO_STOCK_COST
AEVUM_AUTO_POW2_TYPE4_COST
AEVUM_AUTO_PFA3_COST
AEVUM_AUTO_PFA9_COST
```

## PFA9 acceleration experiment

Good-Thomas PFA order does not permit the stock canonical-order `carryFused`
kernel to be reused directly.  The experimental bridge applies carryB while
gathering the next PFA fftP input.  It is disabled by default pending exact
GPU validation and a speed comparison.

Enable only for testing:

```
AEVUM_PFA_LEAD_BRIDGE=1 AEVUM_REG_LEAD_CACHE=1
```

The required exact marker is:

```
PFA9 FFT3161 LEAD-BRIDGE DIFFERENTIAL TEST PASSED
```
