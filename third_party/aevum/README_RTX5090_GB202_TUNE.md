# RTX 5090 / GB202 measured tuning candidate

A contributor reported a repeatable 236 us -> 203 us PRP improvement near exponents 146M-150M using:

```
-fft 1:512:8:512:202 -use INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0
```

This is intentionally not a global default: FFT shape and `-use` tuning are device, driver and exponent-range specific. Before shipping an automatic GB202 entry, capture the complete generated `tune.txt`, exact device name/PCI identity, driver version, and cross-check PRP/LL residues at both range boundaries.

Reproduce with:

```bash
./aevum -tune ntt,minexp=146000000,maxexp=150000000
```


## Automatic tune reuse (issue #36)

A normal standalone Aevum run with no explicit `-fft` now reuses compatible
`FFT3161` entries written by `-tune` to `tune.txt`. This fixes the case where
an autotuned FFT shape was measured and saved but the next run silently fell
back to the built-in selector.

Selection precedence is intentionally:

1. explicit `-fft` from the command line (or `config.txt`),
2. compatible `FFT3161` entry from `tune.txt`,
3. built-in automatic `FFT3161` selection.

Historical GPUOwl `tune.txt` entries without a type prefix are ignored
entirely. Native automatic tune reuse accepts only records beginning exactly
with `1:` (FFT3161). This deliberately avoids ambiguous historical forms such
as `1K:...`, which are dimensions, not FFT type prefixes. The `-use` settings written by `-tune` in `config.txt` continue to
apply normally.

For the RTX 5090 / GB202 report around exponents 147--148M, the measured plan
was:

```
-fft 1:512:8:512:202 -use INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0
```

It was reported at about 203 us/iteration versus about 236 us/iteration for the
previous automatic choice. This exact shape is not hard-coded globally: it is
range- and device-dependent, so the tuner remains the source of truth.
