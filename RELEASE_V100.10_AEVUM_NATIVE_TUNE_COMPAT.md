# PrMers v100.10 — Aevum native tune compatibility / GB202 profile

This candidate addresses the remaining tune-reuse part of issue #36 without
misclassifying the bundled historical GPUOwl/PRPLL table.

## Correct tune-format handling

The bundled 150-line `tune.txt` is an old unprefixed FP64 table using the
single-digit FFT variant encoding `0..3`. Upstream later changed that encoding
to the current three-digit WMH form (`000`, `101`, `202`, ...). Those old
timings are therefore not valid FFT3161 NTT timings and must not simply receive
a `1:` prefix.

v100.10:

- classifies old single-digit unprefixed records as legacy FP64 and summarizes
  them once instead of emitting 150 rejection lines;
- accepts current explicit FFT3161 records (`1:...`);
- accepts modern unprefixed three-digit WMH NTT records such as
  `512:8:512:202` and normalizes them to explicit Type1;
- ignores malformed/mixed records without aborting engine startup;
- makes `AEVUM_TUNE_DIR` authoritative for both the device-neutral resolver and
  runtime.

## RTX 5090 / GB202 measured shape

For automatic PRP on RTX 5090 / GB202 in the measured 146M–150M interval,
Aevum selects the issue #36 measured shape:

    1:512:8:512:202

Safety:

- explicit `-aevum-fft` wins;
- user `AEVUM_TUNE_DIR` wins;
- other devices are unchanged;
- the built-in profile is restricted to 146M–150M;
- `AEVUM_GB202_TUNE=0` disables it;
- `AEVUM_GB202_TUNE=force` is a validation-only override for another GPU.

The original v0.3.4 tune also reported
`LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0`.
Those micro-tune controls are not present in the current embedded Aevum source.
v100.10 deliberately does not claim to reproduce them. The expected first
target is therefore the reporter's explicit-shape class (~225 us/iter), with
the ~201 us historical fully tuned result to be re-measured separately.

## Preserved fixes

- 147.8M remains Type1 and does not regress to same-size Type4.
- v100.09's 197M Type4 4M boundary bridge remains unchanged.
