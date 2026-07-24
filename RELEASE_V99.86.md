# PrMers v99.86 workload-aware Aevum plan policy and audit fix

This candidate follows the validated v99.85 correctness release and deliberately
keeps the experimental M19 path excluded.

## Corrected workload audit

- Resume comparisons now hash the canonical mathematical fields instead of the
  raw `.p95`/`.save` bytes. `PROGRAM`, `TIME`, `WHO`, formatting and harmless
  leading zeroes no longer create false mismatches.
- A candidate that runs but is not word-exact is reported as `REJECTED`, not as
  an infrastructure failure.
- Standard measurements are repeated twice and full measurements three times;
  recommendations use the median.
- Standard/full profiles also run shorter word-exact differentials at additional
  PRP/LL and P-1/ECM exponents to catch size-boundary regressions.
- The report now emits `recommended-workload-plans.env` and `policy.tsv`.
- `--strict-policy` verifies that the built-in selector matches the fastest
  exact measured plan.

## Workload-specific selectors

On non-Apple platforms, automatic Aevum selection now requests a selector that
matches the actual operation mix:

- PRP: `throughput:prp`
- LL: `throughput:ll`
- P-1 Stage 1: `throughput:pm1`
- ECM: `throughput:ecm`

The RTX 3080 calibration validated by the audit is:

- PRP: `4:512:8:512:202`
- LL: `4:1K:2:1K:202`
- P-1 Stage 1: `4:256:16:256:202`
- ECM: stock Type1 `1:1K:4:256:101` until a repeatable Type4 gain is shown.

P-1 Stage 2 V-trace and classic BSGS continue to use Marin.

Environment overrides remain available:

- `PRMERS_AEVUM_PRP_FFT`
- `PRMERS_AEVUM_LL_FFT`
- `PRMERS_AEVUM_PM1_FFT`
- `PRMERS_AEVUM_ECM_FFT`

## Apple policy

macOS remains unchanged: Marin is the default, forced Aevum is limited to
validated PRP/LL stock Type1 FFT3161, and Type4/PFA or Aevum ECM/P-1 requests
are rejected before GPU execution.
