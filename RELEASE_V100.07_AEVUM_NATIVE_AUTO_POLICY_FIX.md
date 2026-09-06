# PrMers v100.07 — Aevum native auto policy fix

Fixes the main performance regression exposed by GitHub issue #36.

## What was wrong

On non-Apple platforms PrMers forced the workload selector `throughput:prp`
(and `throughput:ll`) even when the user had not requested a plan override.
That selector contains RTX-3080-derived Type4 cost assumptions. On the issue
reporter's RTX 5090 this selected `4:512:8:512:202` at roughly 658 us/iter,
while standalone Aevum selected Type1 automatically at roughly 236 us/iter.

The v100.05/v100.06 tune reuse change did not solve that policy problem. The
bundled historical `tune.txt` is also not a native typed Aevum tune file, so its
unprefixed entries remain intentionally ineligible for native FFT3161 reuse.

## Fix

For ordinary Mersenne PRP/LL and Gaussian-pair PRP/Proth on non-Apple systems:

1. an explicit `-aevum-fft` remains authoritative;
2. `PRMERS_AEVUM_PRP_FFT` / `PRMERS_AEVUM_LL_FFT` remain authoritative;
3. otherwise PrMers now passes an empty FFT spec and delegates selection to
   `aevum_engine_resolve_auto_fft()`, matching standalone Aevum.

P-1 and ECM policies are unchanged. Explicit `throughput:prp` and
`throughput:ll` selectors remain available for diagnostics or device-specific
overrides.

## Regression coverage

The AutoPolicy test now checks exponent 147800003 from issue #36 and requires
the device-neutral default path to resolve to Type1 FFT3161. Source-policy
tests also guard against reintroducing `throughput:prp`/`throughput:ll` as
hard-coded App defaults.

## Expected RTX 5090 retest

With no explicit FFT override, startup should report a Type1 FFT3161 family
rather than Type4 FFT323161. Based on the reporter's measurements this should
restore approximately the standalone-Aevum default range (~236 us/iter versus
~658 us/iter). Reaching the separately measured ~201 us/iter tuned result
requires a follow-up device/range-scoped tune-profile solution; this release
does not hard-code the GB202 tune globally.
