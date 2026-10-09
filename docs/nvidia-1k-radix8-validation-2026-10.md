# NVIDIA 1K radix-8 default validation — October 2026

For NVIDIA devices, 1K Aevum plans default to radix-8 when
`AEVUM_RADIX1K` is not explicitly set.

Explicit radix-4 and radix-8 remain overrides. Apple and non-NVIDIA devices
retain radix-4.

For NVIDIA 1K radix-4, the merged GPU configuration defaults `WMUL=1` only
when no explicit or tuned WMUL already exists. This keeps host and kernel
carryFused geometry consistent and avoids an illegal 512-thread work-group.

Authoritative RTX 3080 full-iteration results:

- p=210000017:
  - radix-4 / WMUL=1: 1533, 1539, 1540 us/iteration
  - radix-8 / WMUL=2: 1470, 1473, 1476 us/iteration
  - median gain: +4.481%
  - exact residue: PASS

- p=219999919:
  - radix-4 / WMUL=1: 1526, 1537, 1539 us/iteration
  - radix-8 / WMUL=2: 1468, 1470, 1474 us/iteration
  - median gain: +4.558%
  - exact residue: PASS

Radeon VII keeps the radix-4 default and passes exact non-regression.

Verdict: KEEP.
