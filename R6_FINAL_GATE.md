# R6 final gate

R6 keeps R5's bounded 12-candidate `-use` search and `Gpu::timePRP()` ranking, but a positive implementation profile can no longer be persisted from `timePRP()` alone.

Final promotion requires a second production-engine-style gate:

- exact differential already passed;
- 64 warm-up `square_mul` operations;
- reset to the deterministic residue;
- 256 measured `square_mul(reg,1)` operations;
- alternating baseline/candidate A/B x3;
- median gain >= 1.03;
- at least 2/3 paired wins >= 1.02;
- worst paired ratio >= 0.985.

The implementation cache namespace is bumped to `prp-use-v4`, so R5 positive/negative decisions cannot leak into R6.

The release non-regression harness now uses five alternating Pass4/R6 pairs. It requires median R6 throughput >= 0.985x Pass4 and at least 4/5 paired ratios >= 0.97. This tolerates one isolated DVFS outlier but still rejects persistent regressions.

## Hardware gates to rerun

RTX 3080 minimum gate:
- 21,000,029: preserve automatic positive `-use` selection and >=3% same-version engine gain.
- 180,000,007 and 196,999,969: preserve validated Type4 shape and no regression.

Radeon VII minimum gate:
- 150,000,007: R5 false positive must NOT be promoted; expected defaults-complete unless another profile passes the engine gate.
- 210,000,017: preserve the R5 positive profile if it still passes >=3% engine gain.
- 147,800,003: recheck the prior noisy/negative Pass4 comparison with the new 5-pair release gate.

For final publication confidence, run the full 9-exponent range sweep on both machines.

## RTX 5090 / issue #36 discovery gate

The same harness accepts `AEVUM_TARGET=rtx5090`. Recommended first sweep:
`147800003,180000007,196999969,210000017`.
The 147.8M point is the historical shape+`-use` target; the others test Type4/large-exponent behaviour and a higher-capacity boundary. Return the generated `tuning-output.zip`.
