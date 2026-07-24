# Aevum v0.3.78 workload plan policy and audit fix

- Adds `throughput:prp`, `throughput:ll`, `throughput:pm1` and
  `throughput:ecm` selectors.
- Keeps explicit plans unchanged and keeps PFA outside workload-specific
  defaults until its differential tests are exact.
- Adds host tests for the expected PRP, LL, P-1 and ECM selector geometry.
- Keeps Apple on stock FFT3161.
