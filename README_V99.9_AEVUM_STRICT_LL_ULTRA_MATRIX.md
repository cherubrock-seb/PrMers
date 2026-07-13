# PrMers v99.9 — strict Aevum, LL routing and early compatibility guards

This corrective release keeps the v99.8 Aevum arithmetic paths and changes the
request semantics and validation layer:

- `-aevum` is strict. No hidden Marin fallback is performed when FFT3161 has no
  plan for the exponent.
- Automatic mode still selects Marin normally when Aevum is unavailable or loses
  the workload-specific transform ratio comparison.
- `M216091` is intentionally a Marin case; `M1362763` exercises forced Aevum LL.
- `-llunsafe -marin` is rejected because the old internal NTT LL result is not
  validated.
- `-pm1-ultralowmem -aevum` is rejected before constructing `App`, OpenCL
  contexts, or the MM31 transform.
- The backend matrix recognizes `probably prime`, checks JSON status `P`, treats
  useful `timeout`/`SIGKILL` smoke runs separately, and tests the two rejection
  paths explicitly.
