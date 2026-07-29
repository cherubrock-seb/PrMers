# PrMers v99.93

PrMers v99.93 adds factor-only Gaussian Mersenne campaigns for GMNet.

## New worktodo policy

```text
GMCHAIN=p,pm1_B1,pm1_B2,ecm_B1,ecm_B2,curves,sieve,chunk_bits,finish
```

- `finish=proth` keeps the historical P-1 -> ECM -> Proth pipeline.
- `finish=factor` stops after P-1 and optional ECM.
- Omitting `finish` remains backward compatible and continues to Proth.

P-1 Stage 1 only continues to use `GMPMINUS1` with `B2 = B1`.

GMNet:

```text
https://gmnet.gaussianmersenne.workers.dev/
```

GMRelay:

```text
https://github.com/cherubrock-seb/gmrelay
```
