# PrMers v100.08 — Aevum 1K radix-8 opt-in

This release keeps the gpuowl `6cf0dc` 1K radix-8 implementation available
without allowing it to regress existing GPUs.

## Safety policy

- radix-4 remains the default for every 1K width/height;
- radix-8 is enabled only by explicit `AEVUM_RADIX1K=8`;
- `AEVUM_RADIX1K=4` explicitly selects the safe legacy path;
- any other non-empty value is rejected at Aevum engine creation;
- 256 keeps radix-4 and non-1K dimensions retain their historical behavior.

Radeon VII validation at exponent 20,999,999 showed radix-8 roughly 17–19%
slower. RTX 3080 native gpuowl A/B was essentially neutral (+0.39% median for
the full upstream commit). No GPU-vendor heuristic is therefore hard-coded.

## Issue #36 policy preserved

Ordinary PRP/LL without explicit overrides still pass an empty FFT specification
to Aevum and use plugin-native auto selection. PrMers does not reintroduce
`throughput:prp` / `throughput:ll` defaults.

P-1 and ECM retain their workload-specific selectors.

## UI / diagnostics

The backend console line and the existing WebGUI backend-detail state show:

- `radix1k=4 safe-default`, or
- `radix1k=8 explicit-override`.

No WebGUI state schema change is required.

## Explicit tuning example

```bash
AEVUM_RADIX1K=8 ./prmers ...
```

This is a measurement/diagnostic override only. Device-scoped automatic radix
selection should be added only after repeatable per-GPU tuning data proves a
meaningful win.
