# PrMers v100.40 — Aevum gfx906 PFA3 fused inverse normalization

## Aevum dependency

- Standalone Aevum tag: `v0.3.105-pass5-prp-max`
- Standalone Aevum source SHA: `ce0b8b5ee3aa4d7c79d55858f5bb1fa783073610`
- Standalone release workflow: PASS on Linux x86_64, Windows x86_64, macOS x86_64 and macOS arm64.
- Published standalone release: 9 assets, including per-package SHA256 files and `SHA256SUMS`.
- PrMers vendors the exact released Aevum source delta under `third_party/aevum`.

## Production change

Aevum v0.3.105 enables fused inverse normalization by default only for legal
PFA3 GF31/GF61 plans on AMD gfx906 / Radeon VII.

PrMers does not change its plan selector in this release. Existing
`pfa:auto` ranges remain authoritative, so the new path is used automatically
whenever the existing selector chooses PFA3 on gfx906. Explicit PFA controls
and `AEVUM_PFA_FUSED_INVERSE=0/1` retain precedence.

The standalone Aevum delta from v0.3.104 to v0.3.105 is limited to:

- `Makefile`
- `src/Gpu.cpp`
- `src/UseOptions.h`
- `src/cl/fft-middle.cl`
- `src/cl/fftmiddleout.cl`

There is no PrMers AUTO range, capacity, carry, register-layout, PFA7, PFA9,
Type2, Type4, CUDA, or Apple policy change in this integration.

## Performance and correctness evidence

Radeon VII / gfx906 PFA3 long paired validation:

- median full-iteration gain: `+0.9019%`;
- trimmed mean gain: `+0.9008%`;
- wins: `6/6`;
- exact residues: PASS.

RTX 3080 keeps the fused PFA3 inverse option default-off and passed exact
non-regression validation. Explicit opt-in/out remains available for controlled
validation.

## Pre-tag PrMers gates

Before tagging v100.40, the integration branch must verify:

- vendored Aevum files are byte-identical to standalone `v0.3.105-pass5-prp-max` for the release delta;
- Aevum source/host gates pass from `third_party/aevum`;
- PrMers source and build gates pass;
- existing AUTO policy boundaries remain unchanged;
- a gfx906 PFA3 AUTO probe shows the fused inverse define enabled;
- RTX AUTO remains default-off for the fused PFA3 inverse policy.

No new performance claim is inferred from integration-only probes.
