# PrMers v100.14 — Aevum Pass5 PRP MAX

Release tag: `v4.20.97-alpha-v100.14-aevum-pass5-prp-max`

## Aevum dependency

- Standalone Aevum tag: `v0.3.84-pass5-prp-max`
- Standalone Aevum source SHA: `efd12556fb0e03d8e1107ad353a910699929642c`
- PrMers embeds the released Aevum source under `third_party/aevum`.

## Validated PRP improvement

- Radeon VII geometric-mean speedup near p≈170M: **+13.527%**
- RTX 3080 geometric-mean speedup near p≈170M: **+8.646%**
- Winning production geometry: `1:512:16:512:101`
- Correctness: PASS on paired exact-residue validation.
- Representative exponents: `169999961`, `170000009`, `170000093`.

## Release validation

- Standalone Aevum v0.3.84 release: PASS on Linux, Windows and macOS.
- PrMers non-publishing release preflight: PASS on Linux, Windows and macOS before this metadata-only preparation.
- Embedded Aevum source synchronized to the released standalone state.
- Windows embedded-Aevum host-test recipe synchronized with the released standalone workflow.

The production release must be built from the exact immutable PrMers tag and must not apply release-time tuning overrides.
