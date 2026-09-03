# PrMers v100.05 — Aevum automatic tune reuse

Fixes GitHub issue #36.

Aevum now automatically reuses compatible FFT3161 plans produced by its
tuner and stored in tune.txt.

Selection priority:

1. Explicit `-fft`
2. Compatible native Aevum FFT3161 entry from `tune.txt`
3. Existing built-in automatic FFT3161 selector

Historical GPUOwl tune entries without an explicit Aevum type prefix are
ignored. In particular, dimension strings such as `1K:...` cannot be
misinterpreted as FFT type `1:`.

Regression coverage includes legacy H=3 entries and the historical `1K:`
format.

Validated on cherubrock2 with the full Aevum host regression suite and
runtime GPU smoke test.
