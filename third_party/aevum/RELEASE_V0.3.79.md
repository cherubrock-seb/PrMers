# Aevum v0.3.79 — automatic tune reuse

- Reuse compatible FFT3161 entries from `tune.txt` automatically.
- Explicit `-fft` always retains priority.
- Ignore historical unprefixed GPUOwl tuning entries.
- Prevent `1K:...` legacy plans from being mistaken for FFT type `1:`.
- Add regression tests for issue #36.
