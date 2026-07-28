# PrMers v4.20.78-alpha-v99.89 — Gaussian Stage 2 progress, JSON and worktodo

## Fixed

- P-1 Stage 2 now reports progress inside every product-exponent chunk instead
  of remaining silent for the full 262144-bit block.
- Progress includes global percentage, approximate processed-prime count,
  per-chunk percentage, bit count, bit throughput, elapsed time and ETA.
- Ctrl-C inside a Stage 2 chunk restores the clean chunk-start state and writes
  a resumable checkpoint.
- Removed harmless repeated Aevum `Read ZERO` messages by initializing all P-1
  scratch registers before the first checkpoint.
- Gaussian factoring checkpoint format bumped to version 3. Existing v99.88
  version-2 checkpoints remain loadable and are upgraded on the next save;
  legacy zero scratch registers are normalized after loading.

## Added

- Uniform Gaussian-Mersenne JSON schema version 1 for P-1, ECM, PRP and Proth.
- UTC timestamps, actual OpenCL device name, backend, elapsed time, bounds,
  factor stage/source and nullable mode-specific fields.
- Native worktodo entries: `GMPROTH`, `GMPRP`, `GMPMINUS1`, `GMECM`.
- Conditional `GMCHAIN` entry: P-1 -> optional ECM -> deterministic Proth,
  stopping immediately after a factor.
- Automatic removal/archive of completed Gaussian worktodo lines and restart on
  the next entry. Interrupted/error tasks remain active.
- Prime-exponent worktodo generator and resumable record-campaign wrapper.
- Native Gaussian queue completion removes the exact parsed line, preserving
  comments and unrelated/unknown lines that may precede it.

## Isolation

No Aevum source, OpenCL kernel, ordinary Mersenne PRP/LL/P-1/ECM algorithm or
plan-selection policy was changed. The new behavior is entered only through a
Gaussian CLI mode or native Gaussian worktodo keyword.
