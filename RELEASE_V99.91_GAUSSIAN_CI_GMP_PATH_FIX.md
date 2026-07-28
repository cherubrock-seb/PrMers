# PrMers v99.91 — Gaussian CI GMP path fix

PrMers version:

    4.20.80-alpha-v99.91-gaussian-ci-gmp-path-fix

Aevum remains unchanged:

    v0.3.78-workload-plan-policy-audit-fix

## Correction

The Gaussian worktodo parser test previously invoked the compiler without
using the portable GMP installation exported by the macOS workflow.

The main PrMers build succeeded, but make test-gm could not locate gmpxx.h
on the macOS arm64 and Intel package runners.

The test now resolves GMP in this order:

1. GMP_PREFIX exported by the release workflow.
2. pkg-config.
3. Homebrew.
4. System compiler and linker paths.

## Isolation

No Gaussian arithmetic, Aevum source, OpenCL kernel, Marin kernel, PRP, LL,
P-1 or ECM algorithm has been changed.

## CI

The Gaussian mathematics, dispatch, JSON and native worktodo tests continue
to run in the Linux and macOS release workflows.
