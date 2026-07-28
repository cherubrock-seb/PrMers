# PrMers v99.92 — Gaussian Windows/MSVC portability fix

PrMers version:

    4.20.81-alpha-v99.92-gaussian-windows-msvc-portability-fix

Aevum remains unchanged:

    v0.3.78-workload-plan-policy-audit-fix

## Corrections

The Gaussian-Mersenne implementation used GNU unsigned __int128 for exact
64-bit modular multiplication and overflow-safe admissible-factor generation.

MSVC x64 does not support that keyword.

v99.92 now uses:

1. _umul128 and _udiv128 on MSVC x64.
2. unsigned __int128 on GCC and Clang.
3. An overflow-safe add-and-double fallback on other compilers.

The admissible-factor loops now derive their overflow safety directly from
k <= (limit - 1) / step and no longer require a 128-bit temporary.

CliParser.cpp also exceeded the MSVC nested-block limit because of its very
long option else-if chain. The second half of the parser is now handled by a
separate helper function without changing CLI behavior.

## Isolation

No Aevum source, OpenCL kernel, Marin kernel or ordinary Mersenne arithmetic
has been modified.

## CI

A dedicated Windows/MSVC portability source audit now runs before the Windows
CMake build and is also part of make test-gm.
