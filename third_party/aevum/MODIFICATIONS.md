# Aevum modifications

Modified in 2026 by cherubrock-seb for use as a reusable arithmetic engine and for integration with PrMers.

Main changes from the imported GPUOwl/PRPLL snapshot:

- added a stable C shared-library API in `src/EngineApi.cpp` and `src/EngineApi.h`
- exposed indexed arithmetic registers for external algorithms
- added register set, import, export, copy, square, multiply, add and subtract operations
- added prepared-transform caching for repeated multiplication
- restricted the reusable engine path to paired FFT3161 arithmetic over `GF(M31^2) x GF(M61^2)`
- added admissible FFT3161 plan resolution for external callers
- added a small-factor scratch path that avoids sparse constant transforms and raw `regScale`
- added source audits, API loading tests and GPU arithmetic tests
- added standalone register examples with CPU verification
- added build and installation targets for `libaevum_engine.so`
- added a PrMers adapter implementing an interface compatible with PrMers `engine::Reg` users

Marin by Yves Gallot inspired the register-oriented external interface. Aevum remains a derivative of GPUOwl/PRPLL; it is not a codebase merge or an official Marin backend.

## Aevum 0.3.3

- added GPU-side register equality to the public engine ABI
- switched PrMers Gerbicz checks to register equality instead of full residue export
- made `compactBits` carry propagation bounded and cyclic modulo `2^p - 1`
- added host tests for positive and negative Mersenne carry wrap
- kept direct compact-word export for final GMP conversion and checkpoints
