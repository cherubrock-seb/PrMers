# PrMers v100.19 reliability

Based on v100.18.

Issue #38 reliability fixes:
- Gerbicz-Li rollback after resume restores the actual checkpoint state.
- Stop after three repeated deterministic Gerbicz-Li failures.
- Fix Marin radix-5 divergent OpenCL barrier.
- Recover safely from incomplete Aevum kernel-cache entries.
- Fix MSYS2/MinGW libdl linking.
- Aevum library no longer owns/closes host stdout.
- Remove obsolete build_with_aevum_engine.sh references.

Deliberately not included:
- no PRP square-loop batching;
- no Gerbicz B=1000 performance change;
- no change to v100.18 GPU proof generation.
