# PrMers v100.39 — release source-audit cwd hotfix

PrMers v100.38 tagged the validated Aevum v0.3.104 integration, but the Linux
and macOS release workflows stopped in the PrMers source audit before package
creation.

The failing wrapper launched:

`third_party/aevum/tests/type2_prp_auto_source_test.py`

from the PrMers repository root. That standalone Aevum test intentionally reads
`src/EngineApi.cpp` relative to the Aevum source root, so it failed with
`FileNotFoundError` in the vendored invocation.

v100.39 changes only the PrMers audit wrapper so this test runs with
`third_party/aevum` as its working directory.

There is no runtime, kernel, selector, arithmetic, or vendored Aevum source
change relative to v100.38.

- Standalone Aevum tag: `v0.3.104-pass5-prp-max`
- Standalone Aevum source SHA: `09d8572ad92f7fe9cbb36ec20f12270b838bb931`
- Vendored Aevum tree SHA remains:
  `5ad42dbb5d69b1539620a12cf467a8b2bcc41a3b`
- The validated gfx906 Type2 production policy remains:
  `2:512:8:512:202`, `MULTI_Q=0`, `UNROLL_W=1` in the validated window,
  with explicit `UNROLL_W=0` precedence and RTX behavior unchanged.

The v100.38 Windows workflow succeeded and published its package; Linux and
macOS failed only at this source-audit cwd issue. v100.39 supersedes that
partial release and is intended to complete all platform release workflows.
