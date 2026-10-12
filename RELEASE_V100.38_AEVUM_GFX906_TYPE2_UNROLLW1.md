# PrMers v100.38 — Aevum gfx906 Type2 retained-width policy

## Aevum dependency

- Standalone Aevum tag: `v0.3.104-pass5-prp-max`
- Standalone Aevum source SHA: `09d8572ad92f7fe9cbb36ec20f12270b838bb931`
- PrMers embeds that exact released Aevum source under `third_party/aevum`.

## Production change

For ordinary PRP AUTO on AMD gfx906 / Radeon VII, exponent
`130000000..137000003`, when Aevum selects
`2:512:8:512:202`:

- existing production queue policy remains `MULTI_Q=0`;
- width policy now selects `UNROLL_W=1`;
- explicit `UNROLL_W=0` remains higher precedence;
- RTX behavior is unchanged.

## Performance evidence

R86 Radeon VII paired validation used eight counterbalanced
4096-iteration pairs with exact residues:

- median full-iteration gain: `+4.7218%`;
- trimmed mean gain: `+4.5879%`;
- wins: `8/8`;
- `carryFused` improvement: approximately `+14.49%`.

The final integration gates do not infer a new speedup; they validate that the
measured R86 policy is selected correctly in production AUTO.

## Final correctness / integration gates

- Aevum PR #25 exact-head source gate: PASS.
- Radeon VII direct EngineApi:
  AUTO -> `2:512:8:512:202`,
  `MULTI_Q=0`, `UNROLL_W=1`.
- Explicit `UNROLL_W=0`: PASS with identical 256-step residue hash
  `0x4ab5988ebc5010b6`.
- Native PrMers Bench3 AUTO using the exact PR plugin: exact and production
  probes PASS.
- RTX 3080: same Type2 plan and same deterministic residue as baseline,
  `MULTI_Q=1`, zero gfx906-policy activation.

Apple remains outside this gfx906-only policy.
