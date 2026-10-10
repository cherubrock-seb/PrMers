# PRP shape production-confirmation validation (October 2026)

## Scope

This note records the validation of the generic PRP FFT-shape production-confirmation gate added on branch `cherub/p145-shape-confirm-r93`.

The change keeps the existing short `comparePlans` pass as the cheap exactness, ROE and candidate-ranking stage, but a non-native PRP shape is no longer allowed to become the persistent winner solely from that short synthetic timing. Before it can replace the native plan, it must also pass a longer production-style retained-width square confirmation.

The production confirmation is device-measured. It does not hard-code Radeon or NVIDIA winners.

Validation baseline:

- PrMers / Bench2 continuation source: `bb03404ab74f647b664049f0c97b7381a447a537`
- Patch branch source before this document: `2db48f677be6710a198c8ce2cecf4088e4dd3852`
- Aevum version string: `v0.3.100-pass5-prp-max`
- Radeon VII: OpenCL device 0, gfx906
- NVIDIA RTX 3080: OpenCL device 2
- Historical dense Bench2 remained unchanged and was not rerun.

## Why the extra confirmation is needed

At `p=145000007`, the short shape ranker produced results that did not agree with the public production square path.

The independently validated Radeon production result was:

- native `1:1K:8:256:101`
- old AUTO `1:512:8:512:101`
- native production gain versus old AUTO: **+7.525% median**
- worst observed production gain: **+6.632%**

The same native Radeon winner is not suitable as a global hard-coded replacement: on RTX 3080 it regressed versus the RTX winner by **-4.780%**.

The problem was therefore not the absence of a single universal FFT plan. The problem was that the device-specific shape ranker could promote a false positive.

## R91: metric mismatch proof

Nine exact public-engine measurements were retained or completed with:

- workload: PRP
- register count: 8
- warmup: 64
- measured squares: 384
- three measurements per plan
- identical final residue SHA-256 across all observations:
  `454f80615706e25d4166c033eac6e00367cf501562282f73c3b709b772de22aa`

Results at `p=145000007` on Radeon VII:

| Plan | Role | Public square_mul median | comparePlans claim | Public result vs native |
| --- | --- | ---: | ---: | ---: |
| `1:1K:8:256:101` | native | 1510.903 us/iter | baseline | baseline |
| `1:1K:8:256:202` | short-ranker winner | 1509.292 us/iter | +21.82% | +0.107% |
| `1:1K:4:512:101` | expanded short-ranker winner | 1619.435 us/iter | +5.28% | -6.702% |

The short ranker overstated the first candidate by about 21.7 percentage points and the second by about 12.0 percentage points.

Verdict:

`COMPAREPLANS_METRIC_MISMATCH_PROVEN`

This authorized a generic confirmation stage before a positive PRP shape-cache decision.

## Implementation

The patch adds a PRP-only `confirmPrpPlanProduction` stage in `third_party/aevum/src/EngineApi.cpp`.

The confirmation:

- runs only after the cheap candidate search has produced a non-native winner,
- uses the production retained-width square path,
- warms both native and candidate for 64 squares,
- times paired 256-square sequences,
- alternates A/B order for three pairs,
- requires exact word equality,
- requires median gain of at least the configured threshold and at least 3%,
- requires at least two of three pairs to gain at least 2%,
- requires the worst pair to be at least 0.985x,
- falls back to the native plan on a marginal, noisy, inexact or failed confirmation.

PRP shape-cache keys also include:

`shape-confirm=prp-square-v1`

This prevents positive decisions created by the older short-only metric from bypassing the new confirmation.

## R93: build and source checks

Patch head `2db48f677be6710a198c8ce2cecf4088e4dd3852`:

- Pass-4 source audit: PASS
- Aevum engine build: PASS
- patched engine SHA-256:
  `b1d0b3b9aa22552973db766ad61a2bd98b46b9b6258c4a817f67d47d0c019628`

A fresh Radeon ranking happened to select native directly, so R93 did not exercise the new confirmation branch. This was treated as a validation-path miss, not as a patch failure.

## R94: deterministic production-confirm gate

To exercise the gate without depending on short-ranker noise, disposable deferred-cache checkpoints were used. The production autotune cache was hashed before and after and remained unchanged.

### Radeon VII false-positive rejection

Seeded non-native candidate:

`1:1K:8:256:202`

Production confirmation:

- exact: yes
- median gain: **0.99780x**
- worst pair: **0.96007x**
- wins: **0/3**
- keep: **false**

Final selected and persisted plan:

`1:1K:8:256:101`

A subsequent AUTO call hit the new-generation cache and retained the native plan.

### RTX 3080 positive winner retention

Candidate:

`1:512:8:512:202`

A fresh tune first measured:

- median gain: **1.05473x**
- worst pair: **1.05212x**
- wins: **3/3**
- keep: **true**

The deterministic deferred-checkpoint confirmation then measured:

- median gain: **1.03913x**
- worst pair: **1.03766x**
- wins: **3/3**
- keep: **true**

Final selected and persisted plan:

`1:512:8:512:202`

A subsequent AUTO call hit the new-generation cache and retained the RTX winner.

R94 verdict:

`DEVICE_MEASURED_GENERIC_GATE = PASS`

## R95: neighboring non-regression

Neighbor points were tested with production shape-search defaults:

- candidate cap: 5
- budget: 7000 ms
- minimum short-ranker gain: 1.04
- isolated disposable cache per GPU and exponent
- deferred searches resumed rather than restarted

Results:

| GPU | Exponent | Preserved dense plan | Patched AUTO | Verdict |
| --- | ---: | --- | --- | --- |
| Radeon VII | 130000007 | `1:1K:8:256:101` | `1:1K:8:256:101` | PASS |
| Radeon VII | 150000007 | `1:1K:8:256:101` | `1:1K:8:256:101` | PASS |
| RTX 3080 | 130000007 | `1:1K:8:256:101` | `1:1K:8:256:101` | PASS |
| RTX 3080 | 150000007 | `1:512:8:512:202` | `1:512:8:512:202` | PASS |

Explicit-path residue equality passed at every neighbor point.

Representative current explicit timings were also consistent with the preserved dense behavior:

- Radeon p130: 1512.61 / 1566.21 us vs preserved dense 1502.52 us
- Radeon p150: 1500.36 / 1547.12 us vs preserved dense 1502.61 us
- RTX p130: 742.63 / 743.19 us vs preserved dense 747.73 us
- RTX p150: 712.09 / 712.19 us vs preserved dense 713.07 us

No neighbor selected a different FFT shape.

## Safety and repository state

Throughout R91-R95:

- historical Bench2 dense results were preserved,
- no dense campaign was rerun,
- the real production autotune cache was not modified,
- `main` was not modified,
- `cherub/bench2-v10032-r68` remained at `bb03404ab74f647b664049f0c97b7381a447a537`,
- validation used isolated caches and a dedicated patch worktree.

## Conclusion

The validation supports the production-confirmation design.

The existing `comparePlans` metric remains useful as a cheap exactness/ROE/ranking stage, but its short timing is not sufficiently faithful to be the final authority for a positive PRP shape-cache decision.

The new second-stage confirmation fixes that failure mode without hard-coding GPU-specific plans:

- Radeon VII rejects the p145 false positive and keeps its native winner.
- RTX 3080 independently retains its faster p145 plan.
- p130 and p150 neighboring behavior is unchanged on both GPUs.
- exactness and persistent-cache behavior pass.

After the final relevant source/build tests, this branch is suitable for normal integration. The completed dense and selector campaigns do not need to be repeated.
