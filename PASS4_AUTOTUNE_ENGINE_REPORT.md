# AEVUM Pass 4 — Autotune + Shared Engine Optimization

## Status

Pass 4 source implementation is complete. Local host/source/correctness gates pass. Real GPU performance validation is **pending real hardware validation** on cherubrock1 (RTX 3080, flattened OpenCL device 2) and cherubrock2 (Radeon VII/gfx906, device 0). No speedup is claimed for the new prepared-width bridge until those commands are run.

Baseline provenance preserved: PrMers `a813a122f19ea0424702ba552805b2ceb7c1e606`, issue #36/v100.10 tune-selection compatibility, AEVUM_FUSED_LL, disabled invalid FUSED_MUL3, and PRP_MIDDLE1 remaining explicit/experimental.

## A. Production ultra-short runtime autotuner

Implemented in `third_party/aevum/src/RuntimeAutotune.{h,cpp}` and `third_party/aevum/src/EngineApi.cpp`, with workload propagation through the PrMers AEVUM adapter.

- Default mode: `AEVUM_AUTOTUNE=auto`; controls also accept `off` and `force-retune`/`retune`.
- Default first-use budget: at most **5 serious candidates** and **7000 ms** (`AEVUM_AUTOTUNE_MAX_CANDIDATES`, `AEVUM_AUTOTUNE_BUDGET_MS`).
- Candidate generation starts from native AUTO, measured Pass-3/range-sweep seeds when applicable, the transform-size boundary around 180–197M, the opposite 101/202 variant, and a small ranked neighborhood of admissible Type1/Type4 shapes.
- PFA/PFA9 candidates are excluded from runtime selection because supplied hardware evidence contained WORD MISMATCH results.
- Every timed candidate first passes a deterministic engine-level word-exact differential sequence against native AUTO.
- Timing uses warmup, alternating native/candidate order and 3-sample medians. Default replacement threshold is 1.04x.
- Cache hits do not benchmark. Cache records are human-readable TSV and atomically replaced.
- Cache key includes schema/engine version, OpenCL vendor/device, driver/runtime, workload class, exponent band, register band and tuning-relevant flags.
- Corrupt/stale cache records are ignored safely.

### Manual override precedence

The existing user-control behavior is preserved: explicit FFT/PFA plans are never replaced; the validated GB202/issue #36 profile and explicit tuning/environment sources are respected; compatible `tune.txt` data bypasses runtime search; only then can the persistent cache/runtime tuner act, with native AUTO as fallback. The PrMers `PRMERS_AEVUM_{PRP,LL,PM1,ECM}_FFT` overrides continue to pass explicit plans into the plugin.

## B. Aggressive shared AEVUM engine optimization

Implemented a feature-gated **prepared-multiply -> retained-width bridge** across `third_party/aevum/src/Gpu.{h,cpp}` and `EngineApi.cpp`.

For compatible non-PFA short-carry plans, a prepared multiply can consume and produce the same retained width-transform state used by the existing lead-cache square chain. This can remove the canonical `fftW/carryA/carryB -> fftP` materialization round trip between a prepared multiply and a following square. Observable API operations (read/copy/add/sub/checkpoint/sync) still force canonical materialization.

Safety gates:

- only factor-1 prepared multiplication with distinct source/destination enters the bridge;
- PFA/PFA9 is excluded;
- unsupported/long-carry plans reject the bridge;
- `AEVUM_PREPARED_MUL_LEAD=0|1|auto` provides an explicit gate;
- AUTO first performs word-exact differential validation, then a tiny A/B and selects only at the configured minimum gain (default 1.03x);
- mismatch, exception or regression automatically disables the candidate;
- bridge decision is stored in the same device/workload/exponent-scoped cache.

No new bridge performance number is asserted in this package. It is **pending real hardware validation**.

## Workload coverage

The runtime tuner receives real workload identity for PRP, LL, P-1, low-memory P-1, ECM and their Gaussian backend selections. The new prepared-width bridge specifically targets workloads that actually perform prepared multiplication: P-1 and AEVUM-backed ECM, including GM/GQ P-1 and GM/GQ ECM paths using the same AEVUM engine API. PRP and Gaussian PRP benefit from plan tuning/retained-square machinery, not from this prepared-multiply bridge. LL keeps the already validated AEVUM_FUSED_LL path unchanged and receives only regression validation here.

## Already-measured input gains (not new Pass 4 measurements)

The supplied range-sweep data showed plan-selection gains versus TRUE AUTO, including roughly 1.72x at 180–197M on RTX 3080 and roughly 2.35x there on Radeon VII by selecting the valid 4M Type4 plan `4:512:8:512:202`. These measurements motivate the runtime tuner; they are not measurements of the new shared bridge.

Existing AEVUM_FUSED_LL input measurements are preserved: 1.1880x engine throughput on RTX 3080 and 1.1127x on Radeon VII/gfx906. Pass 4 does not re-claim or extrapolate them.

## Validation completed locally

Passed without target GPU hardware:

- Pass-4 source audit and workload/backend policy audits;
- runtime cache serialization, replacement, invalidation and OFF/AUTO/FORCE-RETUNE parsing;
- issue #36/native-tune compatibility source tests;
- host arithmetic/state/OpenCL-source tests;
- Type4/PFA9 plan/source guards;
- Apple OpenCL 1.2 kernel syntax matrix and retained compatibility tests;
- host-only AEVUM auto-policy test using a device-neutral resolver stub;
- compilation of modified `Gpu.cpp`, `EngineApi.cpp`, `RuntimeAutotune.cpp`, `AutoPolicy.cpp` and `EngineAevum.cpp` translation units.

The sandbox lacks the OpenCL development linker library (`-lOpenCL`), so final shared-engine linking and GPU execution are intentionally deferred to the two target commands. This is an environment limitation, not a source compile failure.

## Hardware validation artifact

`scripts/bench_aevum_pass4.sh` builds the production engine, runs host/source gates, then validates representative exponents near 21M, 100M, 150M, 180M, 197M and 210M. It checks TRUE AUTO, supplied seed, first-use FORCE-RETUNE, cache hit, prepared-width bridge A/B, kernel profiles, and FUSED_LL regression.

Each run creates a compact `tuning-output.zip`. Transform-sized residues are reduced to SHA-256 hashes. `summary.json` records GPU identity, workload, exponent, default/AUTO plan, tuned plan, candidate median timings and rejects, correctness, cache decision, bridge decision/A-B data, and kernel profiling when available. Per-case logs and cache records are retained for Pass 5.
