# Pass 5 — PRP implementation profiles

Status: implementation complete; real GPU acceptance pending. No new GPU speedup is claimed.
Provisional alpha versions: PrMers `4.20.97-alpha-v100.12-aevum-pass5-prp-max`; AEVUM `v0.3.83-pass5-prp-max`.

## Kept implementation
- PRP implementation tuning runs **after** final shape selection. Explicit FFT, tune.txt, GB202 and shape-cache selections stay frozen.
- Implementation precedence: `AEVUM_PRP_USE` > matching compatible tune-entry `-use` > use cache > bounded measured search > defaults.
- `AEVUM_PRP_USE=KEY=VALUE,...` shares native `Args::splitUses` and Gpu's recognized keys, with additional safe numeric validation. An explicitly empty value freezes defaults.
- Optional tune.txt suffix: `-use INPLACE=1,MODM31=2`. Existing two-column entries retain their behavior.
- `AEVUM_PRP_USE_TUNE=off|auto|retune` inherits `AEVUM_AUTOTUNE` when unset. Manual profiles still take precedence over OFF.
- Separate atomic `<shape-cache-path>.prp-use-v2/<hash>.tsv` records leave the shape-cache implementation/format untouched. Existing v1 records remain shape-only.
- Keys include engine version, device/vendor/ordinal, driver/runtime/OpenCL C version, PRP/register class, exponent band, exact exponent, selected shape and implementation flags.
- Normalized sorted winners are cached. Only conclusive negatives persist as `defaults-complete`; incomplete searches are DEFERRED and retry on AUTO. Ambiguous old defaults are ignored; prior positive records remain usable.
- NVIDIA first compares defaults against the complete reported vector: `INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0`.
- Six additional grouped ablations/refinements use upstream knobs. AMD uses upstream's portable memory-policy digits; the NVIDIA vector is never enabled by GPU name alone.
- Dense deterministic word differential and ROE gate precede screening. Before confirmation, both newly built engines execute the full 128-square normal sequence untimed and synchronize. Three alternating measured pairs then use the unchanged gates.
- Automatic acceptance requires median gain >=3%, at least two pairs >=2%, and no pair regressing >1.5%. Every timed pair also compares final words.
- At most seven distinct candidate profiles; after expensive shape tuning the cap is four. Cooperative implementation budgets are 5/8 seconds with a combined 12-second target. A blocking OpenCL compile can exceed this target; incomplete confirmation is never accepted.

## Bounded structural attempt
- `AEVUM_PRP_CARRY_EPOCH=1` is an **off-by-default experiment**, absent from automatic candidates.
- Type1/Type4 carry readiness uses iteration tags to remove consumer flag-reset global stores. Carry arithmetic, producer/consumer fences and transform stages are unchanged.
- Epoch wrap and transitions into legacy LL/multiply signaling clear readiness state. LL/MUL3 kernel ABI remains unchanged.
- Both supplied real GPU campaigns rejected carry epoch; keep it OFF. It removes signaling stores, not an entire FFT data pass. No retry is included in focused validation.
- No second structural hypothesis was attempted without GPU evidence.

## Preservation / rejected work
- Shape autotuner, atomic shape cache, issue #36/manual routing, GB202 profile, Type4 boundary and prepared-multiply bridge remain intact.
- FUSED_LL remains enabled as before; no LL optimization was attempted.
- FUSED_MUL3 remains disabled because of prior p=21000029 word mismatches. PRP_MIDDLE1 stays experimental/off.
- Prior carry_batch1/4, GF61 limb32 and buffer-argument caching experiments were not repeated.
- The campaign rejects compile/word/acceptance failures, removes the affected use-cache entry, and continues. It preserves other exponents' cache entries.

## Validation completed here
- Engine shared-library build; existing host arithmetic/state/OpenCL policy/Type4/PFA/shape-cache tests passed.
- New profile parser/cache/tune-entry tests and protected-source checks passed.
- Campaign tests cover inherited manual-environment isolation, cache eviction and continuation after mismatch, and actual old/new kernel profiling.
- Benchmark C++ build and shell syntax passed. Binary recovery patch applicability passed in a disposable reference copy.
- No GPU platform exists here (`CL_PLATFORM_NOT_FOUND_KHR`). GPU exactness, engine gain, kernel gain and end-to-end gain are **unmeasured** for Pass 5.
- OpenCL syntax matrix needs Clang (unavailable here). Full PrMers/adapter build needs GMP/OpenCL development headers (unavailable here). The Ubuntu commands install these; the campaign records build/test failures.

## Hardware validation
- HWFix: run `AEVUM_TARGET=rtx3080 AEVUM_DEVICES=2 bash scripts/bench_aevum_pass5_hwfix.sh` or `AEVUM_TARGET=radeonVII AEVUM_DEVICES=0 bash scripts/bench_aevum_pass5_hwfix.sh`. The older full campaign remains available but is unnecessary for these fixes.
- Matrix: 21000029, 70000001, 100000007, 147800003, 150000007, 180000007, 196999969, 197000003, 210000017. Includes a small CPU oracle and full word traces at p=21000029.
- The harness builds the preserved Pass-4 engine and current engine, then uses identical selected shapes, seeds, iteration counts and alternating medians.
- Startup JSON separates engine creation time from complete process wall time. Timed PRP throughput excludes creation, warmup and readback.
- Kernel profiling compares real old/new engines; the epoch comparison freezes the same implementation profile on both sides. Profiling is separate from engine timing.
- Runtime tests remove inherited manual overrides and never set AEVUM_TUNE_DIR. Tune-entry precedence is tested separately.
- `RTX5090_PASS5_TEST.md` runs the short reporter matrix, with strict 147.8M/180M/196999969 shape gates. No full PRP job is run before a measured engine win.
- Return each machine's printed `pass5-<target>-*/tuning-output.zip`: summary, measurements, compact logs, cache and kernel timings. Build directories and raw residues are excluded.

## Recovery
- `PASS5_RECOVERY.md`, `PASS5_STATUS.txt` and binary-capable `PASS5_WIP.patch` retain the exact state and next action.
- Run `python3 scripts/checkpoint_pass5.py` after updating recovery notes to refresh the patch/status and sibling recovery ZIP. It never resets current sources.
- With no Git metadata, the patch compares preserved pre-Pass-5 sources; it also includes relevant new source/scripts.

## Hardware follow-up / HWFix
- Supplied RTX3080 p=21000029 full-vector independent engine gain: **1.07555x**. Reduced memory profile internal estimate: **1.15150x**, still awaiting independent acceptance. No gain from the HWFix code itself is claimed.
- Supplied Radeon VII: no independently accepted use profile. Zero word mismatches on either campaign. Incomplete shape/use budget handling and cold confirmation were the demonstrated bugs.
- Fixes: explicit completed-positive/negative/deferred decisions; untimed normal-kernel warmup; independent x3 for every changed final warm-retune profile. No acceptance threshold changed.
- JSON separates `tuner_internal_gain` and `independent_engine_gain`. Rejected profiles are evicted; defaults need non-regression, not an artificial speedup.
- Focused RTX: p=21000029, reduced memory profile word trace and independent x3, final profile cache hit. Focused Radeon: p=180000007, fresh caches, deferred/no-record check when budget expires, next AUTO search, independent x3 and final cache hit.
- Typical focused runs use 12 RTX or 10 Radeon engine invocations; a bounded extra AUTO/warm retune may be needed if still deferred or a different profile wins. No structural or full PrMers campaign runs.
- New semantic C++ tests, 12 focused-flow tests, 3 existing campaign tests, 3 preservation/source checks, shell syntax and engine build passed. See PASS5_HWFIX_TESTS.txt. GPU effects of these fixes remain pending.
- Return only the printed `pass5-hwfix-<target>-*/tuning-output.zip` from each host.

## R5 `-use` autotuner (source-ready; GPU range validation pending)

R5 supersedes the older Pass-5/R4 implementation-timing description above; it
does not change the Pass-4 FFT-shape tuner.

- Up to 12 bounded hardware-aware `-use` profiles are screened; up to four
  finalists are confirmed. The search is resumable and a cold 6-profile slice
  cannot become a final cache decision before the full warm scope is searched.
- Screening and final confirmation use upstream `Gpu::timePRP()` rather than the
  former synthetic square-chain timing proxy.
- Acceptance remains exact words + ROE + median >=3% + >=2/3 wins >=2% + worst
  >=0.985. No known RTX profile is hardcoded as accepted.
- `prp-use-v3` separates R5 implementation decisions from stale R4 negatives.
- Hardware reporting now separates `R5 defaults -> R5 selected -use` from
  `Pass4 -> R5 final` non-regression at identical explicit FFT shape.
- Full default validation matrix: 21000029, 70000001, 100000007, 147800003,
  150000007, 180000007, 196999969, 197000003, 210000017.
- Current status: host/source guards pass; no R5 GPU performance result is
  claimed until new RTX3080/Radeon archives are returned.
