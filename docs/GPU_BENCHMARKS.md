# PrMers GPU benchmarks

## `-bench` versus `-bench2`

`-bench` is the historical transform-size benchmark and remains unchanged.

`-bench2` is a separate production-selector benchmark for ordinary Mersenne PRP.
It deliberately uses the same generic backend path as production:

- workload: PRP;
- register count: 8;
- backend configuration: automatic Marin/Aevum;
- engine creation: `engine::create_gpu(...)`;
- transform truth: `engine::get_size()`;
- backend truth: `engine::is_aevum_backend()`;
- instantiated Aevum plan truth: `aevum_engine_active_plan()` after creation.

A resolver preview is not treated as the executed plan because runtime autotune,
`tune.txt`, device profiles and cached production backend probes can change what is
actually instantiated.

## Modes

`-bench2-mode quick|standard|dense|full`

The modes use the same production selection path but increase the exponent grid,
warmup, timed iterations and repetitions.

The dense/full grid covers approximately 37M through 600M and includes explicit
neighbors around the validated ~197M selector boundary.

## Output

Use `-bench2-out PATH`.

Bench2 writes a versioned `bench2.v1` schema as:

- `bench2.txt`
- `bench2.json`
- `bench2.jsonl`
- `bench2.csv`

Each point records backend, actual plan, transform words/Mwords, bpw, setup time,
steady-state median/min/max timing, dispersion, iterations/s, estimated full-PRP
time, register/checkpoint memory, device/driver/runtime properties, PrMers
version, source SHA and compiler metadata.

The original `us_per_iter_*`, `iterations_per_second` and
`estimated_full_prp_seconds` fields remain the validated square-hot-path metric.
Setup/JIT/backend probing is excluded from that timing and each timed batch ends
with `engine::sync()`.

Bench2 also reports an additive production-PRP estimate:

- `gerbicz_block`
- `gerbicz_checkpasslevel`
- `gerbicz_full_check_interval`
- `gerbicz_boundary_us`
- `gerbicz_full_check_us`
- `gerbicz_amortized_us_per_iter`
- `production_prp_us_per_iter`
- `production_prp_iterations_per_second`
- `production_prp_estimated_seconds`
- `production_prp_probe_exact`

This uses the current ordinary PRP schedule from `RunPrpOrLlMarin.cpp`. For a
fresh default PRP in the campaign range, `B = 1000`; the cheap Gerbicz boundary
occurs every 1000 arithmetic iterations and the full replay/readback check is
amortized over 600000 iterations.

The full-check timing probe constructs the real fresh PRP register state,
advances to the first production boundary, executes the same boundary and
replay/readback/modulo-compare operation sequence, and requires that comparison
to pass. This is intentionally different from the older issue-36 timing helper,
whose historical `B = floor(sqrt(p))` assumption no longer matches fresh
production PRP defaults.

Proof-residue writes, periodic checkpoint files, progress/UI logging and an
explicit `-iterforce` synchronization policy remain separate components: their
cadence is proof-, wall-clock- or user-configuration-dependent and they are not
silently folded into the arithmetic estimate.

## Resume and interruption

Completed exponent/device points are committed as individual files under
`records/` using a temp-file + rename operation. Aggregated JSON/CSV/text files
are rebuilt from those completed records.

A stopped run therefore resumes without retiming completed points. Use
`-bench2-no-resume` to discard the point records for the selected output path.

## Correctness and profiling

Tranche A intentionally labels timing records:

- `exactness_state = NOT_VALIDATED_IN_BENCH2_TIMING`
- `profile_state = NOT_COLLECTED_IN_TIMING_PHASE`

Timing must not be contaminated by profiler collection.

The next bench2 tranche adds architecture-specific resource/profiler adapters
(NVIDIA PTX/SASS/ptxas where supported and AMD ROCm/code-object metrics), while
keeping profiler runs separate from throughput measurement.

Bench3 is delivered below as the public AUTO-versus-legal-plan benchmark.


## `-bench3`

`-bench3` is the public production-plan comparison benchmark. It starts from
the exact production AUTO backend/selector result and compares it with a bounded
set of legal explicit Aevum alternatives:

- power-of-two only: `pow2:auto`;
- native PFA auto: `pfa:auto`;
- PFA3: `pfa:3`;
- validated fixed PFA7: `pfa7:1:512:7:512:202`;
- PFA9: `pfa:9`.

Every explicit request is first passed through the Aevum resolver. Unsupported
plans are reported as `SKIPPED_ILLEGAL`, and concrete plans already represented
by AUTO or another request are reported as `SKIPPED_DUPLICATE`.

Bench3 performs a deterministic full-residue comparison against AUTO before a
candidate is timed. A candidate with a different full residue is classified
`EXACT_MISMATCH` and is not accepted as a performance result. The output stores
a stable residue fingerprint together with the explicit exact-match flag.

Steady-state square timing uses the same engine API as production and each
timed batch is synchronized. Bench3 then adds the same production PRP Gerbicz
boundary/replay timing model used by Bench2. `delta_vs_auto_pct` is based on the
production-representative per-iteration estimate; positive values are faster
than production AUTO.

### Modes and resume

`-bench3-mode quick|standard|dense|full`

Quick and standard deliberately use bounded exponent grids so a public Bench3
run remains practical. Dense/full expand coverage and repetitions.

Use `-bench3-out PATH`. Bench3 writes:

- `bench3.txt`
- `bench3.json`
- `bench3.jsonl`
- `bench3.csv`

Completed exponent/device points are stored atomically in `records/`. Interrupted
runs therefore resume without repeating completed points. Use
`-bench3-no-resume` to clear the selected output's completed point records.

Bench3 also checkpoints individual plans under the point's `partial/`
directory. Each plan runs in an isolated worker process and emits flushed
`BENCH3_PROGRESS` markers while setup, exactness, warmup, timing repetitions,
and the production probe advance. Successful plan checkpoints can therefore be
reused after an interruption without rerunning earlier plans. The authoritative
AUTO exact residue is persisted alongside the AUTO checkpoint so resumed
alternatives are still compared against the same exact reference.

Use `-bench3-plan-timeout N` to bound one isolated plan worker to `N` seconds.
The default is 300 seconds; `0` disables the timeout. While an isolated worker
is still running, the controller prints a `BENCH3_PLAN_HEARTBEAT` every
10 seconds with the most recently published worker phase, total elapsed time,
and the remaining timeout. This makes slow setup/JIT, exactness, timing, or
production-probe phases visible instead of looking hung.

A timed-out plan is reported as `TIMEOUT`; worker/runtime failures are reported
as `ERROR`. Either status keeps the exponent point partial and retryable rather
than publishing a completed `records/` entry. On a normal resume, existing PASS
checkpoints are reused while TIMEOUT and ERROR checkpoints are deliberately
rerun. On Windows, process exit `0xC000013A` is recognized as a console-control
interruption and stops Bench3 as an interruption rather than misclassifying the
worker as an ordinary plan error.

Plan checkpoint contexts are bound to the Bench3 schema and source SHA, mode,
exponent, measurement counts, device index, OpenCL vendor/device name, driver
version, and runtime version. A mismatch in any of those fields invalidates the
checkpoint instead of reusing measurements from a different device mapping,
driver, runtime, build, or benchmark configuration.

Bench3 is a benchmark and classifier only. It does not rewrite production tune
files or selector policy.
