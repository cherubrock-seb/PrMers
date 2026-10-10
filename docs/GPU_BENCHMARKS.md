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

Bench3 will later enumerate legal alternative production plans, perform explicit
exactness classification and report AUTO-versus-best deltas.
