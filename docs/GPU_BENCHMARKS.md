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

Setup/JIT/backend probing is excluded from steady-state timing. Each timed batch
ends with `engine::sync()`.

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
