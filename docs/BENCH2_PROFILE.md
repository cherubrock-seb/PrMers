# Bench2 profiling sidecar

Bench2 timing stays on `bench2.v1`. Profiling runs in a separate process and
writes `bench2.profile.v1`, so diagnostic events and resource queries never
contaminate throughput samples.

The sidecar consumes a completed `PASS` Aevum `.record`, reuses its exponent,
device, register count and active FFT plan, enables `AEVUM_PROFILE_KERNELS=1`,
verifies the created plan and transform, performs a short diagnostic square
sequence, calls `aevum_engine_profile_report()`, parses `AEVUM_RESOURCE` and
`AEVUM_PROFILE`, and writes JSON plus the raw native log.

Generic fields are kernel name, workgroup size, local/private memory, preferred
workgroup multiple, OpenCL query return codes, profile call count and execution
nanoseconds. OpenCL private-memory bytes are not a register count.

Architecture-specific fields are reserved but remain `not_collected` until
trustworthy direct binary metadata is available. NVIDIA register/spill data must
come from direct PTX/cubin evidence; this sidecar does not reassemble extracted
PTX. AMD VGPR/SGPR and segment sizes must come from ROCm code-object metadata.
Architecture enrichment is non-fatal.

Example:

```bash
python3 scripts/bench2_profile.py \
  --record bench2-results/records/d2_p0210000017.record \
  --engine-lib third_party/aevum/build-engine/libaevum_engine.so \
  --tune-dir third_party/aevum \
  --out bench2-results/profiles/d2_p0210000017.profile.json
```

The sidecar never modifies Bench2 timing records, aggregate timing outputs, or
the standard grid, and it never invokes Nsight Compute on the OpenCL path.

Initial validation points are `p=210000017` on RTX 3080 device 2 and Radeon VII
device 0.
