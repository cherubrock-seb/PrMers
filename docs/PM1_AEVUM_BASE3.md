# Experimental Aevum base-3 P-1 exponentiation

**IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED**

Baseline: PrMers `e501da30bab163bf4a8575dbf8efb8660b9e5868`.
Scope: the normal binary P-1 stage-1 loop in `src/modes/RunPM1.cpp`.
Enable with `PRMERS_PM1_AEVUM_BASE3=1`; unset or `0` retains the existing path.
Other values are rejected. This does not modify the embedded Aevum engine.

## Cost removed and exactness

For a set exponent bit, the Aevum path currently issues:

```
square(state)
prepare(temp, base)
multiply(state, temp)
```

During the first, single-chunk exponentiation the base is the integer 3.
The candidate instead issues `square_mul(state, 3)`; a zero bit still
issues `square_mul(state)`. Both compute exactly
`state <- state^2 * 3^bit (mod 2^p - 1)`.

This removes one complete modular multiplication and its preparation at every
set exponent bit. It uses the existing small-factor scaling implementation;
the source does not assume a fused square/carry kernel. The general compiler
cannot infer the constant base across opaque engine calls and register contents.
The existing Marin fast3 path is unchanged; Aevum previously excluded itself
unconditionally.

The target is long, single-chunk P-1 stage 1 on RTX 3080 CUDA/OpenCL and
Radeon VII OpenCL, with the existing normal or three-register low-memory
layout. The potential saving depends on exponent-bit density, scaling cost,
lead-cache behavior and checker overhead. No speedup has been measured.
There is no claimed improvement to ordinary PRP or stage 2.

## Eligibility and safety

The candidate requires Aevum, the explicit flag, `firstChunk`, and
`nextStart == 0`. Multi-chunk exponentiation, later chunks, true B1 delta
extensions and the one-register ultra-low-memory path are unchanged.
Later chunks and extensions use arbitrary bases and must not substitute 3.

Before entering the live loop, a scratch register is set to 1 and squared
with factor 3. This exercises the plugin's existing capacity guard without
changing the live result or base. An unsafe explicit FFT or device failure
propagates as an error; it is not silently converted into a fallback or a
different plan. The resulting canonical 3 must equal RBASE, including on
resume. A non-three restored base selects the original general path.

The Aevum Gerbicz-Li checker continues to use square plus prepared multiply.
Both full-block and partial-block mismatch paths disable the specialization
before existing recovery; tail verification likewise uses the general path.
Existing retry/error-counter behavior is retained. With Gerbicz-Li disabled
by an existing low-memory option, no new checker is implied.

The register counts and representations, bit/chunk counters, checkpoint format,
GCD, stage-2 handoff and external resume exports are unchanged.
Scratch/prepared-register bytes may differ. Compare canonical live state and
the complete exported X, not raw checkpoint bytes or only res64.
OFF-to-ON and ON-to-OFF resume require validation on the same backend.

FFT selection, manual overrides, tune-file precedence, device mapping,
backend selection, cache settings and production defaults are unchanged.
The candidate does not rely on `engine::max_multiplier()`, whose inherited
default is not an Aevum capacity query. The plugin's actual factor guard is
the authority. Set the flag to 0 or unset it for the established schedule.

Aevum #261 optimizes cached preparation for general multiply chains; this
candidate removes the base-3 multiply itself in a narrower P-1 domain.
Validate them separately first, then test their combination if both qualify.
`AEVUM_REG_SCALE3` is a separate engine flag: hold it fixed in each A/B pair.
No active engine branch is modified by this change.

## Pending validation

No build, test, GPU job or benchmark has been executed for this candidate.
Keep at least 15 GiB free on downstream validation hosts.

1. Build normally and run `make test-pm1-aevum-base3`.
   The host test exercises the production helper and eligibility predicate:
   invalid flags, composite/prime Mersenne rings, zero and upper-bound residues,
   capacity exceptions, non-three bases, untouched registers, binary
   exponentiation against an independent right-to-left oracle, and canonical
   state transitions between generic and specialized steps. It is not GPU
   evidence or a real checkpoint-serialization test.
2. Run the retained-evidence GPU harness separately for RTX 3080 OpenCL,
   RTX 3080 CUDA and Radeon VII OpenCL, using the appropriate device:

   ```sh
   python3 tests/pm1_aevum_base3_gpu_test.py --binary ./prmers \
     --backend opencl --device 0 --work-dir /path/to/new-evidence-directory
   ```

   Use `--backend cuda` for CUDA and optionally `--fft SPEC` for an explicit
   legal factor-3-safe plan. The harness requires the actual reported backend,
   compares matching FFT plans and full exported residues for M269 at B1=100
   and B1=2141, injects an error and checks detection/fallback/recovery, and
   checks the three-register low-memory path. Logs and exports are retained.
   Timeouts, missing exports, silent backend fallback and mismatches fail.
3. Run existing checkpoint and handoff tests with both flag values:
   `make test-pm1-stage1-ckpt test-pm1-ckpt-v3-resume`.
   Also run `make test-pm1-stage1-checklevel test-pm1-lowmem-stage1-ckpt
   test-pm1-lowmem-extend test-pm1-extend-ckpt test-pm1-small-items`.
   Force and verify the actual Aevum backend for affected cases; some existing
   scripts select other engines or accept broader output.
4. Validate OFF-to-ON/ON-to-OFF interrupted resume before/after a block boundary,
   both checkpoint versions, fallback from .old, completed stage-1 checkpoints,
   altered B1/-maxe, and a valid-CRC non-three RBASE fixture. Require the same
   final full X and factor result. Verify invalid flags, unsafe explicit FFT
   rejection before the live bit loop, multi-chunk and B1-extension exclusion,
   and unchanged Marin/ultra-low-memory behavior.
5. Short screen: p=21000029, B1=20000, stage 2 disabled, identical explicit
   factor-3-safe FFT, checklevel, backend and engine flags. Require identical
   full X, check outcomes, result and selected plan. Record the final live
   register words at matching boundaries as well. If promising, extend to
   B1=200000 and another legal FFT. Preserve checkpoint/factor evidence.
6. Measure at least three counterbalanced paired A/B runs (AB, BA, AB) of
   complete stage-1 operations, with A=0 and B=1. Include exponent construction,
   verification, export and GCD time; report stage-1 arithmetic separately.
   Require at least **3% repeatable end-to-end improvement for KEEP**.
   Record multiplication/preparation and launch counts, kernel timings and any
   unexpected register/scratch/occupancy changes. Do not count fallback runs
   as specialized performance results. Record a complete ordinary PRP control
   to guard unaffected behavior; P-1 timings do not establish PRP parity.
