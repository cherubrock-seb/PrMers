IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

# F2: private intermediate tile in the legacy forward two-step NTT

Baseline: 6ccd782528c3d8d14789b47ae5e3c5ca19e3f615.
Enable with process environment `PRMERS_NTT_PRIVATE_TILE_V1=1`.
Only the exact string "1" adds the kernel define. Unset, empty, "0" and
other values preserve the original build options. Kernel default is zero.

## Mechanism and scope

Only `kernel_ntt_radix4_mm_2steps` changes. It combines two forward
radix-4 stages and currently uses a local-memory tile for values produced
and consumed by the same work-item. The experiment replaces its two LDS
declarations with `__private ulong local_x[16]`.

The first weighted stage, inverse kernels, other tails, pointwise
multiplication, carry kernels, stage selection and launch dimensions are
unchanged. Both existing unroll directives and every arithmetic
expression, twiddle load and global address remain unchanged. This is a
legacy NTT experiment, not an Aevum, Marin or AUTO-selector change.

The forward pipeline selects this middle kernel for eligible m>=64,
m divisible by 4. Small pipelines that do not use it cannot benefit;
activation of the environment knob alone does not prove active coverage.

## Ownership argument

For local lane l, the original tile starts at 16*l. Its producer loop
p=0..3 writes indices 4*p+0 through 4*p+3 exactly once, covering [0,15].
The consumer loop pass=0..3 reads pass, pass+4, pass+8, pass+12, also
covering [0,15] exactly once. No address includes another lane's tile.
Both loops execute in the same work-item and producer execution precedes
consumer execution. The pointer never escapes and there are no barriers
in this kernel.

Thus the intermediate represents a transpose of one thread's values,
not inter-thread communication. Private ownership removes the LDS
representation; the arithmetic graph and access order within the tile
are retained. This is source reasoning only, not a correctness PASS.

## Concrete cost ledger and risk

At source level, each invocation of this kernel performs 16 ulong LDS
stores and 16 ulong LDS reads per work-item, or 256 bytes of LDS traffic.
The active source contains none of those LDS accesses and does not
declare the 128*LOCAL_SIZE2-byte shared array.

These are source-level deletions, not measured instruction or resource
reductions. A backend may have promoted baseline storage already. The
private tile may become 32 32-bit registers, be partially scalarized, or
spill; compiler allocation, actual local memory and occupancy are
unknown. Lower occupancy or spill traffic may overwhelm the benefit.
Reject identical generated code or production regression.

No kernel, global transfer, twiddle multiplication or barrier disappears.
The hypothesis is exclusively removal of the per-thread LDS round trip
between two already-fused stages. It does not reproduce Aevum directional
fusion #7 or the rejected addition of an LDS plane in A3.

## Owner validation

1. Use identical legacy plans, exponent, local sizes and depths OFF/ON.
   Confirm the pipeline contains the target kernel, including several
   eligible m values and repeated target stages.
2. Compare all words directly after the target stage using identical
   inputs/twiddles. Include zero, one, field boundary values, alternating
   and pseudorandom residues, multiple workgroups and different local
   sizes. Check each of the 16 output locations, not only RES64.
3. Compare complete forward/inverse pipelines and legacy LL, PRP/MUL3
   and proof operations byte for byte. Include power-of-two and supported
   radix-5 pipelines that enter the target stage.
4. Verify default/off and other kernels preserve baseline artifacts.
5. Inspect LDS reads/writes, static local bytes, registers/VGPRs, scratch
   and occupancy. Confirm the private array did not merely move the
   intermediate to spill memory.
6. Only after correctness, alternate isolated stage and full-iteration
   timings on RTX 3080 and Radeon VII. Report stage coverage with results.

No build, compilation, execution, benchmark, SSH or Actions dispatch has
been performed for this candidate.

## Campaign continuity

F1 is published independently as Aevum #19. P2 (#200) has an owner KEEP
disposition pending clean-main integration; no P2 changes are copied here.
K1 (#201) retains proof spectra across operations and remains independent.
P1's shorter carry dispatch regressed despite materially smaller code;
instruction count alone is not an acceptance criterion.

Aevum A1, A2, C1, A3, A4, D1, A5, A6, S1 and directional V2 are closed
negative evidence. Do not resurrect their storage, streaming, ordering,
barrier or conversion variants. Type2 is already integrated by the owner.

Next independent lead: inspect whether a plan-constant Goldilocks
twiddle can eliminate a full modular multiply; reject generic limb
rewrites and trivial compiler rescheduling. Do not broaden F1 or F2
before owner arbitration.
