# K1: reuse a base spectrum in legacy proof exponentiation

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `eb5cb944b7c16a61554e319f7eea389afa2cf7de`.
Opt-in environment: `PRMERS_POW_REUSE_FORWARD_V1=1`; all other values are off.
Scope: NttEngine::powInPlace, called by Proof.cpp and ProofSet.cpp for hash-sized
exponents. This is a proof-work optimization, not a PRP squaring-loop claim.
The branch is independent of P1/P2, Aevum and Type2 host changes.

## Deleted phase

At a nonfinal set bit, the existing right-to-left power loop performs:

1. Multiply A by B: copy A/B, forward A/B, pointwise multiply, inverse A,
   two-pass carry A, copy A back.
2. Square B: allocate/copy B again, forward B again, pointwise square,
   inverse B, two-pass carry B, copy B back.

The candidate retains the first B spectrum after the multiply. The multiply's
second argument is read-only. It then squares that same spectrum directly.
Both inverse/carry sequences and both output copies are retained.

| Per eligible multiply-square pair | Baseline | Candidate |
| --- | ---: | ---: |
| Complete forward_simple pipelines | 3 | 2 |
| Complete inverse_simple pipelines | 2 | 2 |
| Pointwise kernels | 2 | 2 |
| Carry kernels | 4 | 4 |
| Full-residue copies | 5 | 4 |
| Temporary allocations/releases | 3 | 2 |
| Peak pair temporary buffers | 2 | 2 |

If the forward pipeline has F stages, the host deletes F kernel enqueues per
eligible bit, plus their stage traffic and arithmetic. The repeated B copy
also disappears. These are control-flow counts, not measured timing gains.

For positive exponent e, the number of eligible bits is popcount(e)-1: every
set bit except its most significant bit has a following square.
Examples: e=0/1/2/4 has zero pairs, 3/5/6 has one, 7 has two,
and 0xffffffffffffffff has 63. There is no benefit for powers of two.

## State and ordering argument

The loop invariant is A * B^e = original_base^original_exp modulo 2^E-1.
For odd e>1 the original step updates A'=A*B, B'=B^2, e'=(e-1)/2.
The candidate performs the same updates, then right-shifts e and continues.
The untouched path handles zero bits and the final set bit. The early returns
for exponents 0 and 1 are unchanged.

The stronger representation argument does not rely only on modular equality:
the same forward_simple pipeline reads the same B word array. Its exact output
already exists in tmpB, and pointwiseMul(tmpA,tmpB) writes only tmpA. Squaring
tmpB thus starts from the same spectrum as the deleted recomputation.
The inverse_simple and carryGPU sequences are identical for both results.
No conversion to a different canonical form is substituted.

NttPipeline.hpp setStageArgs2 binds only BufX to the selected temporary;
twiddles and digit-weight tables remain the original inputs. The existing
Context queue is in order. It orders reads of tmpB before its square and
both carryGPU uses of blockCarryBuf. No barriers/events/finishes are added.

The public result and base may alias: powInPlace already copies the base into
a private base_copy_buf and holds a separate private accumulator. The new
temporaries are allocated independently of both. No retained spectrum escapes
this iteration or call, and no cache invalidation protocol is introduced.
The existing final copy remains the public canonicalization boundary.

## Resource and failure limits

Two pair temporaries are live, matching the existing multiply's peak. tmpB's
transform-domain lifetime is extended across accumulator inverse/carry work;
tmpA is also retained until the pair finishes. This can alter allocation reuse
and memory residency despite the unchanged peak buffer count.

Both allocations are checked. If the second fails, the first is released.
Copy, stage, pointwise and carry failures propagate through the existing error
helpers; the pair releases both temporaries, then pow's outer handler releases
its private base and accumulator. The public result is not updated mid-pair.
Do not silently fall back after partially enqueuing a pair.

This reuses existing queue/thread-safety assumptions; it does not make one
NttEngine instance safe for concurrent kernel-argument mutation.

## Owner validation

No build, execution, benchmark or CI inspection was performed while authoring.

1. Toggle only the environment option on the same legacy proof context. Assert
   the path is reached; Aevum/Marin timing cannot arbitrate this candidate.
2. Compare all output words for exponents 0,1,2,3,4,5,6,7, powers of two,
   alternating bits, highest-bit-only, random hashes and UINT64_MAX.
   Test result==base and distinct buffers; retain a copy to check base preservation.
3. Cover supported power-of-two and factor-five simple pipelines, several
   Mersenne exponents and input residues including zero, one, near modulus,
   boundary digit patterns and random valid saved proof words.
4. Cross-check canonical integer residues against an independent bigint modular
   power. Also require dense baseline equality, not only a short residue hash.
5. Run proof construction/reduction and proof verification on identical saved
   inputs; compare intermediate words, proof hashes/files and final acceptance.
6. Inject allocation, copy and enqueue failures at pair stages; confirm error
   propagation, resource cleanup, unchanged public inputs and a clean retry.
7. Trace launches/copies: each eligible pair must omit exactly one complete
   forward pipeline and one full-buffer copy. Record actual memory high-water.
8. After correctness gates, use counterbalanced pow-call and complete proof
   timings on RTX 3080 and Radeon VII. Include sparse exponents with no saving.
   Main PRP iterations/second are not the relevant metric.

## Campaign resume

P2 PrMers #200 deletes aligned mask lookahead; S1 Aevum #17 shares the first
normalized word's GF31/GF61 encoding. Both are independent, off by default and
await owner validation. K1 does not include their source changes.

Closed owner dispositions: A2 #9 REJECT_PERFORMANCE (~-4.8% RTX);
C1 #10 REJECT_PERFORMANCE (noise); A3 #11 REJECT_PERFORMANCE (extra LDS);
A4 #12 REJECT_NO_CODEGEN_EFFECT. Do not reopen these mechanisms.
Existing A1 #8, D1 #13, A5 #14, A6 #15 and P1 #198 must not be repeated.
Overlap guards: Aevum #7, owner FFT323161 CRT/i128, Type2 #16 / PrMers #199.

A targeted upstream comparison found FFT3161 CRT already equivalent in
gwoltman/gpuowl. The inspected M61_SM120 optimization targets a different GPU
generation and explicitly reports no modM61 change on older devices; it was
not ported. The current FFT323161 CRT changes remain in the owner's lane.

Next: small FFT3161 inverse-to-carry fusion only with an identified arithmetic
or materialization deletion and a complete range proof. Do not broaden S1 to
the second word or manufacture a lifetime variant. The PR body carries the
exact current head and the authoritative PERF_CAMPAIGN block.
