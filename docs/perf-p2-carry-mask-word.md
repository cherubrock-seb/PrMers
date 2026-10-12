# P2: one packed mask word per aligned legacy carry block

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline: `eb5cb944b7c16a61554e319f7eea389afa2cf7de`.
Opt-in: environment `PRMERS_CARRY_MASK_WORD_V1=1`; default zero.
Independent branch from main; does not include P1 #198.

## Deleted work and scope

The three mask-consuming legacy carry kernels use a generic sliding window.
They load two initial mask words and maintain lookahead, wrap predicates,
selects, and offsets after every four digits.

When depth D is one of 4, 8, 16, 32, 64, one right-shifted mask word suffices.
The candidate has one source mask load per work-item reaching initialization
and one right shift per quad. It does not change D, carry arithmetic, digit-width
dispatch, memory order, launch count, early exits, or the final unmasked digit.
The MUL3 per-quad normalization reset and outgoing-carry correction are intact.

The compiler may already remove unused final updates and fold some wrap logic.
Thus the source load count is not a claim about emitted loads or GPU traffic.
The mask working set is small; latency, registers and end-to-end benefit remain
unmeasured. Owner codegen inspection must establish the actual deletion.

## Range proof

D divides 64, and start = gid * D. Hence off = start mod 64 is a multiple of D,
0 <= off <= 64-D. Every consumed digit index has displacement j < D, so
off+j < 64. The nibble at quad k is exactly bits off+4k through off+4k+3
of maskPacked[start/64]. No consumed nibble requires the next word.

Initialize merged = maskPacked[start/64] >> off. After k quad updates it equals
that word shifted right by off+4k. Each shift is by four, never by 64.
For D=64, off=0; the final consumed nibble starts at bit 60.
For the second carry kernel, the final quad uses the same current merged value.
For D=4 the loop is skipped and the initial value supplies that final quad.
An early return can only remove a suffix of the same mask sequence.

The original sliding window remains verbatim in the inactive preprocessor
branches. Depths outside the five values, including 1/2, 12/24/48 and >64, use
the original source. The general multiplication-by-base kernel is untouched.

## Owner arbitration

No build, preprocessing, execution or timing was performed during authoring.

1. Confirm the selected backend is legacy and record D, source head and build
   flags. Compare with the same plan/depth, toggling only this option.
2. For every supported depth, cover all reachable offsets and several adjacent
   mask words. Include zero/one, alternating, sparse and random masks; change
   neighboring words to expose accidental lookahead dependence.
3. Compare all digits and carry-array entries for ordinary carry, MUL3 and the
   second pass. Include signed-residue bit patterns, wraparound sums, long
   carry propagation, zero/nonzero incoming carry and final unmasked digits.
4. Cover inactive depths and verify baseline-equivalent preprocessing/codegen.
5. Inspect actual mask loads, integer instructions, registers and spills on
   RTX 3080 and Radeon VII. Reject if the compiler already emits identical code.
6. Only after exact full-word validation, use counterbalanced carry-kernel and
   full-PRP timing at fixed depth/plan. No selector changes.

## Campaign resume

P2 is distinct from P1's arithmetic dispatch change. The two-value-width
specialization previously suggested in A6 is deferred: replacing variable
shifts with extra selects has no established cost advantage.

A2 #9, C1 #10 and A3 #11 were rejected for performance; A4 #12 had no codegen
effect. Their mechanisms are not repeated. A1 #8, D1 #13, A5 #14, A6 #15 and
P1 #198 remain implemented pending owner arbitration. Aevum #7 and the
validated Type2 lane #16 / PrMers #199 are overlap guards.

Next: FFT3161 arithmetic or representation conversion with a provable removed
operation. The PR body is the authoritative resume block and exact head pointer.
