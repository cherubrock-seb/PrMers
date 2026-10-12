IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

# S2: replace four root-8 modular products with shifts

Baseline: 6ccd782528c3d8d14789b47ae5e3c5ca19e3f615.
Enable with process environment `PRMERS_NTT_ROOT8_SHIFTS_V1=1`.
Only the exact string "1" adds the define; kernel default is zero.

Scope is only `kernel_ntt_radix4_radix2_square_radix2_radix4`, the
existing eight-word forward/square/inverse tail of the legacy NTT.
Neither pipeline selection nor its LDS representation changes.

## Guarded constants and deleted work

The experiment is active only when all four existing host-supplied
twiddle constants match:

| Constant | Field value | Replacement |
| --- | --- | --- |
| W15_01_Y | -2^24 | modMul2Pow24, then modular negation |
| W15_2_Y | -2^72 | prmersMul2Pow72, then modular negation |
| WI15_2_X | -2^48 | modMuli, then modular negation |
| WI15_01_Y | +2^72 | prmersMul2Pow72 |

These are plan/table invariants, not exponent-specific assumptions.
The table generator uses powers of the eighth root. The existing tail
already hardcodes the +2^48 forward and +2^24 inverse twiddles.
Nevertheless, activation checks the actual four build definitions;
a different root orientation or missing constant retains the original
four modMul calls.

Four generic 64x64 modular product paths disappear per eight-word tail.
Each generic path expresses a low product, mul_hi and Reduce. They are
replaced by fixed shifts and modular corrections; three modular
negations remain explicit. The two +2^72 replacements use one modSub
each rather than a general multiply and two-part reduction.

This is a source cost ledger, not an instruction-count or speedup claim.
The compiler may already partially specialize constant multiplication;
owner inspection must establish the actual deletion. Extra correction
dependencies may erase a benefit.

## Arithmetic derivation

Let B=2^32 and p=B^2-B+1. Then B^2=B-1 and B^3=-1 modulo p.
In particular 2^72=2^40-2^8 and 2^96=-1 modulo p.

For any unsigned 64-bit x, write x=L+H*2^24 with 0<=L<2^24 and
0<=H<2^40. Then:

    x*2^72 = L*2^40 - (L*2^8 + H)  (mod p).

The first operand is at most 2^64-2^40, below p. The second is at
most (2^32-256)+(2^40-1), also below p. All shifts and the addition
fit unsigned 64 bits. The existing modSub therefore returns the
canonical difference: for underflow it subtracts p's complement
2^32-1 from the wrapped result, which is equivalent to adding p.

The existing modMul2Pow24 and modMuli helpers already compute the low
and high words of x*2^24 and x*2^48 via shifts, then use the same
reduction formula as modMul. They are reused unchanged.
modMulPminus1 maps a canonical y to p-y, mapping zero back to zero.

The generic Reduce used by the reference returns canonical residues:
write hi=h1*B+h0. Its first fold adds h0*B to lo with one p correction.
The only possible result >=p after that correction has h0=B-1;
subtracting h0+h1 then places it below p. Otherwise the subsequent
subtraction either stays nonnegative below p or uses its existing
underflow correction. Thus equal field values have the same final
representative, not merely equal congruence classes.

The four substitutions preserve the same twiddle field elements.
Squares, other butterflies, weights, carry operations and output
permutation remain untouched. This is a static derivation; it does
not constitute device exactness validation.

## Owner arbitration

1. Confirm all four guard comparisons and target-tail coverage on the
   exact compiled plan. A request log alone does not prove activation.
2. Compare each substituted product with original modMul and an
   independent integer oracle. Cover 0,1,p-1,p,2^64-1, boundaries of
   bits 8/24/32/40/48, both sides of modSub underflow, sparse bits and
   deterministic random values. Include a limb grid for L/H extremes.
3. Compare every output word of the complete eight-word tail on hostile
   field inputs; then compare full legacy iterations/proof operations.
   Check supported power-of-two and radix-5 pipelines that use this tail.
4. With one twiddle changed, missing defines, knob off, and pipelines
   using other tails, verify fallback and unchanged artifacts.
5. Inspect multiplication/reduction instructions, registers, spills and
   LDS, then alternate tail and full-iteration timings on both GPUs.
   Reject inactive coverage, identical code generation or regression.

No compilation, execution, benchmark, SSH or Actions dispatch was run.
This source experiment is independent of P2 mask reuse, K1 proof
spectrum retention and F2 private tiles. Aevum F1 remains pending.
It does not revisit rejected P1 carry dispatch or S1 residue conversion.

Next independent lead: check whether FFT3161 CRT's fixed M31 factor can
be combined with immediate word extraction to delete an entire wide
operation. Do not broaden this root specialization before arbitration.
