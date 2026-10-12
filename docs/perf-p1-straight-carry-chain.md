# Straight four-digit carry chain for the legacy OpenCL engine

IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED

Baseline PrMers: `eb5cb944b7c16a61554e319f7eea389afa2cf7de`.
Kernel implementation: `c196aa9a9c2b166db67241ca882f61dde3f5f4e6`.
Opt-in wiring: `c35cbd20d9075682a318744c35ef90a8f8e3ace5`.
Unrun differential fixture: `645bb04ba96200a061ce2aae2919a06e5dccf188`.
Campaign sequence P1, following Aevum A1/A2/C1/A3/A4/D1 (PRs #8–#13).

## Selection

Set the environment variable `PRMERS_CARRY_STRAIGHT_V1=1` before
creating a legacy OpenCL Program. Only the exact string 1 appends
`-DPRMERS_CARRY_STRAIGHT_V1=1` and emits an experimental-path notice.
Absent, empty, 0 or any other value leaves production JIT options
unchanged. The kernel macro also defaults to 0.

The scope is kernels/prmers.cl and its legacy Program builder.
Aevum (including the vendored copy), Marin, backend selection, work
sizes and all defaults remain unchanged. A run selecting Aevum or
Marin cannot establish a performance result for this candidate.

## Independent hypothesis

digit_adc4 currently classifies each four-digit width vector as:
all equal, first three equal, two adjacent pairs, alternating pairs,
or generic. Each case then computes the same dependent carry chain.
The candidate removes this pattern dispatch and directly spells out
the four unsigned additions, masks and shifts.

This targets predicate/control state and potential divergence between
work-items with different width patterns. It does not change the
four-digit batch, local propagation depth, two-kernel carry protocol,
or inter-worker carry direction. It is not the rejected carry_batch1/4
experiment, cache-buffer-args or an Aevum streaming variant.

All callers of digit_adc4 use the selected helper, including plain,
multiply-by-3, carry follow-up and multiply-by-base kernels.
digit_adc4_last retains its original unmasked final component.

## Equivalence argument

For component j, let s_j=(lhs_j+c_j) mod 2^64. Both paths return
r_j=s_j & ((1UL<<w_j)-1) and pass c_(j+1)=s_j>>w_j onward.
The original specialized branches substitute w0 for w1/w2/w3 only
under their equality predicates. They therefore evaluate the same
width at every component as the straight path. The incoming carry
and the final outgoing carry are unchanged.

The separate digit_adc helper has explicit overflow-bit handling;
it is deliberately not used here, since doing so would alter the
production digit_adc4 semantics. This experiment preserves unsigned
wraparound even on hostile overflow inputs. Equality to baseline is
not a new proof that arbitrary inputs satisfy the engine's arithmetic
preconditions.

## Cost ledger and uncertainty

The source removes four width-pattern predicates, associated width
XORs, and the multi-case dispatch. It keeps four serial additions,
four right shifts and four output masks. There are no new memory
accesses, barriers, launches or carry buffers.

The original equal-width branches can share masks. The candidate may
compute more masks when equality is known only at runtime. A compiler
may already lower the original efficiently, or common-subexpression
elimination may recover sharing in the candidate. Register use,
branch lowering, occupancy and actual instruction counts are unknown.

This does not shorten the four-stage arithmetic dependency itself.
Reject if extra masks or scheduling effects outweigh the dispatch
saving, or if the helper is too small a part of end-to-end time.

## Owner-only validation

The new tests/perf_straight_carry_device.cpp extracts both guarded
implementations from the candidate source into one OpenCL program.
It compares every returned digit and outgoing carry on one selected
GPU. It covers all 16 two-width patterns, five width pairs including
1/2 and 62/63, four incoming carry values, zeros, full-width maxima,
alternating maxima, radix boundaries and fixed-seed random values:
4,160 cases and 20,800 compared words. It has not been compiled or run.
It is a differential helper fixture, not a timing or full-engine test.

Example owner build/run, from a separate validation checkout on Linux:

```sh
c++ -std=c++17 -O2 tests/perf_straight_carry_device.cpp -lOpenCL -o /tmp/perf-straight-carry-device
/tmp/perf-straight-carry-device 0 kernels/prmers.cl
```

Then validate both exact environment states through the normal legacy
Program path on RTX 3080 and Radeon VII. Compare full digit vectors,
block carries and complete residues for identical legal exponents,
transform sizes, propagation depths, all carry modes and block seams.
Retain the existing multiply-by-3 GMP regression as additional coverage.
The standalone fixture does not exercise environment wiring or masks.

Only after equality, inspect registers/spills and branch/mask lowering,
then alternate same-profile carry-kernel and complete legacy-workload
timings. Confirm the experimental notice and actual backend. Do not
compare a legacy candidate with an Aevum/Marin production run.

## Resume

Aevum D1 is PR #13 and remains independent. Existing A1/A2/C1/A3/A4
candidates remain pending owner validation; none is accepted or enabled.
Next source question: specialize two-value digit widths into compile-time
mask selections so the carry dependency no longer includes variable
width shifts. This would be a different arithmetic-cost hypothesis and
must justify its added per-digit selects before implementation.
