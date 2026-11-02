# Experimental LL-SAFE unit-norm square

Status: **IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED**

Baseline: PrMers `e501da30bab163bf4a8575dbf8efb8660b9e5868`.
This candidate applies only to the main LL-SAFE (`-ll`) recurrence in
`src/modes/RunLlSafeMarin.cpp`. It does not modify the embedded Aevum engine.

## Mechanism

Write the LL-SAFE pair as `A + B*sqrt(3)` modulo `M = 2^p - 1`.
The initial pair `(2,1)` has norm `A*A - 3*B*B = 1`. Norm is
multiplicative, so every correct square retains it, including when M is
composite. Consequently:

- General square: `(A*A + 3*B*B, 2*A*B)`.
- Unit-norm square: `(2*A*A - 1, 2*A*B)`.

The helper prepares the old A before overwriting it, computes `2*A*A-1`,
then multiplies the unchanged B by that prepared A with factor 2.
The production helper is shared with the host arithmetic regression.

Per live iteration the engine schedule changes from eight API calls
(two squares, one multiplication, one preparation, three copies and one add)
to four (one preparation, one square, one scalar subtraction and one multiplication).
This eliminates a modular square, three explicit copies and a register addition.
Scalar factors still incur the existing backend's scaling cost; no fused carry
or new kernel is assumed. The general verification path remains unchanged.
A successful verification additionally checks the live pair's norm, costing
two squares, two copies, one register subtraction, one constant write and one
equality operation per full check, not per iteration. A resumed run checks up
to two pairs before starting its timer.

The intended domain is sustained LL-SAFE work on RTX 3080 CUDA/OpenCL and
Radeon VII OpenCL. Savings may be smaller for short jobs or frequent checks.
There is no measured speedup and no claim about ordinary PRP throughput.

## Activation and safety

`PRMERS_LLSAFE_NORM_SQUARE=1` enables the experiment.
Unset or `0` preserves the existing schedule; other values are rejected.
The setting is read only by LL-SAFE, after the existing quick-check exit.

Only the live result pair uses the specialization. All Gerbicz-Li checker
squares and pair products use general arithmetic. Before accepting a checked
block, the live result must also satisfy the norm condition. A failed pair
check or norm check disables the specialization for the rest of that process,
then follows the existing retry limit and rollback path.

On resume, both the live result and the saved rollback result must have norm
one. If either fails, that run uses the general schedule and its existing
recovery behavior. The norm check is a precondition check, not a replacement
for Gerbicz-Li verification or a proof that checkpoint data is otherwise valid.

The 18-register layout, checkpoint format and pre-final pair are retained.
Scratch registers 12, 13 and 14 may differ; they are overwritten before use.
All logical state remains in the existing representation. OFF-to-ON and
ON-to-OFF resumes require downstream validation on the same backend.
No new cross-backend checkpoint compatibility is implied.

The factor-3 FFT capacity guard remains necessary for the general checker and
norm checks. Manual FFT selection, tune-file precedence, device mapping,
backend selection and kernel caches are unchanged. Unset the variable or use
`0` for the safe fallback. LL-SAFE2, LL-UNSAFE, PRP, P-1 and ECM do not use
this helper. Other devices/backends remain unvalidated; no architecture-specific
performance assumption or production enablement is included.

## Validation handoff

None of the following commands or GPU checks has been executed for this candidate.
Keep at least 15 GiB free on each validation host.

1. Build with the normal supported toolchain. Run `make test-llsafe-norm-square`
   and `python3 tests/llsafe_factor_capacity_source_test.py`.
   The new host test exhaustively compares every pair modulo 3, 7, 15, 31, 63
   and 127, including composite rings and zero coordinates; checks repeated
   squaring against general pair arithmetic; checks invalid live/rollback
   norms, prepared-operand lifetime, unrelated registers and strict flag parsing.
   It is an integer arithmetic adapter, not GPU evidence.
2. With each flag value, run `make test-llsafe-error-recovery`,
   `make test-llsafe-resume-iter0-check`, `make test-final-checkpoint` and
   `make test-llsafe-worktodo-retire`. The existing recovery scripts explicitly
   select Marin; also reproduce their cases with forced Aevum.
3. On RTX 3080 OpenCL, RTX 3080 CUDA and Radeon VII OpenCL, run isolated OFF/ON
   LL-SAFE jobs for p=227 (composite), p=4423 (Mersenne prime) and p=6563
   (iteration-zero verification/resume case), with `-checklevel 1` and normal
   checking. Compare verdicts, Gerbicz-Li behavior and **all canonical words** of
   the logical registers 0..11 and 16..17 at matching iteration/checkpoint
   boundaries, including the final pre-square state. Res64 alone is insufficient;
   raw checkpoint bytes include deliberately different scratch/prepared data.
4. Interrupt before/after a block boundary and after the iteration-zero check.
   Exercise OFF-to-ON and ON-to-OFF resume. Inject errors before/after a saved
   boundary; require detection, the fallback log, unchanged bounded recovery,
   and the same final full state. Exercise a valid-CRC checkpoint fixture with
   a non-unit live pair and separately a non-unit saved rollback pair; require
   the resume gate to select general squares. Include a completed checkpoint
   retained by failed result persistence. Do not bypass existing CRC/backend
   checks. A missing fallback or unexpected accepted corrupted state rejects
   the candidate.
5. Short performance screen: p=21000029, the same factor-3-safe explicit FFT,
   backend, checklevel and cache settings in both arms. Use bounded identical
   iteration windows first; compare all logical words at the end and check for
   fallback. Account for verification and startup separately. Check a second
   meaningful exponent/FFT if promising.
6. For a promising screen, measure at least three counterbalanced paired runs
   (AB, BA, AB; A=OFF, B=ON), using the same binary and changing only this flag.
   Require **at least 3% repeatable end-to-end improvement** for KEEP.
   Include complete LL-SAFE operation timings and checkpoint/checker overhead,
   not only isolated square timings. Ordinary PRP is unchanged; record a
   full-operation PRP control rather than claiming a PRP speedup from LL-SAFE.
   Record kernel timing/launch counts and any unexpected register, scratch or
   occupancy differences. Exclude runs that fell back from speedup claims.

The algebraic identity is exact under its precondition. Hardware arithmetic,
error-injection coverage, checkpoint transitions, and end-to-end benefit are
still acceptance gates.
