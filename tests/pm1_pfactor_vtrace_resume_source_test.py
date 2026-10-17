#!/usr/bin/env python3
"""Host-only guards for PFactor parsing and baby-batched P-1 V-trace resume."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
pm1 = (root / "src/modes/RunPM1.cpp").read_text(encoding="utf-8")
parser_test = (root / "tests/gaussian_worktodo_parser_test.cpp").read_text(encoding="utf-8")

# The field report used a fractional tests_saved value.  Keep that exact shape
# under regression so it cannot silently fall out of the worktodo parser.
forum_line = (
    "Pfactor=0123456789ABCDEF0123456789ABCDEF,"
    "1,2,268113067,-1,80,1.3"
)
assert forum_line in parser_test
assert "PFACTOR_FRACTIONAL_TESTS_SAVED=PASS" in parser_test
assert "math::choosePm1Bounds(268113067u, 80.0, 1.3)" in parser_test

# Baby-window V-trace needs a distinct checkpoint contract: saved_idx is the
# active baby-window start and saved_p is the next prime in that pass.
assert "useBabyBatching && !usePair95) ? 25 : 24" in pm1
assert "version == 24 || version == 25" in pm1
assert "resumeBatchStart = resumed_s2 ? static_cast<size_t>(resume_idx) : 0" in pm1
assert "bPrime = resumeThisPass ? resume_p_u64 : p0u" in pm1
assert "save_ckpt_s2(eng, bPrime, static_cast<uint64_t>(batchStart), cur_k, et);" in pm1
assert "save_ckpt_s2(eng, p0u, static_cast<uint64_t>(nextBatchStart), cur_k, et);" in pm1
assert "Resuming Stage 2 from checkpoint at pass" in pm1
assert "Backup V-trace baby-batched Stage 2 at pass" in pm1

# The old behaviour explicitly discarded any Stage-2 checkpoint and admitted
# that an interrupted pass could not be resumed.
assert "baby-batched checkpoints are not implemented" not in pm1
assert "does not checkpoint inside a baby-batched pass" not in pm1

# Model the key replay invariant independently of GPU arithmetic: a checkpoint
# taken after a prefix of a pass must resume at the next item, not multiply the
# prefix twice.  Each pass contributes a deterministic sequence of factors.
passes = [
    [2, 3, 5, 7],
    [11, 13, 17],
]
full = 1
for window in passes:
    for term in window:
        full *= term

acc = 1
for term in passes[0]:
    acc *= term
# Interrupt in pass 2 after two terms.  RACC contains those terms and saved_p
# points to the third (next) term.
acc *= passes[1][0]
acc *= passes[1][1]
saved_next = 2
for term in passes[1][saved_next:]:
    acc *= term

assert acc == full, (acc, full)
print("pm1 PFactor / baby-batched V-trace resume source regression passed")
