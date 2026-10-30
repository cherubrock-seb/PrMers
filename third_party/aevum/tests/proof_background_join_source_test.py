#!/usr/bin/env python3
"""Gpu::isPrimePRP() must wait for the Background thread before it generates the proof.

The proof residue for k == E is queued on the Background thread and only a few hundred squarings (up to the next check)
separate it from the call to saveProof(), whose computeProof() reads it back.  On a slow disk or a busy queue the
file can still be missing or half written, and the proof then fails (or, with a retry, is made at a lower power).
isPrimePRP() therefore has to call background->waitEmpty() before saveProof().

Checks the source of Gpu::isPrimePRP(): waitEmpty() comes before the saveProof() call, after the loop reaches kEndEnd.
"""
import pathlib
import re
import sys

src = (pathlib.Path(__file__).resolve().parent.parent / "src" / "Gpu.cpp").read_text()
start = src.find("PRPResult Gpu::isPrimePRP(")
end = src.find("LLResult Gpu::isPrimeLL(", start)
if start < 0 or end < 0:
    print("FAIL: Gpu::isPrimePRP not found")
    sys.exit(1)
body = src[start:end]

m = re.search(r"if \(k >= kEndEnd\) \{(.*?)return \{", body, re.S)
if not m:
    print("FAIL: the k >= kEndEnd block of isPrimePRP was not found")
    sys.exit(1)
block = m.group(1)
wait = block.find("background->waitEmpty()")
save = block.find("saveProof(")
if save < 0:
    print("FAIL: saveProof is not called in the k >= kEndEnd block")
    sys.exit(1)
if wait < 0 or wait > save:
    print("FAIL: isPrimePRP does not call background->waitEmpty() before saveProof()")
    sys.exit(1)
print("proof_background_join_source_test: ok")
