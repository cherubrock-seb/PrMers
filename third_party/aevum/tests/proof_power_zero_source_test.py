#!/usr/bin/env python3
"""Gpu::isPrimePRP() must cope with a proof power of 0.

Gpu::getProofPower() returns 0 ("Proof generation disabled!") when no complete set of proof residues can be built any more,
for example on a restart whose (E+1)/2 residue is missing.  ProofSet rejects power 0 with "Invalid proof power", so the
constructor must only run when there is a proof to make; otherwise the worker leaves its task loop on every restart and the
test can never finish.

Checks the source of Gpu::isPrimePRP(): the ProofSet is held in a std::optional that is only filled when power != 0, and
neither next() nor saveProof() is reached without it.
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
fail = []

if re.search(r"\n\s*ProofSet proofSet\s*\{", body):
    fail.append("isPrimePRP constructs a ProofSet unconditionally")
if not re.search(r"std::optional<ProofSet>\s+proofSet\s*;", body):
    fail.append("proofSet is not a std::optional<ProofSet>")
if not re.search(r"if \(power\) \{\s*proofSet\.emplace\(E, power\);\s*\}", body):
    fail.append("the ProofSet is not constructed only for power != 0")
if not re.search(r"persistK = proofSet \? proofSet->next\(k\) : u64\(-1\);", body):
    fail.append("the initial persistK does not handle a missing ProofSet")
m = re.search(r"if \(k >= kEndEnd\) \{(.*?)return \{", body, re.S)
if not m or not re.search(r"if \(proofSet\) \{[^}]*saveProof\(args, \*proofSet\)", m.group(1)):
    fail.append("saveProof is called without checking for a ProofSet")

if fail:
    for f in fail:
        print("FAIL:", f)
    sys.exit(1)
print("proof_power_zero_source_test: ok")
