#!/usr/bin/env python3
"""Gpu::saveProof() must try hard to make a proof, and a failed proof must not lose a finished PRP result.

saveProof(): the attempts are the initial power twice, each lower power down to 5 (verified), then min(initial, 4) unverified.
Every failure moves on to the next attempt.  CRCError, ReadError and WriteError are plain structs, not std::exception, so they
and fs::filesystem_error are caught by type and mark a file error; const char*, std::string and std::exception are caught too.
A stop request after a failure is propagated.  If every attempt fails: a file error throws "bad proof generation"; anything
else (most likely the GPU) calls Signal::requestStop() and throws, so every worker stops without reporting.

isPrimePRP(): the saveProof() call is wrapped in catch (...), which rethrows when a stop was requested and otherwise reports
the result without a proof.

The end-to-end behaviour on a device is tests/proof_retry_gpu_test.sh (make test-gpu-proof-retry).
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
src = (ROOT / "src" / "Gpu.cpp").read_text()
fail = []

m = re.search(r"fs::path Gpu::saveProof\(const Args& args, ProofSet& proofSet\) \{\n(.*?)\n\}\n", src, re.S)
if not m:
    print("FAIL: Gpu::saveProof not found")
    sys.exit(1)
body = m.group(1)

for needle, what in (
    ("attempts{{initialPower, true}, {initialPower, true}}", "the initial power is not tried twice, verified"),
    ("for (u32 p = initialPower; p-- > 5; ) { attempts.push_back({p, true}); }", "lower powers down to 5 are not tried"),
    ("attempts.push_back({min(initialPower, 4u), false});", "no last-resort unverified attempt at min(initial, 4)"),
    ("while (proofSet.power > power) { proofSet.reducePower(); }", "the power is not reduced to the attempt's"),
    ("if (Signal::stopRequested()) { throw \"stop requested\"; }", "a stop request after a failed attempt is not propagated"),
):
    if needle not in body:
        fail.append(f"saveProof: {what}")

compute = body.find("proofSet.computeProof(this)")
tryPos = body.rfind("try {", 0, compute)
if compute < 0 or tryPos < 0:
    fail.append("saveProof: computeProof is not inside a try block")

handlers = {mm.group(1).strip(): mm.group(2) for mm in re.finditer(r"catch \(([^)]*)\)\s*\{(.*?)\n    \}", body, re.S)}
for exc in ("const CRCError& e", "const ReadError& e", "const WriteError& e", "const fs::filesystem_error& e"):
    if exc not in handlers:
        fail.append(f"saveProof: no catch ({exc})")
    elif "fileError = true;" not in handlers[exc]:
        fail.append(f"saveProof: catch ({exc}) does not mark a file error")
for exc in ("const char* mes", "const string& mes", "const std::exception& e"):
    if exc not in handlers:
        fail.append(f"saveProof: no catch ({exc})")
    elif "fileError = true;" in handlers[exc]:
        fail.append(f"saveProof: catch ({exc}) marks a file error")
# std::exception must come after fs::filesystem_error (which derives from it)
if body.find("catch (const std::exception&") < body.find("catch (const fs::filesystem_error&"):
    fail.append("saveProof: catch (std::exception) precedes catch (fs::filesystem_error)")

loop = body.find("for (auto [power, verify] : attempts)")
after = body[loop:]
after_loop = after[after.find('log("Proof of power %u failed'):]
ife = after_loop.find('if (fileError) { throw "bad proof generation"; }')
req = after_loop.find("Signal::requestStop();")
if loop < 0 or ife < 0:
    fail.append("saveProof: a final file error does not throw to the caller")
if req < 0 or (ife >= 0 and req < ife):
    fail.append("saveProof: requestStop() is not on the non-file-error path after the loop")

sig = (ROOT / "src" / "Signal.h").read_text()
if "static void requestStop();" not in sig:
    fail.append("Signal::requestStop() is not declared")

call = re.search(r"if \(k >= kEndEnd\) \{(.*?)return \{isPrime, finalRes64", src, re.S)
if not call:
    fail.append("isPrimePRP: the end-of-test block was not found")
else:
    c = call.group(1)
    t = c.find("try {")
    s = c.find("saveProof(args, *proofSet)")
    h = c.find("catch (...)")
    if not (0 <= t < s < h):
        fail.append("isPrimePRP: saveProof() is not wrapped in try / catch (...)")
    elif "if (Signal::stopRequested()) { throw; }" not in c[h:]:
        fail.append("isPrimePRP: the catch does not rethrow on a stop request")
    elif "reporting the result without a proof" not in c[h:]:
        fail.append("isPrimePRP: the catch does not report the result without a proof")

if fail:
    for f in fail:
        print("FAIL:", f)
    sys.exit(1)
print("proof_crc_retry_source_test: ok")
