from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mgr = (ROOT / "src/core/ProofManager.cpp").read_text()
hdr = (ROOT / "include/core/ProofManager.hpp").read_text()
legacy = (ROOT / "src/modes/RunPrpOrLl.cpp").read_text()
marin = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()

# The result of Proof::verify must decide whether the proof is kept.
assert "class ProofVerificationError" in hdr
assert "if (verify && !loadedProof.verify(gpu, proofPower))" in mgr
assert "throw ProofVerificationError" in mgr
assert mgr.index("throw ProofVerificationError") < mgr.index("fancyRename(tmpPath, finalPath)")

# Both PRP drivers: a failed verification is not retried, falls back to no
# CPU proof, and leaves no proof metadata in the reported JSON.
for name, src in (("legacy", legacy), ("marin", marin)):
    assert "core::ProofVerificationError" in src, name
    catch = src.index("catch (const core::ProofVerificationError& e)")
    tail = src[catch:catch + 1500]
    assert "options.proof = false" in tail or "proofSaved" in src, name

assert "if (!proofSaved)" in legacy
assert "options.proof = false;" in legacy

# Marin: the GPU retry loop must rethrow instead of retrying/falling back.
loop = marin.index("catch (const core::ProofVerificationError&)")
assert "throw;" in marin[loop:loop + 300]

print("Proof verification result source regression: PASS")
