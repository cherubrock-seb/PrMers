from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

proof = (ROOT / "src/core/ProofMarin.cpp").read_text()
proofset = (ROOT / "src/core/ProofSetMarin.cpp").read_text()
run = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()

assert "canonicalProofWords" in proof
assert "padded.resize(expectedWords, 0u)" in proof
assert "hasher.update(padded.data(), nBytes)" in proof

assert "bufferPool[0] == 0" in proofset
assert "levelResult.resize(expectedWords, 0u)" in proofset

persist = run.index("provisionalOptions.proof = false")
proof_start = run.index("Generating PRP proof file...")

assert persist < proof_start

print("ProofMarin source/control-flow regression: PASS")
