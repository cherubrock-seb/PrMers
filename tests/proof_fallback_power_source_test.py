from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

marin = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()
mgr = (ROOT / "include/core/ProofManagerMarin.hpp").read_text()

# The CPU fallback proof is written at the power the checkpoints were saved
# for, not at the lowered power the GPU retry loop left in options.proofPower.
assert "uint32_t power() const" in mgr

fallback = marin.index("proofFilePath = proofManagerMarin.proof();")
tail = marin[fallback:fallback + 600]
assert "options.proofPower = proofManagerMarin.power();" in tail

# The assignment belongs to the fallback only: the GPU path reports the power
# of the attempt that succeeded.
assert marin.count("options.proofPower = proofManagerMarin.power();") == 1

print("Proof fallback power source regression: PASS")
