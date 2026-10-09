from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mgr = (ROOT / "src/core/ProofManagerMarin.cpp").read_text()
gpu = (ROOT / "src/core/ProofManager.cpp").read_text()

# The CPU fallback writes its proof where the GPU path does: proof/ under the
# save path (-f), like the checkpoints and results.txt.
assert 'ensureDir(where.proofDir())' in gpu
cpu = mgr[mgr.index("ProofManagerMarin::proof(bool verify, const GpuProofVerifier& gpuVerify) const"):]
assert 'proofSet_.location().proofDir()' in cpu
assert 'current_path()' not in cpu and 'current_path()' not in gpu

# A proof that does not read back is an error, not a warning, and only a
# validated file gets its final name.
assert "Warning: Proof file validation failed" not in mgr
assert "Proof file validation failed: " in cpu
assert cpu.index("ProofMarin::load(tmpPath)") < cpu.index("rename(tmpPath, proofFilePath)")

# The CPU proof is verified (unless -noverify) before it takes its final name,
# and a proof that fails is reported as the GPU path reports one. The GPU
# verifies it when it can, the CPU when that is cheap, else it is kept with a
# warning (the policy lives in verifyFallbackProof).
assert "verifyFallbackProof(" in cpu and "tmpPath" in cpu[cpu.index("verifyFallbackProof("):][:120]
assert "cpuVerifyMaxSeconds()" in cpu
assert cpu.index("verifyFallbackProof(") < cpu.index("rename(tmpPath, proofFilePath)")
assert "throw ProofVerificationError(" in cpu
assert "Status::Skipped" in cpu and "Warning: " in cpu
marin = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()
assert "proofManagerMarin.proof(options.verify, gpuVerify)" in marin
# The GPU verifier is the one the normal path uses (Proof::verify), and what
# makes it unusable throws, so the CPU policy takes over.
gv = marin[marin.index("const core::GpuProofVerifier gpuVerify"):marin.index("proofManagerMarin.proof(options.verify, gpuVerify)")]
assert "core::Proof::load(proofFile).verify(" in gv
assert "ensureProofGpuBackend();" in gv

print("CPU proof fallback source regression: PASS")
