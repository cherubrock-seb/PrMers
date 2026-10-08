from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mgr = (ROOT / "src/core/ProofManagerMarin.cpp").read_text()
gpu = (ROOT / "src/core/ProofManager.cpp").read_text()

# The CPU fallback writes its proof where the GPU path does: proof/ under the
# save path (-f), like the checkpoints and results.txt.
assert 'ensureDir(where.proofDir())' in gpu
cpu = mgr[mgr.index("ProofManagerMarin::proof() const"):]
assert 'proofSet_.location().proofDir()' in cpu
assert 'current_path()' not in cpu and 'current_path()' not in gpu

# A proof that does not read back is an error, not a warning, and only a
# validated file gets its final name.
assert "Warning: Proof file validation failed" not in mgr
assert "Proof file validation failed: " in cpu
assert cpu.index("ProofMarin::load(tmpPath)") < cpu.index("rename(tmpPath, proofFilePath)")

print("CPU proof fallback source regression: PASS")
