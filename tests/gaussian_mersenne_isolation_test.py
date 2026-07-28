#!/usr/bin/env python3
"""Source-level guard: Gaussian mode is opt-in and does not alter Aevum kernels."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
cli = (root / "src/io/CliParser.cpp").read_text()
app = (root / "src/core/App.cpp").read_text()
hpp = (root / "include/io/CliParser.hpp").read_text()

for flag in ("-gm-proth", "-gm-prp", "-gm-safe", "-gm-sieve", "-gm-base"):
    assert flag in cli
assert "runGaussianMersenne" in app
assert "gaussian_mersenne = false" in hpp
assert 'o.mode == "prp" || o.mode == "gm-proth" || o.mode == "gm-prp"' in app
assert "if (!o.gaussian_mersenne)" in app

# The extension must not touch the embedded Aevum tree. This works in source
# checkouts and is intentionally skipped in exported archives without .git.
if (root / ".git").exists():
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "third_party/aevum"],
        cwd=root,
        text=True,
    ).strip()
    assert not changed, f"Gaussian extension modified existing Aevum sources: {changed}"

print("Gaussian-Mersenne CLI/isolation test passed")
