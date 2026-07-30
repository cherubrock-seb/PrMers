#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
script = root / "scripts" / "generate_gaussian_worktodo.py"
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "worktodo.txt"
    subprocess.run([
        str(script), "--start", "1000000", "--count", "3", "--output", str(out),
        "--mode", "chain", "--pm1-b1", "100000", "--pm1-b2", "1000000",
        "--ecm-b1", "2000", "--ecm-b2", "0", "--curves", "2",
        "--sieve", "1000000000000", "--chunk-bits", "262144", "--family", "BOTH",
    ], check=True)
    entries = [line for line in out.read_text().splitlines() if line and not line.startswith("#")]
    assert len(entries) == 3
    assert entries[0] == "GMCHAIN=1000003,100000,1000000,2000,0,2,1000000000000,262144,proth,BOTH"
    assert all(line.startswith("GMCHAIN=") for line in entries)
print("Gaussian worktodo generator test passed")
