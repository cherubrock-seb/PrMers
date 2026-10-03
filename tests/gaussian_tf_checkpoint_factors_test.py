#!/usr/bin/env python3
"""The GM/GQ trial-factoring checkpoint must carry the factors found so far."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
host = (root / "src/modes/RunGaussianTrialFactor.cpp").read_text()

# The checkpoint is written with the found factors and read back on resume.
assert "saveCheckpoint(checkpoint, nextK, found)" in host
assert "const std::vector<FoundFactor>& found" in host
assert "found = saved->found;" in host
assert "struct TfCheckpoint" in host

# Resume state must be restored before the target check that ends the loop,
# and `found` must not be re-created (emptied) after the checkpoint load.
load = host.index("found = saved->found;")
loop = host.index("while (nextK <= lastK && !targetSatisfied(request, found))")
assert load < loop
run = host.index("int runTrialFactor(")
assert host.count("std::vector<FoundFactor> found;", run) == 1
assert host.index("std::vector<FoundFactor> found;", run) < load

# Round-trip of the on-disk format (first line nextK, then "factor family-bit").
text = "1034\n41201 1\n"
lines = text.split("\n")
assert int(lines[0]) == 1034
assert [tuple(map(int, l.split())) for l in lines[1:] if l] == [(41201, 1)]
print("gaussian TF checkpoint factor test OK")
