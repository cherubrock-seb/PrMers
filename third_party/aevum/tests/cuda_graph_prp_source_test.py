#!/usr/bin/env python3
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "src" / "Gpu.cpp").read_text()

needle = "square(bufData, bufData, leadIn, leadOut, false, false, true);"
assert src.count(needle) == 1, "production PRP square must explicitly set prp=true"

# The graph scope must remain restricted to retained-width PRP squares.
assert "cuda_graph_squares_ && prp &&" in src
assert "leadIn == LEAD_WIDTH && leadOut == LEAD_WIDTH" in src
assert "!doMul3 && !doLL" in src

print("cuda graph PRP eligibility source gate: PASS")
