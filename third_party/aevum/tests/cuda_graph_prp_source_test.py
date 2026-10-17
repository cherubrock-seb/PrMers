#!/usr/bin/env python3
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "src" / "Gpu.cpp").read_text()
lines = [line.strip() for line in src.splitlines()]

needle = "square(bufData, bufData, leadIn, leadOut, false, false, true);"
assert lines.count(needle) == 1, "production PRP square must be a real C++ statement with prp=true"
assert r"\n    square(bufData, bufData, leadIn, leadOut" not in src, "literal escaped-newline source corruption"

assert "cuda_graph_squares_ && prp &&" in src
assert "leadIn == LEAD_WIDTH && leadOut == LEAD_WIDTH" in src
assert "!doMul3 && !doLL" in src

print("cuda graph PRP eligibility source gate: PASS")
