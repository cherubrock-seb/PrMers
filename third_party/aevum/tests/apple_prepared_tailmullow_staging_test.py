#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()

start = gpu.index("      if (kern == KTAILMULLOW) {")
end = gpu.index("      if (kern == KMIDOUT)", start)
block = gpu[start:end]

# The stock CRT route is retained: FP/GF31 stay on their upstream MUL_LOW
# kernels and non-Apple GF61 remains byte-for-byte on ktailMulLowGF61.
assert "if (cache_group == 1) ktailMulLow(*out, *in1, *in2);" in block
assert "if (cache_group == 2) ktailMulLowGF31(*out, *in1, *in2);" in block
assert "#if defined(__APPLE__)" in block and "#else" in block
assert block.count("ktailMulLowGF61(*out, *in1, *in2);") == 2

# Apple transforms only the live fftMiddleIn operand.  The prepared operand
# from fftHinGF61 is never loaded, transformed, or written again.
assert "ktailMulGF61LoadScalarApple(*out, fullBase, *in1, fullBase);" in block
assert "ktailMulGF61LoadScalarApple(*in2" not in block
assert "runAppleTailMulLowGF61Fft(out, fullBase, in1, fullBase, out, fullBase);" in block
assert "ktailMulGF61PairSpecialScalarApple(*raw, rawBase, *out, fullBase, *in2, fullBase, bufTrigH);" in block
assert "ktailMulGF61PairNormalScalarApple(*raw, rawBase, *out, fullBase, *in2, fullBase, bufTrigH);" in block
assert "runAppleTailMulLowGF61Fft(raw, rawBase, in1, fullBase, out, fullBase);" in block

# No alternate modulus, FFT type, carry, or arithmetic operation is introduced.
for forbidden in ("FFT3231", "FFT_TYPE=52", "carryM", "GF31-only", "FP32+GF31"):
    assert forbidden not in block

# The prepared operand is read-only throughout the Apple branch.
for forbidden_write in (
    "ktailMulGF61LoadScalarApple(*in2",
    "ktailMulGF61FftRadixApple(*in2",
    "ktailMulGF61FftTwiddleApple(*in2",
    "ktailMulGF61FftFinalApple(*in2",
):
    assert forbidden_write not in block

print("PASS: Apple prepared GF61 tailMulLow uses exact staged CRT arithmetic")
