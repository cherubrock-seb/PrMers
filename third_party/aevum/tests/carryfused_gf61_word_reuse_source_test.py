#!/usr/bin/env python3
from pathlib import Path

src = Path("src/cl/carryfused.cl").read_text()
use = Path("src/UseOptions.h").read_text()

assert '"AEVUM_CARRY_REUSE_GF61_WU"' in use
assert src.count("#if !AEVUM_CARRY_REUSE_GF61_WU") == 2
assert src.count("u61[i] = as_ulong2(carryWords);") == 2
assert src.count("carryFinal(as_long2(u61[i]), carry[i], biglit0)") == 2
assert src.count("Word2 carryWordsFinal = wu[i];") == 2

fft3261 = src.split("#elif FFT_TYPE == FFT3261", 1)[1].split("#elif FFT_TYPE == FFT3161", 1)[0]
fft3161 = src.split("#elif FFT_TYPE == FFT3161", 1)[1].split("#elif FFT_TYPE == FFT323161", 1)[0]

for section in (fft3261, fft3161):
    assert "GF61 u61[NW];" in section
    assert "Word2 wu[NW];" in section
    assert "as_ulong2(carryWords)" in section
    assert "as_long2(u61[i])" in section

print("ok: FFT3161/FFT3261 optionally reuse exact 128-bit GF61 slots for Word2 carry payload")
