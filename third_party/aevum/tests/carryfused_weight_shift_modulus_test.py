#!/usr/bin/env python3
"""In every carryFused variant the packed frac_bits/weight_shift counter must be reduced modulo the same prime as the weight shift itself.

The counter's weight_shift half is a shift amount for an NTT lane: modulo 31 for a GF(M31^2) lane and modulo 61 for GF(M61^2).  The
per-iteration "bigstep" is reduced with  % (<p>ULL << 32)  and the loop body only subtracts p once, so a bigstep reduced modulo the wrong
prime leaves shifts the single correction cannot bring back into range and the inverse weights come out wrong (the Gerbicz check fails
from the first block, with ROEmax pinned at 0.5).  FFT3231 once used 61 with a GF31 lane.
"""
import pathlib
import re
import sys

src = (pathlib.Path(__file__).resolve().parent.parent / "src" / "cl" / "carryfused.cl").read_text()
lines = src.split("\n")

fail = []
checked = 0
shift_mod = {}
for n, line in enumerate(lines, 1):
    # Single-lane kernels use bigword_weight_shift / combo_bigstep; the dual-lane ones prefix each lane with m31_ or m61_.
    m = re.search(r"const u32 (m31_|m61_)?bigword_weight_shift = .* % (\d+);", line)
    if m:
        shift_mod[m.group(1) or ""] = int(m.group(2))
        continue
    m = re.search(r"(m31_|m61_)?combo_bigstep = .* % \((\d+)ULL << 32\);", line)
    if m:
        checked += 1
        prefix = m.group(1) or ""
        if prefix not in shift_mod:
            fail.append(f"carryfused.cl:{n}: combo_bigstep without a preceding bigword_weight_shift")
        elif int(m.group(2)) != shift_mod[prefix]:
            fail.append(f"carryfused.cl:{n}: combo_bigstep is reduced modulo {m.group(2)} but the weight shift is modulo {shift_mod[prefix]}")

if checked < 6:
    fail.append(f"only {checked} combo_bigstep definitions found, the scanner is broken")
for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print(f"ok: {checked} carryFused combo_bigstep moduli match their weight-shift moduli")
