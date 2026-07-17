#!/usr/bin/env python3
"""Reject restrict-qualified Apple GF61 middle-in input/output aliasing."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
cl = (root / "src/cl/fftmiddlein.cl").read_text()
base = (root / "src/cl/base.cl").read_text()

assert "#define P(x) global x * restrict" in base
assert "#define CP(x) const P(x)" in base

# These two operations are deliberately in-place.  They must expose one data
# pointer, not separate restrict-qualified input and output pointers.
assert re.search(r"KERNEL\(IN_WG\) fftMiddleInGF61ApplyScalarApple\(P\(T2\) io, CP\(T2\) factor\)", cl)
assert re.search(r"KERNEL\(IN_WG\) fftMiddleInGF61MulApple\(P\(T2\) io, Trig trig\)", cl)
assert "fftMiddleInGF61ApplyScalarApple(P(T2) out, CP(T2) data" not in cl
assert "fftMiddleInGF61MulApple(P(T2) out, CP(T2) in" not in cl

assert "kfftMidInGF61ApplyScalarApple(*out, *in);" in gpu
assert "kfftMidInGF61MulApple(*in);" in gpu
assert "kfftMidInGF61ApplyScalarApple(*out, *out, *in);" not in gpu
assert "kfftMidInGF61MulApple(*in, *in);" not in gpu
assert "kfftMidInGF61MulApple.setFixedArgs(1, bufTrigM);" in gpu

print("Aevum Apple GF61 middle-in restrict-alias audit passed")
