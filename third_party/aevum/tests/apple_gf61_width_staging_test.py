#!/usr/bin/env python3
"""Validate the Apple GF61 global-stage decomposition for WIDTH=256/NW=4."""
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
cl = (root / "src/cl/fftp.cl").read_text()

stages = [1, 4, 8, 16, 64, 256, 512]
assert "fftP61WidthRadixApple" in cl
for stage in stages:
    assert f"fftP61TwiddleShuffle{stage}Apple" in cl
    assert f"case {stage}:" in gpu
assert "fftP61WidthFinalApple" in cl
assert "kfftP61WidthRadixApple(*current);" in gpu
assert "std::swap(current, alternate);" in gpu
assert "kfftP61WidthFinalApple(*out, *current);" in gpu
assert "for (u32 stage = firstStage; stage < width_workgroup; stage *= nW)" in gpu
assert "fftP61WidthStageFused1Apple" in cl
assert "fftP61WeightStage1FusedApple" in cl
assert "AEVUM_APPLE_FFTP_V55" in gpu

# The new radix and scalar twiddle+shuffle entries must not use LDS/barriers.
radix_body = cl.split("KERNEL(G_W) fftP61WidthRadixApple", 1)[1].split("#define DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE", 1)[0]
assert "local " not in radix_body
assert "bar(" not in radix_body
macro_body = cl.split("#define DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE", 1)[1].split("DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle1Apple", 1)[0]
assert "local " not in macro_body
assert "bar(" not in macro_body

# Match FFTShape::nW(): 256/1024 use radix 4; 512/4096 use radix 8.
def nw(width: int) -> int:
    return 4 if width in (256, 1024) else 8

for width in (256, 512, 1024, 4096):
    radix = nw(width)
    wg = width // radix
    expected = []
    s = 1
    while s < wg:
        expected.append(s)
        s *= radix
    assert all(s in stages for s in expected), (width, radix, expected)

# Arithmetic/permutation equivalence for the exact M2 smoke plan.
P = (1 << 61) - 1
WIDTH = 256
RADIX = 4
WG = WIDTH // RADIX

def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)

def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)

def cmul(a, b):
    return ((a[0] * b[0] - a[1] * b[1]) % P,
            (a[0] * b[1] + a[1] * b[0]) % P)

def fft4(u):
    a0, a1, a2, a3 = u
    x0, x2 = add(a0, a2), sub(a0, a2)
    y0, y3 = add(a1, a3), sub(a1, a3)
    # Multiplication by the fourth root used by fft4: (x,y) -> (-y,x).
    y3t = ((-y3[1]) % P, y3[0])
    return [add(x0, y0), add(x2, y3t), sub(x0, y0), sub(x2, y3t)]

def original_stage(flat, trig, f):
    lanes = []
    for me in range(WG):
        u = fft4([flat[i * WG + me] for i in range(RADIX)])
        base = me & ~(f - 1)
        for i in range(1, RADIX):
            u[i] = cmul(u[i], trig[(i - 1) * WG + base])
        lanes.append(u)
    shuffled = [None] * WIDTH
    for me, u in enumerate(lanes):
        for i, value in enumerate(u):
            dst = i * f + (me & ~(f - 1)) * RADIX + (me & (f - 1))
            assert shuffled[dst] is None
            shuffled[dst] = value
    assert all(v is not None for v in shuffled)
    return shuffled

def new_stage(flat, trig, f):
    # In-place radix kernel.
    radix_out = list(flat)
    for me in range(WG):
        u = fft4([flat[i * WG + me] for i in range(RADIX)])
        for i, value in enumerate(u):
            radix_out[i * WG + me] = value
    # Scalar twiddle+shuffle kernel.
    out = [None] * WIDTH
    for x, value in enumerate(radix_out):
        i, me = divmod(x, WG)
        base = me & ~(f - 1)
        if i:
            value = cmul(value, trig[(i - 1) * WG + base])
        dst = i * f + (me & ~(f - 1)) * RADIX + (me & (f - 1))
        assert out[dst] is None
        out[dst] = value
    assert all(v is not None for v in out)
    return out

rng = random.Random(0xA3E014)
for _ in range(8):
    data = [(rng.randrange(P), rng.randrange(P)) for _ in range(WIDTH)]
    trig = [(rng.randrange(P), rng.randrange(P)) for _ in range((RADIX - 1) * WG)]
    old = list(data)
    new = list(data)
    for f in (1, 4, 16):
        old = original_stage(old, trig, f)
        new = new_stage(new, trig, f)
        assert old == new
    # Final radix has no twiddle or shuffle.
    old_final = list(old)
    new_final = list(new)
    for me in range(WG):
        a = fft4([old[i * WG + me] for i in range(RADIX)])
        b = fft4([new[i * WG + me] for i in range(RADIX)])
        for i in range(RADIX):
            old_final[i * WG + me] = a[i]
            new_final[i * WG + me] = b[i]
    assert old_final == new_final

print("Aevum Apple GF61 global-stage arithmetic and dispatch test passed")
