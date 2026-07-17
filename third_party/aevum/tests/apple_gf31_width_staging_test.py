#!/usr/bin/env python3
"""Validate the Apple GF31 fully-global fftP decomposition."""
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
cl = (root / "src/cl/fftp.cl").read_text()

names = [
    "fftP31WeightScalarApple", "fftP31WidthRadixApple",
    "fftP31TwiddleShuffle1Apple", "fftP31TwiddleShuffle4Apple",
    "fftP31TwiddleShuffle8Apple", "fftP31TwiddleShuffle16Apple",
    "fftP31TwiddleShuffle64Apple", "fftP31TwiddleShuffle256Apple",
    "fftP31TwiddleShuffle512Apple", "fftP31WidthFinalApple",
]
for name in names:
    assert name in cl and ("k" + name) in gpu
assert "kfftP31Apple(*out, in);" not in gpu
assert "runGF31();\n    runGF61();" in gpu

# No LDS/barriers in the new Apple GF31 path.
block = cl.split("KERNEL(256) fftP31WeightScalarApple", 1)[1].split(
    "// Apple's Metal pipeline compiler also rejects", 1
)[0]
assert "local " not in block
assert "bar(" not in block

# Exact width-stage equivalence for WIDTH=256, NW=4.
P = (1 << 31) - 1
WIDTH = 256
RADIX = 4
WG = WIDTH // RADIX

def add(a, b): return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)
def sub(a, b): return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)
def cmul(a, b):
    return ((a[0] * b[0] - a[1] * b[1]) % P,
            (a[0] * b[1] + a[1] * b[0]) % P)
def fft4(u):
    a0, a1, a2, a3 = u
    x0, x2 = add(a0, a2), sub(a0, a2)
    y0, y3 = add(a1, a3), sub(a1, a3)
    y3t = ((-y3[1]) % P, y3[0])
    return [add(x0, y0), add(x2, y3t), sub(x0, y0), sub(x2, y3t)]

def stage(flat, trig, f):
    radix = list(flat)
    for me in range(WG):
        u = fft4([flat[i * WG + me] for i in range(RADIX)])
        for i, v in enumerate(u): radix[i * WG + me] = v
    out = [None] * WIDTH
    for x, v in enumerate(radix):
        i, me = divmod(x, WG)
        base = me & ~(f - 1)
        if i: v = cmul(v, trig[(i - 1) * WG + base])
        dst = i * f + (me & ~(f - 1)) * RADIX + (me & (f - 1))
        assert out[dst] is None
        out[dst] = v
    assert all(v is not None for v in out)
    return out

rng = random.Random(0x31A9955)
for _ in range(8):
    data = [(rng.randrange(P), rng.randrange(P)) for _ in range(WIDTH)]
    trig = [(rng.randrange(P), rng.randrange(P)) for _ in range((RADIX - 1) * WG)]
    for f in (1, 4, 16): data = stage(data, trig, f)
    final = list(data)
    for me in range(WG):
        u = fft4([data[i * WG + me] for i in range(RADIX)])
        for i, v in enumerate(u): final[i * WG + me] = v
    assert len(final) == WIDTH


# Scalar weight indexing must match the original grouped NW loop for the exact
# p=859553 M2 plan.
MASK64 = (1 << 64) - 1
EXP = 859553
BIG_HEIGHT = 512
FRAC_HI = 1198014463
NWORDS = WIDTH * BIG_HEIGHT * 2

def make_u64(hi, lo): return ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
def combo_frac(i): return (i * (FRAC_HI + 1) - 1) & MASK64
log2_root_two = ((1 << 30) // NWORDS) % 31
big_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31
shift_minus_one = (big_shift + 30) % 31
combo_step = make_u64(shift_minus_one, FRAC_HI)
combo_bigstep = (
    combo_frac(WG * BIG_HEIGHT * 2 - 1)
    + make_u64((WG * BIG_HEIGHT * 2 - 1) * shift_minus_one, 0)
) % (31 << 32)

def grouped(line, me, lane):
    word_index = (me * BIG_HEIGHT + line) * 2
    combo = (combo_frac(word_index) + make_u64(word_index * shift_minus_one, 0xFFFFFFFF)) & MASK64
    high = (combo >> 32) % 31
    for i in range(lane + 1):
        s0 = high
        combo = (combo + combo_step) & MASK64
        high = (combo >> 32) % 31
        s1 = high
        if i == lane: return s0, s1
        combo = (combo + combo_bigstep) & MASK64
        high = (combo >> 32) % 31

def scalar(p):
    line, x = divmod(p, WIDTH)
    word_index = (line + BIG_HEIGHT * x) * 2
    combo = (combo_frac(word_index) + make_u64(word_index * shift_minus_one, 0xFFFFFFFF)) & MASK64
    s0 = (combo >> 32) % 31
    combo = (combo + combo_step) & MASK64
    return s0, (combo >> 32) % 31

for line in range(BIG_HEIGHT):
    for me in range(WG):
        for lane in range(RADIX):
            pidx = line * WIDTH + lane * WG + me
            assert grouped(line, me, lane) == scalar(pidx)

print("Aevum Apple GF31 fully-global fftP staging test passed")
