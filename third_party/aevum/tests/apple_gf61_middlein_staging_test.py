#!/usr/bin/env python3
"""Validate Apple GF61 scalar middle-in staging, arithmetic, buffers and LDS mapping."""
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
cl = (root / "src/cl/fftmiddlein.cl").read_text()

kernels = [
    "fftMiddleInGF61LoadScalarApple",
    "fftMiddleInGF61Mul2FactorScalarApple",
    "fftMiddleInGF61ApplyScalarApple",
    "fftMiddleInGF61FftApple",
    "fftMiddleInGF61MulApple",
    "fftMiddleInGF61TransposeApple",
]
for name in kernels:
    assert name in cl, name
for member in (
    "kfftMidInGF61LoadScalarApple",
    "kfftMidInGF61Mul2FactorScalarApple",
    "kfftMidInGF61ApplyScalarApple",
    "kfftMidInGF61FftApple",
    "kfftMidInGF61MulApple",
    "kfftMidInGF61TransposeApple",
):
    assert member in gpu, member

order = [
    "kfftMidInGF61LoadScalarApple(*out, *in);",
    "kfftMidInGF61Mul2FactorScalarApple(*in);",
    "kfftMidInGF61ApplyScalarApple(*out, *in);",
    "kfftMidInGF61FftApple(*in, *out);",
    "kfftMidInGF61MulApple(*in);",
    "kfftMidInGF61TransposeApple(*out, *in);",
]
pos = [gpu.index(x) for x in order]
assert pos == sorted(pos)
assert 'config["INPLACE"] = "0";' in gpu
assert "hN / (BIG_H / SMALL_H) * fft.shape.middle" in gpu

assert "fftMiddleInGF61ApplyScalarApple(P(T2) io, CP(T2) factor)" in cl
assert "fftMiddleInGF61MulApple(P(T2) io, Trig trig)" in cl
assert "fftMiddleInGF61ApplyScalarApple(P(T2) out, CP(T2) data" not in cl
assert "fftMiddleInGF61MulApple(P(T2) out, CP(T2) in" not in cl

# Scalar front-end stages must not materialize GF61 u[MIDDLE]. Only the FFT,
# post-multiply and final transpose need the per-thread middle vector.
parts = {}
for name in kernels:
    start = cl.index(f"KERNEL(IN_WG) {name}")
    end = cl.find("KERNEL(IN_WG)", start + 1)
    if end < 0:
        end = cl.index("#endif", start)
    parts[name] = cl[start:end]
for name in kernels[:3]:
    assert "GF61 u[MIDDLE]" not in parts[name]
    assert "local Z61" not in parts[name]
    assert "middleShuffle(" not in parts[name]
for name in kernels[3:5]:
    assert "local Z61" not in parts[name]
assert "local Z61 lds" in parts[kernels[5]]
assert "middleShuffle(lds, u, IN_WG, IN_SIZEX)" in parts[kernels[5]]
assert "flat_group % MIDDLE" in parts[kernels[0]]
assert "flat_group % MIDDLE" in parts[kernels[1]]
assert "flat_group % MIDDLE" in parts[kernels[2]]

P = (1 << 61) - 1
WG = 64
BLOCK = 16
SIZEY = WG // BLOCK
rng = random.Random(0xA3E06123)

def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)

def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)

def cmul(a, b):
    return ((a[0] * b[0] - a[1] * b[1]) % P,
            (a[0] * b[1] + a[1] * b[0]) % P)

def factor_binary(base, w, k):
    # Mirror the fixed four-bit powering used by the Apple kernel.
    power = w
    if k & 1:
        base = cmul(base, power)
    power = cmul(power, power)
    if k & 2:
        base = cmul(base, power)
    power = cmul(power, power)
    if k & 4:
        base = cmul(base, power)
    power = cmul(power, power)
    if k & 8:
        base = cmul(base, power)
    return base

def factor_reference(base, w, k):
    for _ in range(k):
        base = cmul(base, w)
    return base

def local_fft(values):
    # Exact radix-2 for the M2 smoke; reversible deterministic mixing for larger
    # MIDDLE values is enough to validate staging and global layout.
    if len(values) == 2:
        return [add(values[0], values[1]), sub(values[0], values[1])]
    out = list(values)
    step = 1
    while step < len(out):
        for j in range(0, len(out), 2 * step):
            for k in range(step):
                a, b = out[j + k], out[j + k + step]
                out[j + k], out[j + k + step] = add(a, b), sub(a, b)
        step *= 2
    return out

def transpose_threads(values):
    out = [[None] * len(values[0]) for _ in range(WG)]
    for old_me in range(WG):
        mx, my = old_me % BLOCK, old_me // BLOCK
        dst_me = mx * SIZEY + my
        out[dst_me] = list(values[old_me])
    return out

for middle in (2, 4, 8, 16):
    for _ in range(12):
        initial = [[(rng.randrange(P), rng.randrange(P)) for _ in range(middle)] for _ in range(WG)]
        base = [(rng.randrange(P), rng.randrange(P)) for _ in range(WG)]
        w = [(rng.randrange(P), rng.randrange(P)) for _ in range(WG)]
        pre = [[factor_reference(base[me], w[me], i) for i in range(middle)] for me in range(WG)]
        pre_binary = [[factor_binary(base[me], w[me], i) for i in range(middle)] for me in range(WG)]
        assert pre_binary == pre
        post = [[(rng.randrange(P), rng.randrange(P)) for _ in range(middle)] for _ in range(WG)]

        original = []
        for me in range(WG):
            u = [cmul(initial[me][i], pre[me][i]) for i in range(middle)]
            u = local_fft(u)
            u = [cmul(u[i], post[me][i]) for i in range(middle)]
            original.append(u)
        original = transpose_threads(original)

        # Scalar load and factor kernels flatten [middle][WG] independently.
        flat_data = [initial[me][i] for i in range(middle) for me in range(WG)]
        flat_factor = [pre_binary[me][i] for i in range(middle) for me in range(WG)]
        flat_applied = [cmul(flat_data[j], flat_factor[j]) for j in range(len(flat_data))]
        read_applied = [[flat_applied[i * WG + me] for i in range(middle)] for me in range(WG)]
        phase_fft = [local_fft(u) for u in read_applied]
        phase_post = [[cmul(phase_fft[me][i], post[me][i]) for i in range(middle)] for me in range(WG)]
        staged = transpose_threads(phase_post)
        assert staged == original

print("Aevum Apple GF61 scalar middle-in staging, arithmetic, buffer and LDS mapping test passed")
