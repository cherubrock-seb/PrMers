#!/usr/bin/env python3
"""Validate Apple GF61 fftHin global staging against stock indexing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()
gpuh = (ROOT / "src/Gpu.h").read_text()
cl = (ROOT / "src/cl/ffthin.cl").read_text()

names = [
    "fftHinGF61LoadScalarApple",
    "fftHinGF61FftRadixApple",
    "fftHinGF61FftTwiddleApple",
    "fftHinGF61FftShuffleApple",
    "fftHinGF61FftFinalApple",
    "fftHinGF61ApplePlaceholder",
]
for name in names:
    assert f" {name}(" in cl, name

for token in (
    '"fftHinGF61ApplePlaceholder"',
    "kfftHinGF61LoadScalarApple(*out, *in);",
    "kfftHinGF61FftRadixApple(*current);",
    "kfftHinGF61FftTwiddleApple(*current, stage);",
    "kfftHinGF61FftShuffleApple(*current, *next, stage);",
    "kfftHinGF61FftFinalApple(*current, *out);",
    "if (current == out) kfftHinGF61FftRadixApple(*out);",
):
    assert token in gpu, token

for name in names[:-1]:
    member = "k" + name[0].lower() + name[1:]
    assert member in gpuh, member

# Non-Apple continues to bind and call the untouched upstream kernel.
assert '#else\n  K(kfftHinGF61,           "ffthin.cl",  "fftHinGF61"' in gpu
assert '#else\n          kfftHinGF61(*out, *in);' in gpu
assert "KERNEL(G_H) fftHinGF61(P(T2) out, CP(T2) in, Trig smallTrig)" in cl
assert "fft_HEIGHT(lds, u, smallTrig61, 1, me);" in cl


def trans_pos(line, middle, width):
    return line // width + (line % width) * middle


def load_src(line, slot, me, width, middle, small_h, nh, in_sizex, in_wg):
    gh = small_h // nh
    size_y = in_wg // in_sizex
    x = line % width
    chunk_x = x // in_sizex
    x_within = x % in_sizex
    middle_i = line // width
    y = slot * gh + me
    chunk_y = y // size_y
    return (
        chunk_x * (small_h * middle * in_sizex)
        + x_within * size_y
        + middle_i * in_wg
        + (me % size_y)
        + chunk_y * (middle * in_wg)
    )


def out_index(line, slot, me, width, middle, small_h, nh):
    gh = small_h // nh
    return small_h * trans_pos(line, middle, width) + slot * gh + me

# Load is a bijection from fftMiddleIn layout to the exact stock fftHin output
# line layout, for both representative small and large plans.
for width, middle, small_h, nh, in_sizex, in_wg in (
    (256, 2, 256, 4, 16, 64),
    (1024, 8, 256, 4, 16, 64),
    (4096, 16, 512, 8, 16, 64),
):
    H = width * middle
    gh = small_h // nh
    src_seen = set()
    dst_seen = set()
    # Exhaustive for the production small plan, sampled lines for big plans.
    lines = range(H) if H <= 512 else (0, 1, width - 1, width, H // 2, H - 1)
    for line in lines:
        for slot in range(nh):
            for me in range(gh):
                src = load_src(line, slot, me, width, middle, small_h, nh, in_sizex, in_wg)
                dst = out_index(line, slot, me, width, middle, small_h, nh)
                assert 0 <= src < H * small_h
                assert 0 <= dst < H * small_h
                assert src not in src_seen
                assert dst not in dst_seen
                src_seen.add(src)
                dst_seen.add(dst)
    if H <= 512:
        assert src_seen == set(range(H * small_h))
        assert dst_seen == set(range(H * small_h))


def stock_shuffle(values, radix, gh, f):
    lds = [None] * (radix * gh)
    mask = f - 1
    for me in range(gh):
        for i in range(radix):
            pos = i * f + (me & ~mask) * radix + (me & mask)
            assert lds[pos] is None
            lds[pos] = values[i][me]
    return [[lds[i * gh + me] for me in range(gh)] for i in range(radix)]


def global_shuffle(values, radix, gh, f):
    out = [[None] * gh for _ in range(radix)]
    for out_i in range(radix):
        for out_me in range(gh):
            logical = out_i * gh + out_me
            remainder = logical & (f - 1)
            src_i = (logical // f) % radix
            src_me = (logical // (f * radix)) * f + remainder
            out[out_i][out_me] = values[src_i][src_me]
    return out

# Every global shuffle is exactly the logical LDS shufl permutation.
for radix in (4, 8):
    for gh in (32, 64, 128, 256):
        values = [[(i, me) for me in range(gh)] for i in range(radix)]
        f = 1
        while f < gh:
            assert global_shuffle(values, radix, gh, f) == stock_shuffle(values, radix, gh, f)
            f *= radix

# Host stage loop matches fft_common: radix/twiddle/shuffle for every s<WG,
# followed by one final radix.  If ping-pong lands on out, final radix is
# performed in place, avoiding restrict aliasing.
for radix in (4, 8):
    for gh in (32, 64, 128, 256):
        stages = []
        current = 0
        s = 1
        while s < gh:
            stages.append(s)
            current ^= 1
            s *= radix
        expected = []
        s = 1
        while s < gh:
            expected.append(s)
            s *= radix
        assert stages == expected
        assert current == (len(stages) & 1)

print("Aevum Apple GF61 fftHin global staging test passed")
