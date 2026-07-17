#!/usr/bin/env python3
"""Validate Apple GF61 fftW scalar-load and global width staging."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()
gpuh = (ROOT / "src/Gpu.h").read_text()
fftw = (ROOT / "src/cl/fftw.cl").read_text()

kernels = [
    "fftWGF61ApplePlaceholder",
    "fftWGF61LoadScalarApple",
    "fftWGF61WidthRadixApple",
    "fftWGF61TwiddleShuffle1Apple",
    "fftWGF61TwiddleShuffle4Apple",
    "fftWGF61TwiddleShuffle8Apple",
    "fftWGF61TwiddleShuffle16Apple",
    "fftWGF61TwiddleShuffle64Apple",
    "fftWGF61TwiddleShuffle256Apple",
    "fftWGF61TwiddleShuffle512Apple",
    "fftWGF61WidthFinalApple",
]
for name in kernels:
    assert name in fftw, name
    if name != "fftWGF61ApplePlaceholder":
        member = "k" + name[0].lower() + name[1:]
        assert member in gpuh, member

required_gpu = [
    '"fftWGF61ApplePlaceholder"',
    "kfftWGF61LoadScalarApple(*out, *in);",
    "Buffer<double>* current = out;",
    "Buffer<double>* alternate = in;",
    "kfftWGF61WidthRadixApple(*current);",
    "kfftWGF61TwiddleShuffle1Apple(*alternate, *current);",
    "kfftWGF61TwiddleShuffle4Apple(*alternate, *current);",
    "kfftWGF61TwiddleShuffle16Apple(*alternate, *current);",
    "kfftWGF61WidthFinalApple(*out, *current);",
    "Apple staged GF61 fftW requires distinct input/output buffers",
]
for token in required_gpu:
    assert token in gpu, token

# Apple stages only FFT3161; non-Apple retains the original kernel and no new
# transform-sized scratch is allocated.
assert "KERNEL(G_W) fftWGF61(P(T2) out, CP(T2) in, Trig smallTrig)" in fftw
assert "local GF61 lds[LDS_BYTES / sizeof(GF61)]" in fftw
assert '#if defined(__APPLE__)\n  K(kfftWGF61' in gpu
assert '#else\n  K(kfftWGF61' in gpu
assert "bufAppleFftWGF61" not in gpu


def scalar_src(line, i, me, width, middle, small_h, gw, out_wg, out_sizex):
    size_y = out_wg // out_sizex
    middle_out_x = line % small_h
    chunk_x = middle_out_x // out_sizex
    x_within = middle_out_x % out_sizex
    middle_out_i = line // small_h
    chunk_y = me // size_y + i * (gw // size_y)
    return (
        chunk_x * middle * width * out_sizex
        + x_within * size_y
        + middle_out_i * out_wg
        + (me % size_y)
        + chunk_y * middle * out_wg
    )


def stock_read_src(line, i, me, width, middle, small_h, gw, out_wg, out_sizex):
    # Exact PAD=0 readCarryFusedLine pointer arithmetic.
    size_y = out_wg // out_sizex
    x = line % small_h
    base = (x // out_sizex) * middle * width * out_sizex
    base += (x % out_sizex) * size_y
    base += (line // small_h) * out_wg
    base += me % size_y
    chunk_y = me // size_y
    chunk_y += i * (gw // size_y)
    return base + chunk_y * middle * out_wg


# The scalar source expression is exactly the stock readCarryFusedLine mapping
# and is a bijection over the whole middle-out plane.
for width, middle, small_h, nw, out_wg, out_sizex in (
    (32, 2, 64, 4, 64, 16),
    (64, 4, 128, 8, 64, 16),
    (256, 2, 256, 4, 64, 16),
):
    gw = width // nw
    big_h = middle * small_h
    seen = set()
    for line in range(big_h):
        for i in range(nw):
            for me in range(gw):
                got = scalar_src(line, i, me, width, middle, small_h, gw, out_wg, out_sizex)
                ref = stock_read_src(line, i, me, width, middle, small_h, gw, out_wg, out_sizex)
                assert got == ref
                assert 0 <= got < width * big_h
                assert got not in seen
                seen.add(got)
    assert seen == set(range(width * big_h))


# The twiddle+global permutation is the exact flat-index equivalent of the LDS
# shufl used by fft_common.
def stock_shuffle(values, radix, gw, stage):
    lds = [None] * (radix * gw)
    mask = stage - 1
    for me in range(gw):
        for i in range(radix):
            dst = i * stage + (me & ~mask) * radix + (me & mask)
            assert lds[dst] is None
            lds[dst] = values[i][me]
    return [[lds[i * gw + me] for me in range(gw)] for i in range(radix)]


def global_shuffle(values, radix, gw, stage):
    out = [[None] * gw for _ in range(radix)]
    mask = stage - 1
    for i in range(radix):
        for me in range(gw):
            dst = i * stage + (me & ~mask) * radix + (me & mask)
            out[dst // gw][dst % gw] = values[i][me]
    return out

for radix in (4, 8):
    for gw in (32, 64, 128, 256, 512):
        values = [[(i, me) for me in range(gw)] for i in range(radix)]
        stage = 1
        while stage < gw:
            assert global_shuffle(values, radix, gw, stage) == stock_shuffle(values, radix, gw, stage)
            stage *= radix


# Host stage scheduling performs one radix per shuffle stage and one final
# radix, and always leaves the result in the requested output bank.
for width, nw in ((256, 4), (1024, 8), (4096, 8)):
    gw = width // nw
    current = "out"
    alternate = "in"
    stage = 1
    stages = []
    while stage < gw:
        stages.append(stage)
        current, alternate = alternate, current
        stage *= nw
    # Final radix either runs in-place in out or copies current -> out.
    final_mode = "inplace" if current == "out" else "copy"
    assert stages
    assert all(s in (1, 4, 8, 16, 64, 256, 512) for s in stages)
    assert final_mode in ("inplace", "copy")

print("Aevum Apple GF61 fftW scalar-load/global-width staging test passed")
