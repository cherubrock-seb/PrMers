#!/usr/bin/env python3
"""Validate Apple GF61 normal-line tail-square global staging."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()
gpuh = (ROOT / "src/Gpu.h").read_text()
tail = (ROOT / "src/cl/tailsquare.cl").read_text()

kernels = [
    "tailSquareGF61LoadScalarApple",
    "tailSquareGF61FftRadixApple",
    "tailSquareGF61FftTwiddleApple",
    "tailSquareGF61FftShuffleApple",
    "tailSquareGF61FftFinalApple",
    "tailSquareGF61ReverseCrossApple",
    "tailSquareGF61PairApple",
    "tailSquareGF61ApplePlaceholder",
]
for name in kernels:
    assert f" {name}(" in tail, name
    assert f"k{name[0].lower() + name[1:]}" in gpuh or name == "tailSquareGF61ApplePlaceholder"

required_gpu = [
    '"tailSquareGF61ApplePlaceholder"',
    "ktailSquareGF61LoadScalarApple(*out, *in);",
    "ktailSquareGF61FftRadixApple(*current);",
    "ktailSquareGF61FftTwiddleApple(*current, bufTrigH, stage);",
    "ktailSquareGF61FftShuffleApple(*current, *next, stage);",
    "ktailSquareGF61FftFinalApple(*current, *out);",
    "ktailSquareGF61ReverseCrossApple(*out, *in);",
    "ktailSquareGF61PairApple(*in, bufTrigH);",
    "ktailSquareGF61ReverseCrossApple(*in, *out);",
]
for token in required_gpu:
    assert token in gpu, token

# Apple reuses the existing input/output transform buffers as ping-pong banks.
assert "Buffer<double> *current = out;" in gpu
assert "Buffer<double> *next = current == out ? in : out;" in gpu
assert "runAppleTailGF61Fft();" in gpu
block_start = gpu.index("ktailSquareGF61LoadScalarApple(*out, *in);")
block_end = gpu.index("} else {\n            ktailSquareGF61(*out, *in);", block_start)
block = gpu[block_start:block_end]
assert block.count("runAppleTailGF61Fft();") == 2
assert "bufAppleTailGF61" not in gpu

# The stock double-wide kernel remains in source and remains selected outside
# Apple.  Apple only avoids creating it by binding the signature-compatible
# placeholder to the legacy member.
assert "KERNEL(G_H * 2) tailSquareGF61" in tail
assert "local GF61 lds[2 * LDS_BYTES / sizeof(GF61)]" in tail
assert "#if defined(__APPLE__)\n  K(ktailSquareGF61" in gpu
assert "#else\n  K(ktailSquareGF61" in gpu


def trans_pos(line, middle, width):
    return line // width + (line % width) * middle


def normal_lines(H):
    half = H // 2
    return list(range(1, half)) + list(range(half + 1, H))


def ordinal_line(ordinal, H):
    half = H // 2
    return ordinal + 1 if ordinal < half - 1 else ordinal + 2


# Ordinal mapping is bijective and excludes only 0 and H/2.  Final line bases
# are also disjoint because transPos is a permutation.
for width in (64, 256, 1024):
    for middle in (2, 4, 8, 16):
        H = width * middle
        lines = [ordinal_line(o, H) for o in range(H - 2)]
        assert lines == normal_lines(H)
        assert len(set(lines)) == H - 2
        mapped = [trans_pos(line, middle, width) for line in lines]
        assert len(set(mapped)) == H - 2
        special = {trans_pos(0, middle, width), trans_pos(H // 2, middle, width)}
        assert not (set(mapped) & special)


# Scalar PAD=0 load mapping exactly matches one readTailFusedLine iteration and
# covers every normal-line GF61 value without collision.
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

# Exhaustive on compact representative shapes; larger plans use the same
# affine/block mapping and are sampled below to keep this source test fast.
for width, middle, small_h, nh, in_sizex, in_wg in (
    (32, 2, 64, 4, 16, 64),
    (64, 4, 128, 8, 16, 64),
):
    H = width * middle
    gh = small_h // nh
    all_indices = set()
    for line in range(H):
        for slot in range(nh):
            for me in range(gh):
                idx = load_src(line, slot, me, width, middle, small_h, nh, in_sizex, in_wg)
                assert idx not in all_indices
                all_indices.add(idx)
    assert all_indices == set(range(H * small_h))

for width, middle, small_h, nh, in_sizex, in_wg in (
    (256, 2, 256, 4, 16, 64),
    (1024, 8, 512, 8, 16, 64),
    (4096, 16, 1024, 8, 16, 64),
):
    H = width * middle
    gh = small_h // nh
    probes = [0, 1, H // 2 - 1, H // 2, H - 2, H - 1]
    seen = set()
    for line in probes:
        for slot in range(nh):
            for me in (0, gh // 2, gh - 1):
                idx = load_src(line, slot, me, width, middle, small_h, nh, in_sizex, in_wg)
                assert 0 <= idx < H * small_h
                assert idx not in seen
                seen.add(idx)


# Global shuffling reproduces the logical LDS shufl permutation exactly.
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

for radix in (4, 8):
    for gh in (32, 64, 128, 256):
        values = [[(i, me) for me in range(gh)] for i in range(radix)]
        f = 1
        while f < gh:
            assert global_shuffle(values, radix, gh, f) == stock_shuffle(values, radix, gh, f)
            f *= radix


# Cross-line reverse is the exact global equivalent of revCrossLine and is an
# involution.  First-half values stay in their own line; second-half values
# cross to H-line and reverse both slot and lane.
def reverse_cross(lines, H, nh, gh):
    out = {}
    for line, values in lines.items():
        for slot in range(nh):
            for me in range(gh):
                dst_line, dst_slot, dst_me = line, slot, me
                if slot >= nh // 2:
                    j = slot - nh // 2
                    dst_line = H - line
                    dst_slot = nh // 2 + (nh // 2 - 1 - j)
                    dst_me = gh - 1 - me
                key = (dst_line, dst_slot, dst_me)
                assert key not in out
                out[key] = values[slot][me]
    return {
        line: [[out[(line, slot, me)] for me in range(gh)] for slot in range(nh)]
        for line in lines
    }

for H in (64, 128, 512):
    for nh in (4, 8, 16):
        gh = 32
        values = {
            line: [[(line, slot, me) for me in range(gh)] for slot in range(nh)]
            for line in normal_lines(H)
        }
        once = reverse_cross(values, H, nh, gh)
        twice = reverse_cross(once, H, nh, gh)
        assert twice == values
        for line in normal_lines(H):
            pair = H - line
            for slot in range(nh // 2):
                assert once[line][slot] == values[line][slot]
            for j in range(nh // 2):
                dst_slot = nh // 2 + (nh // 2 - 1 - j)
                for me in range(gh):
                    assert once[pair][dst_slot][gh - 1 - me] == values[line][nh // 2 + j][me]


# Pair coverage and trig multiplier indexing match the two stock double-wide
# halves.  Multiplier indexes 2..H-1 are covered exactly once per normal line.
for H in (64, 128, 512):
    for nh in (4, 8, 16):
        gh = 64
        covered = set()
        trig_indexes = []
        quarter = nh // 4
        for ordinal, line in enumerate(normal_lines(H)):
            side = int(line > H // 2)
            line_u = H - line if side else line
            trig_indexes.append(line_u * 2 + side)
            for pair_slot in range(nh // 2):
                i = pair_slot % quarter
                typ = pair_slot // quarter
                a = i + typ * quarter
                b = a + nh // 2
                for me in range(gh):
                    key = (line, me, a, b, i, typ)
                    assert key not in covered
                    covered.add(key)
        assert len(covered) == (H - 2) * gh * nh // 2
        assert sorted(trig_indexes) == list(range(2, H))

# Verify the current constructor launch sizes reduce to exact workgroup counts.
for H in (64, 128, 512):
    for small_h, nh in ((256, 4), (512, 8)):
        gh = small_h // nh
        normal_threads = (H - 2) * gh
        assert normal_threads * nh == (H - 2) * small_h
        assert normal_threads * nh // 2 == (H - 2) * small_h // 2

print("Aevum Apple GF61 normal-line main-tail global staging test passed")
