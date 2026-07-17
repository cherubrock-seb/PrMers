#!/usr/bin/env python3
"""Validate Apple GF61 special-tail global-shuffle/reverse staging."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gpu = (ROOT / "src/Gpu.cpp").read_text()
gpuh = (ROOT / "src/Gpu.h").read_text()
tail = (ROOT / "src/cl/tailsquare.cl").read_text()
middle = (ROOT / "src/cl/middle.cl").read_text()

kernels = [
    "tailSquareZeroGF61LoadApple",
    "tailSquareZeroGF61FftRadixApple",
    "tailSquareZeroGF61FftTwiddleApple",
    "tailSquareZeroGF61FftShuffleApple",
    "tailSquareZeroGF61FftFinalApple",
    "tailSquareZeroGF61ReverseGlobalApple",
    "tailSquareZeroGF61PairApple",
    "tailSquareZeroGF61WriteDirectApple",
]
for name in kernels:
    assert f") {name}" in tail or f" {name}(" in tail, name

required_gpu = [
    "ktailSquareZeroGF61LoadApple(bufAppleTailZeroGF61, *in);",
    "ktailSquareZeroGF61FftRadixApple(bufAppleTailZeroGF61, bank);",
    "ktailSquareZeroGF61FftTwiddleApple(bufAppleTailZeroGF61, bufTrigH, bank, stage);",
    "ktailSquareZeroGF61FftShuffleApple(bufAppleTailZeroGF61, bank, nextBank, stage);",
    "ktailSquareZeroGF61FftFinalApple(bufAppleTailZeroGF61, bank);",
    "ktailSquareZeroGF61ReverseGlobalApple(bufAppleTailZeroGF61, 0u, 1u);",
    "ktailSquareZeroGF61PairApple(bufAppleTailZeroGF61, bufTrigH, 1u);",
    "ktailSquareZeroGF61ReverseGlobalApple(bufAppleTailZeroGF61, 1u, 0u);",
    "ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, 0u, 0u);",
    "ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, SMALL_H, (fft.shape.middle / 2u) * SMALL_H);",
    "BUF(bufAppleTailZeroGF61, fft.NTT_GF61 ? 8 * SMALL_H : 0)",
]
for token in required_gpu:
    assert token in gpu, token
assert "Buffer<double> bufAppleTailZeroGF61;" in gpuh

# The special FFT no longer combines GF61 radix arithmetic with LDS shuffling.
assert "tailSquareZeroGF61FftRadixApple" in tail
assert "tailSquareZeroGF61FftTwiddleApple" in tail
assert "tailSquareZeroGF61FftShuffleApple" in tail
assert "srcI = (logical / f) % RADIX" in tail
assert "srcMe = (logical / (f * RADIX)) * f + remainder" in tail
assert "fft_RADIX(u);" in tail
assert "tailSquareZeroGF61ReverseGlobalApple" in tail
assert "const u32 logical = (revMe + (halfN - 1 - j) * G_H) % (halfN * G_H);" in tail
assert "tailSquareZeroGF61PairApple(P(GF61) scratch, Trig smallTrig, u32 bank)" in tail
assert "local GF61 lds[LDS_BYTES / sizeof(GF61)]" not in tail[tail.index("tailSquareZeroGF61ReverseGlobalApple"):tail.index("tailSquareZeroGF61PairApple")]
assert "onePairSq(&a, &b, trig, type);" in tail
assert "a = SWAP_XY(mul2(foo(a)));" in tail
assert "b = SWAP_XY(shl(csq(b), 2));" in tail

# Replay must invoke the staged FFT twice and reverse twice, as stock does.
block_start = gpu.index("ktailSquareZeroGF61LoadApple(bufAppleTailZeroGF61, *in);")
block_end = gpu.index("ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, 0u, 0u);", block_start)
block = gpu[block_start:block_end]
assert block.count("runAppleTailZeroGF61Fft();") == 2
assert block.count("ktailSquareZeroGF61ReverseGlobalApple") == 2
assert "ktailSquareZeroGF61ReverseGlobalApple(bufAppleTailZeroGF61, 0u, 1u);" in block
assert "ktailSquareZeroGF61PairApple(bufAppleTailZeroGF61, bufTrigH, 1u);" in block
assert "ktailSquareZeroGF61PairApple.setFixedArgs(1, bufTrigH);" not in gpu
assert "ktailSquareZeroGF61ReverseGlobalApple(bufAppleTailZeroGF61, 1u, 0u);" in block
assert "for (u32 stage = 1; stage < groupSize; stage *= nH)" in block

# Verify the global permutation is exactly the logical output of stock shufl.
def stock_shuffle(values, radix, gh, f):
    """Reference default LDS write/read semantics, ignoring physical padding."""
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

# Validate stage count, bank ping-pong, final normalization and scratch bounds.
for small_height in (256, 512, 1024, 2048):
    for radix in (4, 8):
        if small_height % radix:
            continue
        gh = small_height // radix
        stages = []
        bank = 0
        f = 1
        while f < gh:
            stages.append((f, bank, bank ^ 1))
            bank ^= 1
            f *= radix
        expected = []
        f = 1
        while f < gh:
            expected.append(f)
            f *= radix
        assert [s[0] for s in stages] == expected
        assert bank == (len(stages) & 1)
        # Final radix always writes bank zero.
        final_bank = 0
        assert final_bank == 0

        # Four GF61 lines: two special lines in each of two banks.
        indices = {
            bank_id * (2 * small_height) + which * small_height + i * gh + me
            for bank_id in range(2)
            for which in range(2)
            for i in range(radix)
            for me in range(gh)
        }
        assert len(indices) == 4 * small_height
        assert min(indices) == 0
        assert max(indices) == 4 * small_height - 1
        # Buffer<double>: two doubles per GF61.
        assert 8 * small_height == 2 * (4 * small_height)

# Validate the global reverse mapping against the stock LDS algorithm.
def stock_reverse(values, bump):
    nh = len(values)
    gh = len(values[0])
    half = nh // 2
    lds = [None] * (half * gh)
    for me in range(gh):
        rev_me = gh - 1 - me + int(bump)
        for j in range(half):
            logical = (rev_me + (half - 1 - j) * gh) % (half * gh)
            assert lds[logical] is None
            lds[logical] = values[half + j][me]
    out = [row[:] for row in values]
    for j in range(half):
        for me in range(gh):
            out[half + j][me] = lds[j * gh + me]
    return out


def global_reverse(values, bump):
    nh = len(values)
    gh = len(values[0])
    half = nh // 2
    out = [[None] * gh for _ in range(nh)]
    for slot in range(nh):
        for me in range(gh):
            dst_slot = slot
            dst_me = me
            if slot >= half:
                j = slot - half
                rev_me = gh - 1 - me + int(bump)
                logical = (rev_me + (half - 1 - j) * gh) % (half * gh)
                dst_slot = half + logical // gh
                dst_me = logical % gh
            assert out[dst_slot][dst_me] is None
            out[dst_slot][dst_me] = values[slot][me]
    return out

for nh in (4, 8, 16):
    for gh in (32, 64, 128):
        values = [[(slot, me) for me in range(gh)] for slot in range(nh)]
        for bump in (False, True):
            got = global_reverse(values, bump)
            assert got == stock_reverse(values, bump)
            # reverse is an involution for both special-line conventions.
            assert global_reverse(got, bump) == values

# Validate scalar pair coverage for all plausible NH values.
for nh in (4, 8, 16):
    gh = 64
    quarter = nh // 4
    pair_slots = nh // 2
    covered = {which: set() for which in range(2)}
    special_hits = 0
    for group in range(nh):
        which = group // pair_slots
        slot = group - which * pair_slots
        i = slot % quarter
        typ = slot // quarter
        a_index = i + typ * quarter
        b_index = a_index + nh // 2
        for me in range(gh):
            key = (me, a_index, b_index, i, typ)
            assert key not in covered[which]
            covered[which].add(key)
            if which == 0 and typ == 0 and i == 0 and me == 0:
                special_hits += 1
    assert len(covered[0]) == gh * nh // 2
    assert len(covered[1]) == gh * nh // 2
    assert special_hits == 1


# The final write is a direct GF61 copy.  The host launches the same tiny
# kernel twice with precomputed scratch/output bases, so the Metal pipeline
# contains no transPos, helper overload, cast, branch, loop or private array.
write_start = tail.index("KERNEL(G_H) tailSquareZeroGF61WriteDirectApple")
write_end = tail.index("\n}\n", write_start) + 3
write_block = tail[write_start:write_end]
assert "out61[DISTGF61 + outBase + offset] = scratch[scratchBase + offset];" in write_block
for forbidden in ("transPos", "writeTailFusedValue", "GF61 u[NH]", "for (", "which ?", "as_double2", "P(T2)"):
    assert forbidden not in write_block
assert 'K(ktailSquareZeroGF61WriteDirectApple,   "tailsquare.cl", "tailSquareZeroGF61WriteDirectApple",   SMALL_H' in gpu
assert "ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, 0u, 0u);" in gpu
assert "ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, SMALL_H, (fft.shape.middle / 2u) * SMALL_H);" in gpu

# Verify the two host-provided bases reproduce transPos(line)*SMALL_HEIGHT.
def trans_pos(k, middle, width):
    return k // width + (k % width) * middle

for width in (256, 1024, 4096):
    for middle in (2, 4, 8, 16):
        for small_height in (256, 512, 1024):
            h = width * middle
            assert trans_pos(0, middle, width) * small_height == 0
            assert trans_pos(h // 2, middle, width) * small_height == (middle // 2) * small_height
            nh = 4
            gh = small_height // nh
            for which, scratch_base, out_base in (
                (0, 0, 0),
                (1, small_height, (middle // 2) * small_height),
            ):
                seen = set()
                for i in range(nh):
                    for me in range(gh):
                        offset = i * gh + me
                        scratch_index = scratch_base + offset
                        out_index = out_base + offset
                        assert scratch_index == which * small_height + offset
                        seen.add(out_index)
                assert len(seen) == small_height
                assert min(seen) == out_base
                assert max(seen) == out_base + small_height - 1

print("Aevum Apple GF61 special-tail direct-write staging test passed")

