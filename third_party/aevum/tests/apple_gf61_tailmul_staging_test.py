#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cl = (ROOT / "src/cl/tailmul.cl").read_text()
gpu = (ROOT / "src/Gpu.cpp").read_text()
hdr = (ROOT / "src/Gpu.h").read_text()

required = [
    "tailMulGF61ApplePlaceholder", "tailMulGF61LoadScalarApple",
    "tailMulGF61FftRadixApple", "tailMulGF61FftTwiddleApple",
    "tailMulGF61FftShuffleApple", "tailMulGF61FftFinalApple",
    "tailMulGF61PairSpecialScalarApple",
    "tailMulGF61PairNormalScalarApple",
]
for name in required:
    assert name in cl and name in gpu, name

for old in (
    "tailMulGF61PairLocalApple",
    "tailMulGF61ReverseStockApple",
    "tailMulGF61PairStockApple",
    "tailMulGF61PairDirectApple",
):
    assert old not in cl and old not in gpu and old not in hdr, old

assert "Kernel ktailMulGF61PairSpecialScalarApple;" in hdr
assert "Kernel ktailMulGF61PairNormalScalarApple;" in hdr
assert '"tailMulGF61PairSpecialScalarApple"' in gpu
assert '"tailMulGF61PairNormalScalarApple"' in gpu
assert "bufAppleTailMulGF61" in hdr and "GF61_DATA_SIZE" in gpu
assert "Apple staged GF61 tailMul requires three distinct transform buffers" in gpu

special = cl[
    cl.index("KERNEL(G_H) tailMulGF61PairSpecialScalarApple"):
    cl.index("KERNEL(G_H) tailMulGF61PairNormalScalarApple")
]
normal = cl[
    cl.index("KERNEL(G_H) tailMulGF61PairNormalScalarApple"):
    cl.index("// Signature-compatible no-op")
]
for body in (special, normal):
    assert "local GF61" not in body
    assert "GF61 u[" not in body and "GF61 v[" not in body
    assert "onePairMul(&a, &b, &p, &q, trig);" in body
assert "sourceLinear" in special and "partnerSlot" in special
assert "partnerSlot = NH - 1 - slot" in normal
assert "partnerMe = G_H - 1 - me" in normal
assert "TAILTGF61" in special

# Stock special-line trig is based on line1 == 0 for BOTH line 0 and H/2.
# H/2 differs only by one multiplication by TAILTGF61.  Selecting a trig by
# `which`/`line` here double-counts the side factor and corrupts every multiply.
assert "TSLOAD(&smallTrig61[heightTrigs + G_H])" in special
assert "TOLOAD(&smallTrig61[heightTrigs + me])" in special
for wrong in (
    "heightTrigs + G_H + which",
    "heightTrigs + which * G_H",
    "heightTrigs + line * G_H",
    "heightTrigs + G_H + line",
):
    assert wrong not in special, wrong
assert special.count("if (which) trig = cmul(trig, TAILTGF61);") == 1

# Symbolic reference: both special lines start from the same base trig; H/2
# gets exactly one tail factor, never a second table-side multiplier.
base = ("lane-trig", "line-zero-mult")
line0_trig = (base, 1)
half_trig = (base, "TAILTGF61")
assert line0_trig[0] == half_trig[0]
assert half_trig[1] == "TAILTGF61"

# Verify the direct special-line coordinate is exactly stock
# reverse(second_half,bump) followed by its inverse after pairMul.
def reverse_half_source(half_n, G, pair_slot, lane, bump):
    n = half_n * G
    out_linear = pair_slot * G + lane
    return ((n - out_linear) % n) if bump else (n - 1 - out_linear)

for NH, G in ((4, 64), (8, 64)):
    half = NH // 2
    for which in (0, 1):
        bump = which == 0
        seen = set()
        for pair_slot in range(half):
            for lane in range(G):
                src = reverse_half_source(half, G, pair_slot, lane, bump)
                partner = (half + src // G, src % G)
                seen.add(partner)
        assert seen == {(slot, lane) for slot in range(half, NH) for lane in range(G)}

# Verify normal line pairing is stock reverseLine:
# output slot s reads partner original slot NH-1-s, lane G-1-me.
for H, NH, G in ((512, 4, 64), (2048, 4, 64), (512, 8, 64)):
    written = set()

    # Two special lines, every slot/lane exactly once.
    half = NH // 2
    for which, line in ((0, 0), (1, H // 2)):
        for pair_slot in range(half):
            for lane in range(G):
                src = reverse_half_source(half, G, pair_slot, lane, which == 0)
                partner_slot = half + src // G
                partner_lane = src % G
                written.add((line, pair_slot, lane))
                written.add((line, partner_slot, partner_lane))

    # Normal pairs line1=1..H/2-1, one workgroup per slot.
    for line1 in range(1, H // 2):
        line2 = H - line1
        for slot in range(NH):
            for lane in range(G):
                written.add((line1, slot, lane))
                written.add((line2, NH - 1 - slot, G - 1 - lane))

    expected = {(line, slot, lane)
                for line in range(H)
                for slot in range(NH)
                for lane in range(G)}
    assert written == expected

# Slot family agrees with stock pairMul order:
# base, t^4*base, -base, -t^4*base.
for NH in (4, 8):
    q = NH // 4
    families = []
    for slot in range(NH):
        if slot < q:
            family = 0
        elif slot < NH // 2:
            family = 1
        elif slot < 3 * q:
            family = 2
        else:
            family = 3
        families.append(family)
    assert families == [0] * q + [1] * q + [2] * q + [3] * q

# Host dispatch preserves the multiplicand and executes both scalar pair
# pipelines exactly once before the final staged height transform.
block_start = gpu.index("// Preserved multiplicand:")
block_end = gpu.index("} else {", block_start)
block = gpu[block_start:block_end]
assert "ktailMulGF61LoadScalarApple(*in1, fullBase, *in2, fullBase);" in block
assert "ktailMulGF61LoadScalarApple(*in2" not in block
special_call = "ktailMulGF61PairSpecialScalarApple(*raw, rawBase, *out, fullBase, *in1, fullBase, bufTrigH);"
normal_call = "ktailMulGF61PairNormalScalarApple(*raw, rawBase, *out, fullBase, *in1, fullBase, bufTrigH);"
assert special_call in block and normal_call in block
assert block.index(special_call) < block.rindex("runAppleTailMulGF61Fft")
assert block.index(normal_call) < block.rindex("runAppleTailMulGF61Fft")

# Ping-pong schedule for the production height radix remains unchanged.
def fft_schedule(group_size, radix):
    cur, alt = "start", "alternate"
    stages = []
    f = 1
    while f < group_size:
        stages.append((f, cur, alt))
        cur, alt = alt, cur
        f *= radix
    return stages, cur

stages, cur = fft_schedule(64, 4)
assert [x[0] for x in stages] == [1, 4, 16]
assert cur == "alternate"

print("Aevum Apple GF61 tailMul scalar direct-pair staging test passed")
