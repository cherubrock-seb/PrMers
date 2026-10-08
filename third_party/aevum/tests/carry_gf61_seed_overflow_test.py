#!/usr/bin/env python3
"""The unfused carry kernels must not let the GF61 weight-shift seed wrap in 32 bits.

carry.cl seeds the counter with make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF).  Both factors are u32, and word_index spans
the whole transform (up to NWORDS - 2), so for an M61 lane at NWORDS = 2^27 the product exceeds 2^32 whenever the shift is at least 33 and
the wrapped value is no longer congruent modulo 61.  The seed must reduce word_index modulo 61 first (the GF31 lanes cannot wrap).

Part 1 checks the source; part 2 emulates the u32/u64 arithmetic for the shapes that matter.
"""
import pathlib
import re
import sys

src = (pathlib.Path(__file__).resolve().parent.parent / "src" / "cl" / "carry.cl").read_text().split("\n")

fail = []
mod = {}
checked = 0
for n, line in enumerate(src, 1):
    m = re.search(r"const u32 (m31_|m61_)?bigword_weight_shift = .* % (\d+);", line)
    if m:
        mod[m.group(1) or ""] = int(m.group(2))
        continue
    m = re.search(r"(m31_|m61_)?combo_counter = comboFracBits\(word_index\) \+ make_u64\((.*?) \* (m31_|m61_)?bigword_weight_shift_minus1, 0xFFFFFFFF\)", line)
    if m:
        prefix = m.group(1) or ""
        if mod.get(prefix) == 61:
            checked += 1
            if m.group(2).replace(" ", "") != "word_index%61":
                fail.append(f"carry.cl:{n}: GF61 seed multiplies {m.group(2)!r}, it must reduce word_index % 61 first")
if checked < 3:
    fail.append(f"only {checked} GF61 seeds found, the scanner is broken")


def u32(x):
    return x & 0xFFFFFFFF


bad_old = bad_new = 0
nwords = 1 << 27
for shift_minus1 in range(61):
    for word_index in (2, nwords // 2, nwords - 2, nwords // 3 * 2, 123456786):
        exact = (word_index * shift_minus1) % 61
        if u32(word_index * shift_minus1) % 61 != exact:
            bad_old += 1
        if u32(word_index % 61 * shift_minus1) % 61 != exact:
            bad_new += 1
if bad_new:
    fail.append(f"the reduced seed is still inexact in {bad_new} cases")
if not bad_old:
    fail.append("the model no longer shows the overflow; the test is stale")

for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print(f"ok: {checked} GF61 seeds reduce word_index mod 61 (unreduced seed is wrong in {bad_old} sampled cases at 2^27 words)")
