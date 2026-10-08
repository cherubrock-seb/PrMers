#!/usr/bin/env python3
"""The Apple scalar GF61 fftP weighting kernels must not let the weight-shift seed wrap in 32 bits.

fftP61WeightScalarApple and fftP61WeightStage1FusedApple compute word_index = (line + BIG_HEIGHT * x) * 2, which spans the whole transform
(up to NWORDS - 2), unlike the grouped fftP kernels whose word_index stays below NWORDS / NW.  Seeding the M61 weight shift with
word_index * shift in u32 arithmetic wraps at NWORDS = 2^27 whenever the shift is at least 33 (2^32 mod 61 = 57, so the shift changes
after the % 61).  The seed must reduce word_index modulo 61 first.
"""
import pathlib
import re
import sys

text = (pathlib.Path(__file__).resolve().parent.parent / "src" / "cl" / "fftp.cl").read_text()

fail = []
for name in ("fftP61WeightScalarApple", "fftP61WeightStage1FusedApple"):
    i = text.find("KERNEL(")
    m = re.search(r"KERNEL\([^)]*\) " + name + r"\(", text)
    if not m:
        fail.append(f"{name} not found")
        continue
    end = text.find("\nKERNEL(", m.end())
    body = text[m.start(): end if end != -1 else len(text)]
    seed = re.search(r"make_u64\(([^,]*?) \* (?:bigword_weight_shift_minus1|step), 0xFFFFFFFF\)", body)
    if not seed:
        fail.append(f"{name}: seed expression not found")
    elif seed.group(1).replace(" ", "") != "word_index%61":
        fail.append(f"{name}: seed multiplies {seed.group(1)!r}; it must reduce word_index % 61 first")


def u32(x):
    return x & 0xFFFFFFFF


bad_old = bad_new = 0
nwords = 1 << 27
for shift in range(61):
    for word_index in (2, nwords // 2, nwords - 2, nwords // 3 * 2, 123456786):
        exact = (word_index * shift) % 61
        bad_old += u32(word_index * shift) % 61 != exact
        bad_new += u32(word_index % 61 * shift) % 61 != exact
if bad_new:
    fail.append(f"the reduced seed is still inexact in {bad_new} cases")
if not bad_old:
    fail.append("the model no longer shows the overflow; the test is stale")

for f in fail:
    print("FAIL:", f)
if fail:
    sys.exit(1)
print(f"ok: both Apple GF61 fftP weight seeds reduce word_index mod 61 (unreduced seed is wrong in {bad_old} sampled cases at 2^27 words)")
