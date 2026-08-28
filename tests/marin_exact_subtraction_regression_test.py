#!/usr/bin/env python3
from pathlib import Path
import random

s = Path("include/marin/engine_gpu.h").read_text()
v = Path("include/core/Version.hpp").read_text()
for needle in (
    "subtract_reg_group_p1",
    "subtract_reg_group_scan",
    "subtract_reg_group_apply",
    "engine::addsub(sum_out, diff_out, a, b);",
    "engine::addsub_copy(sum, diff, sum_copy, diff_copy, a, b);",
):
    assert needle in s, needle
assert any(version in v for version in (
    "4.20.88-alpha-v99.99-marin-exact-subtraction",
    "4.20.89-alpha-v100.00-gm-ecm-special32",
    "4.20.89-alpha-v100.01-gm-ecm-special32",
))

def group_sub(y, x, widths, group_digits):
    n = len(y)
    assert n % group_digits == 0
    ngr = n // group_digits
    r = [0] * n
    maps = []
    for g in range(ngr):
        b = 0
        eq = 1
        lo, hi = g * group_digits, (g + 1) * group_digits
        for k in range(lo, hi):
            if y[k] != x[k]:
                eq = 0
            B = 1 << widths[k]
            sub = x[k] + b
            borrow = 1 if y[k] < sub else 0
            r[k] = y[k] - sub + (B if borrow else 0)
            b = borrow
        maps.append((b, eq))

    incoming = [0] * ngr
    b = 0
    for g, (gen, eq) in enumerate(maps):
        incoming[g] = b
        b = gen | (eq & b)
    if b:
        b = 1
        for g, (gen, eq) in enumerate(maps):
            incoming[g] = b
            b = gen | (eq & b)

    for g, inc in enumerate(incoming):
        if not inc:
            continue
        b = 1
        lo, hi = g * group_digits, (g + 1) * group_digits
        for k in range(lo, hi):
            if not b:
                break
            B = 1 << widths[k]
            borrow = 1 if r[k] < b else 0
            r[k] = r[k] - b + (B if borrow else 0)
            b = borrow
    return r

def value(d, widths):
    out = 0
    shift = 0
    for x, w in zip(d, widths):
        out |= x << shift
        shift += w
    return out, shift

rng = random.Random(0x999EC0)
for n in (8, 16, 32, 64):
    for gd in (2, 4, 8, 16):
        if n % gd:
            continue
        for _ in range(2000):
            widths = [rng.randint(2, 10) for _ in range(n)]
            y = [rng.randrange(1 << w) for w in widths]
            x = [rng.randrange(1 << w) for w in widths]
            r = group_sub(y, x, widths, gd)
            Y, q = value(y, widths)
            X, _ = value(x, widths)
            R, _ = value(r, widths)
            M = (1 << q) - 1
            if R == M:
                R = 0
            assert R == (Y - X) % M

print("Marin v99.99 exact subtraction regression: OK")
