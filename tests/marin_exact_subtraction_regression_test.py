#!/usr/bin/env python3
"""Model of the Marin exact group subtraction (subtract_reg_group_p1/scan/apply).

The model mirrors the OpenCL kernels embedded in include/marin/engine_gpu.h and the
sbc_reg primitive in kernels/marin.cl, and checks them against exact integer arithmetic
modulo 2^q - 1 for inputs that satisfy the real digit invariant of the engine: digits are
non-negative integers that are NOT necessarily below their base.  The carry kernels leave
lane s3 of each 4-digit block as d + c (adc4), so digits equal to 2^w are routine; the model
also stresses digits up to 2^(w+1) - 1 and multi-unit borrows crossing group boundaries.
"""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
engine_src = (ROOT / "include/marin/engine_gpu.h").read_text()
kernel_src = (ROOT / "kernels/marin.cl").read_text()
header_src = (ROOT / "include/marin/ocl/kernel.h").read_text()

for needle in (
    "subtract_reg_group_p1",
    "subtract_reg_group_scan",
    "subtract_reg_group_apply",
    "engine::addsub(sum_out, diff_out, a, b);",
    "engine::addsub_copy(sum, diff, sum_copy, diff_copy, a, b);",
    "carry[ngr + grp] = (r0_high != 0ul) ? ~0ul : r0;",
    "b = gen[g] + ((b > r0[g]) ? 1ul : 0ul);",
    "std::max(n / 4, 3 * carry_groups)",
):
    assert needle in engine_src, needle
# The group borrow must be derived from the result value, never from the digit representation.
assert "if (yv != xv) equal = 0u;" not in engine_src
assert "INLINE uint64 sbc_reg(const uint64 lhs, const uint64 rhs, const uint_8 width, uint64 * const borrow)" in kernel_src
assert "INLINE uint64 sbc_reg(const uint64 lhs, const uint64 rhs, const uint_8 width, uint64 * const borrow)" in header_src

SAT = (1 << 64) - 1


def sbc_reg(lhs, rhs, w, borrow):
    """kernels/marin.cl sbc_reg: exact digit subtraction with a multi-unit borrow."""
    sub = rhs + borrow
    if lhs >= sub:
        return lhs - sub, 0
    t = sub - lhs
    b = -(-t // (1 << w))
    return (b << w) - t, b


def group_sub(y, x, widths, group_digits):
    n = len(y)
    assert n % group_digits == 0
    ngr = n // group_digits
    r = [0] * n
    gen = [0] * ngr
    r0 = [0] * ngr

    # p1: independent groups, borrow in 0. R0 is the value of the result digits, saturated
    # to 2^64 - 1.
    for g in range(ngr):
        b = 0
        value = 0
        shift = 0
        for k in range(g * group_digits, (g + 1) * group_digits):
            r[k], b = sbc_reg(y[k], x[k], widths[k], b)
            value += r[k] << shift
            shift += widths[k]
        gen[g] = b
        r0[g] = min(value, SAT)

    # scan: cyclic borrow, iterate until the wrapped borrow is stable
    incoming = [0] * ngr
    wrap = 0
    for _ in range(64):
        b = wrap
        for g in range(ngr):
            incoming[g] = b
            b = gen[g] + (1 if b > r0[g] else 0)
        if b == wrap:
            break
        wrap = b
    else:
        raise AssertionError("scan did not converge")

    # apply
    for g in range(ngr):
        b = incoming[g]
        for k in range(g * group_digits, (g + 1) * group_digits):
            if b == 0:
                break
            r[k], b = sbc_reg(r[k], 0, widths[k], b)
    return r


def value(d, widths):
    out = 0
    shift = 0
    for v, w in zip(d, widths):
        out += v << shift
        shift += w
    return out, shift


def check(y, x, widths, gd):
    r = group_sub(y, x, widths, gd)
    Y, q = value(y, widths)
    X, _ = value(x, widths)
    R, _ = value(r, widths)
    M = (1 << q) - 1
    assert R % M == (Y - X) % M, (y, x, widths, gd, r)
    for k in range(len(r)):
        # a result digit is non-negative and never worse than the input digit
        assert 0 <= r[k] <= max(y[k], (1 << widths[k]) - 1), (k, r[k], y[k], widths[k])


rng = random.Random(0x999EC0)
cases = 0
for n in (4, 8, 16, 32, 64):
    for gd in (2, 4, 8, 16):
        if n % gd:
            continue
        for _ in range(1500):
            widths = [rng.randint(2, 10) for _ in range(n)]
            B = [1 << w for w in widths]
            y = [rng.randrange(B[k]) for k in range(n)]
            x = [rng.randrange(B[k]) for k in range(n)]
            check(y, x, widths, gd)
            cases += 1

            # producer-style digits: lane s3 of each block may equal its base
            y2 = [B[k] if (k % 4 == 3 and rng.random() < 0.5) else y[k] for k in range(n)]
            x2 = [B[k] if (k % 4 == 3 and rng.random() < 0.5) else x[k] for k in range(n)]
            check(y2, x2, widths, gd)
            check(x2, y2, widths, gd)
            cases += 2

            # stress bound: any digit anywhere up to 2^(w+1) - 1
            y3 = [rng.randrange(2 * B[k]) for k in range(n)]
            x3 = [rng.randrange(2 * B[k]) for k in range(n)]
            check(y3, x3, widths, gd)
            cases += 1

        # targeted: borrow entering a lane whose subtrahend digit equals the base
        widths = [rng.randint(2, 10) for _ in range(n)]
        B = [1 << w for w in widths]
        for k in range(1, n):
            y = [0] * n
            x = [0] * n
            x[k] = B[k]
            x[k - 1] = 5
            y[k - 1] = 1
            check(y, x, widths, gd)
            cases += 1

        # targeted: groups equal in value but not in representation must propagate a borrow
        ngr = n // gd
        if ngr >= 3 and gd >= 2:
            x = [rng.randrange(B[k]) for k in range(n)]
            y = list(x)
            for g in range(1, ngr - 1):
                k = g * gd
                x[k] = B[k]
                x[k + 1] = 0
                y[k] = 0
                y[k + 1] = 1
            x[0] = 5
            y[0] = 1
            check(y, x, widths, gd)
            # wrap through group 0: the top group generates, group 0 is equal in value only
            x = [rng.randrange(B[k]) for k in range(n)]
            y = list(x)
            x[0] = B[0]
            x[1] = 0
            y[0] = 0
            y[1] = 1
            x[n - 1] = 5
            y[n - 1] = 1
            check(y, x, widths, gd)
            cases += 2

        # targeted: every digit of x at its base (X > 2^q, multi-unit wrapped borrow), y small
        x = [B[k] for k in range(n)]
        y = [0] * n
        check(y, x, widths, gd)
        y[0] = 1
        check(y, x, widths, gd)
        x = [2 * B[k] - 1 for k in range(n)]
        check(y, x, widths, gd)
        check([0] * n, x, widths, gd)
        cases += 4

        # 0 - 1, 1 - 7, y - y, 0 - 0
        check([0] * n, [1] + [0] * (n - 1), widths, gd)
        check([1] + [0] * (n - 1), [7] + [0] * (n - 1), widths, gd)
        y = [rng.randrange(2 * B[k]) for k in range(n)]
        check(y, y, widths, gd)
        check([0] * n, [0] * n, widths, gd)
        cases += 4

print(f"Marin exact subtraction regression ({cases} cases): OK")
