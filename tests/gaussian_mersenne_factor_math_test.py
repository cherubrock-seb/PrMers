#!/usr/bin/env python3
"""Independent integer checks for Gaussian-Mersenne P-1/ECM lifting.

This test does not import PrMers code.  It verifies the identities and the exact
small factor examples used by the GPU validation script.
"""
from __future__ import annotations

from math import gcd, isqrt, lcm


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d <= isqrt(n):
        if n % d == 0:
            return False
        d += 2
    return True


def gm(p: int) -> int:
    assert is_prime(p)
    chi = 1 if p % 8 in (1, 7) else -1
    return (1 << p) - chi * (1 << ((p + 1) // 2)) + 1


def primes_to(limit: int) -> list[int]:
    return [n for n in range(2, limit + 1) if is_prime(n)]


def stage1_scalar(b1: int) -> int:
    e = 1
    for q in primes_to(b1):
        power = q
        while power * q <= b1:
            power *= q
        e *= power
    return e




def sliding_window_pow(base: int, exponent: int, modulus: int, width: int = 4) -> int:
    odd = [base % modulus]
    base2 = base * base % modulus
    for _ in range(1, 1 << (width - 1)):
        odd.append(odd[-1] * base2 % modulus)
    result = 1
    i = exponent.bit_length() - 1
    while i >= 0:
        if ((exponent >> i) & 1) == 0:
            result = result * result % modulus
            i -= 1
            continue
        low = max(0, i - width + 1)
        while low < i and ((exponent >> low) & 1) == 0:
            low += 1
        value = 0
        for bit in range(i, low - 1, -1):
            value = (value << 1) | ((exponent >> bit) & 1)
        for _ in range(low, i + 1):
            result = result * result % modulus
        result = result * odd[(value - 1) // 2] % modulus
        i = low - 1
    return result

def inv(a: int, n: int) -> int:
    return pow(a, -1, n)


def suyama(n: int, sigma: int) -> tuple[int, int]:
    sigma %= n
    if sigma < 6:
        sigma += 6
    u = (sigma * sigma - 5) % n
    v = (4 * sigma) % n
    x = pow(u, 3, n)
    z = pow(v, 3, n)
    x_aff = x * inv(z, n) % n
    num = pow((v - u) % n, 3, n) * (3 * u + v) % n
    den = 16 * x * v % n
    a24 = num * inv(den, n) % n
    return x_aff, a24


def xdouble(point: tuple[int, int], a24: int, n: int) -> tuple[int, int]:
    x, z = point
    a = (x + z) % n
    b = (x - z) % n
    aa = a * a % n
    bb = b * b % n
    e = (aa - bb) % n
    return aa * bb % n, e * (bb + a24 * e) % n


def xdbladd(p2: tuple[int, int], p3: tuple[int, int], xdiff: int,
            a24: int, n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    x2, z2 = p2
    x3, z3 = p3
    a, b = (x2 + z2) % n, (x2 - z2) % n
    c, d = (x3 + z3) % n, (x3 - z3) % n
    da, cb = d * a % n, c * b % n
    add = ((da + cb) ** 2 % n, xdiff * ((da - cb) ** 2 % n) % n)
    aa, bb = a * a % n, b * b % n
    e = (aa - bb) % n
    dbl = (aa * bb % n, e * (bb + a24 * e) % n)
    return dbl, add


def ladder(x: int, a24: int, k: int, n: int) -> tuple[int, int]:
    assert k >= 1
    p2 = (x, 1)
    p3 = xdouble(p2, a24, n)
    for bit in bin(k)[3:]:
        if bit == "1":
            p3, p2 = xdbladd(p3, p2, x, a24, n)
        else:
            p2, p3 = xdbladd(p2, p3, x, a24, n)
    return p2


# Exact sparse norm and lift identity, both Legendre-sign branches.
for p in (3, 5, 7, 11, 13, 17, 19, 23, 31, 43):
    n = gm(p)
    assert ((1 << (2 * p)) + 1) % n == 0
    assert ((1 << (4 * p)) - 1) % n == 0

# Width-4 Stage 2 exponentiation matches generic modular powering.
for modulus in (gm(13), gm(17), gm(23)):
    for base in (2, 3, 17, modulus - 2):
        for exponent in (1, 2, 3, 15, 16, 17, 12345, 65537):
            assert sliding_window_pow(base, exponent, modulus) == pow(base, exponent, modulus)

# P-1 Stage 1: adding the guaranteed 4p factor isolates 53 in G_13.
n13 = gm(13)
e13 = lcm(stage1_scalar(2), 4 * 13)
assert gcd(pow(3, e13, n13) - 1, n13) == 53

# P-1 low-memory Stage 2 continuation: the extra prime 3 finds 277 in G_23.
n23 = gm(23)
h23 = pow(3, lcm(stage1_scalar(2), 4 * 23), n23)
assert gcd(h23 - 1, n23) == 1
h23 = pow(h23, 3, n23)
assert gcd(h23 - 1, n23) == 277

# ECM Stage 1 known deterministic curve: G_17, sigma=7, B1=50 -> factor 137.
n17 = gm(17)
x, a24 = suyama(n17, 7)
_, z = ladder(x, a24, stage1_scalar(50), n17)
assert gcd(z, n17) == 137

# ECM Stage 2 product-exponent continuation: sigma=14, B1=2 then prime 3.
x, a24 = suyama(n17, 14)
x1, z1 = ladder(x, a24, stage1_scalar(2), n17)
assert gcd(z1, n17) == 1
x1 = x1 * inv(z1, n17) % n17
_, z2 = ladder(x1, a24, 3, n17)
assert gcd(z2, n17) == 137

print("Gaussian-Mersenne lifted P-1/ECM mathematics passed")
