#!/usr/bin/env python3
"""Independent exact-integer checks for the Gaussian-Mersenne extension."""
from pathlib import Path


def jacobi(a: int, n: int) -> int:
    assert n > 0 and n & 1
    a %= n
    result = 1
    while a:
        while a & 1 == 0:
            a //= 2
            if n & 7 in (3, 5):
                result = -result
        a, n = n, a
        if a & 3 == 3 and n & 3 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def chi2(p: int) -> int:
    return 1 if p & 7 in (1, 7) else -1


def gm(p: int) -> int:
    if p == 2:
        return 5
    return (1 << p) - chi2(p) * (1 << ((p + 1) // 2)) + 1


def choose_base(n: int) -> int:
    for a in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        if jacobi(a, n) == -1:
            return a
    raise AssertionError("small test case has no base")


def first_admissible_factor(p: int, limit: int) -> int:
    n = gm(p)
    if n % 5 == 0 and n != 5:
        return 5
    step = 4 * p
    for k in range(1, (limit - 1) // step + 1):
        q = step * k + 1
        # Tiny exact primality check is sufficient for this regression case.
        prime = q >= 2 and all(q % d for d in range(2, int(q**0.5) + 1))
        if prime and n % q == 0:
            return q
    return 0


def chain_residue(p: int, a: int, n: int) -> int:
    """Mirror RunGaussianMersenne.cpp's optimized Euler chain modulo n."""
    m = (p + 1) // 2
    x = a % n
    if chi2(p) > 0:
        for _ in range(m - 2):
            x = x * x * a % n
        for _ in range(m - 1):
            x = x * x % n
    else:
        for _ in range(m - 1):
            x = x * x % n
        x = x * a % n
        for _ in range(m - 1):
            x = x * x % n
    return x


known_prime_exponents = (2, 3, 5, 7, 11, 19, 29, 47, 73, 79, 113)
for p in known_prime_exponents:
    n = gm(p)
    if p > 2:
        # Exact factor-lift identities used by the Aevum path.
        assert pow(2, 2 * p, n) == n - 1, p
        assert pow(2, 4 * p, n) == 1, p
        a = choose_base(n)
        expected = pow(a, (n - 1) // 2, n)
        got = chain_residue(p, a, n)
        lifted_modulus = (1 << (4 * p)) - 1
        lifted = chain_residue(p, a, lifted_modulus)
        assert lifted % n == got, (p, a, lifted % n, got)
        assert got == expected == n - 1, (p, a, got, expected)
        assert got * got % n == 1

# The optimized q=4*k*p+1 sieve must find the first factors immediately.
assert first_admissible_factor(13, 1_000_000) == 53
assert first_admissible_factor(17, 1_000_000) != 0

# Small composite exponents/norms exercise both signs and failure paths.
for p in (13, 17, 23, 31, 37, 41, 43):
    n = gm(p)
    assert pow(2, 2 * p, n) == n - 1
    assert pow(2, 4 * p, n) == 1

root = Path(__file__).resolve().parents[1]
source = (root / "src/modes/RunGaussianMersenne.cpp").read_text()
for needle in (
    "4ULL * p64",
    "deterministic Proth proof",
    "full block replay",
    "mpz_jacobi",
    "lifted_residue % n",
):
    assert needle in source, needle

print("Gaussian-Mersenne exact arithmetic and factor-lift test passed")
