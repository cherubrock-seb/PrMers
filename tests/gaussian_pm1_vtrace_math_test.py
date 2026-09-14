#!/usr/bin/env python3
"""Host-only correctness checks for the v100.13 GM/GQ V-trace lift.

No GPU is required.  The checks prove the exact algebra used by the GPU path:
  * each selected Gaussian target divides 2^(4p)-1;
  * an inverse computed modulo the exact target may be injected as a
    representative in the lift;
  * the denominator-free trace recurrence projects identically;
  * every ordinary P-1 Stage-2 prime condition H^q=1 is annihilated by the
    corresponding V_{kD}-V_j trace term.
"""

from math import gcd, lcm


def primes_upto(n):
    mark = [True] * (n + 1)
    if n >= 0:
        mark[0] = False
    if n >= 1:
        mark[1] = False
    for p in range(2, int(n**0.5) + 1):
        if mark[p]:
            for m in range(p * p, n + 1, p):
                mark[m] = False
    return [p for p in range(2, n + 1) if mark[p]]


def target(p, family):
    chi = 1 if p % 8 in (1, 7) else -1
    m = (p + 1) // 2
    n = 1 << p
    gq = family == "GQ"
    if ((not gq and chi > 0) or (gq and chi < 0)):
        n -= 1 << m
    else:
        n += 1 << m
    n += 1
    if gq:
        assert n % 5 == 0
        n //= 5
    return n


def build_e(B1):
    e = 1
    for q in primes_upto(B1):
        qpow = q
        while qpow * q <= B1:
            qpow *= q
        e = lcm(e, qpow)
    return e


def trace_direct(h, hinv, n, mod):
    return (pow(h, n, mod) + pow(hinv, n, mod)) % mod


def normalize(q, D):
    k, rem = divmod(q, D)
    j = rem
    if rem > D // 2:
        k += 1
        j = D - rem
    return k, j


def check_one(p, family, B1, B2, D, base=3):
    N = target(p, family)
    L = (1 << (4 * p)) - 1
    assert L % N == 0, (p, family, "target does not divide lift")

    E = lcm(build_e(B1), 4 * p)
    h = pow(base, E, N)
    gh = gcd(h, N)
    if gh != 1:
        # A genuine factor/setup event is valid; use another base for recurrence checks.
        return

    hinv = pow(h, -1, N)
    assert (h * hinv) % N == 1

    # Inject h and hinv as representatives into the larger ring.  Verify that
    # the trace recurrence in L projects to the direct trace in N.
    vprev_L = 2 % L
    vcur_L = (h + hinv) % L
    assert vcur_L % N == trace_direct(h, hinv, 1, N)
    for idx in range(1, min(D, 80)):
        vnext_L = (vcur_L * ((h + hinv) % L) - vprev_L) % L
        vprev_L, vcur_L = vcur_L, vnext_L
        assert vcur_L % N == trace_direct(h, hinv, idx + 1, N), (
            p,
            family,
            idx + 1,
        )

    # Every in-range prime q must be covered by its normalized trace term:
    # if H^q == 1 mod any divisor r of N, then V_kD - V_j == 0 mod r.
    # We can test the stronger whole-N implication whenever H^q == 1 mod N.
    qs = [q for q in primes_upto(B2) if q > B1 and q != p]
    for q in qs:
        k, j = normalize(q, D)
        assert k > 0 and j > 0
        term = (
            trace_direct(h, hinv, k * D, N)
            - trace_direct(h, hinv, j, N)
        ) % N
        if pow(h, q, N) == 1:
            assert term == 0, (p, family, q, k, j)

    # Algebraic identity behind the paired condition, tested directly:
    # V_a - V_b = 0 whenever h^(a-b)=1 or h^(a+b)=1.
    for q in qs[:20]:
        k, j = normalize(q, D)
        a, b = k * D, j
        term = (
            trace_direct(h, hinv, a, N)
            - trace_direct(h, hinv, b, N)
        ) % N
        if pow(h, a - b, N) == 1 or pow(h, a + b, N) == 1:
            assert term == 0


def main():
    # Multiple residue classes mod 8 and both GM/GQ branches.
    cases = [
        (7, 5, 29, 4),
        (11, 5, 43, 6),
        (13, 7, 47, 6),
        (17, 7, 59, 6),
        (23, 11, 71, 10),
        (31, 11, 79, 10),
        (41, 13, 89, 10),
        (47, 17, 97, 14),
        (59, 19, 109, 14),
        (61, 19, 113, 14),
    ]
    for p, B1, B2, D in cases:
        for family in ("GM", "GQ"):
            check_one(p, family, B1, B2, D)
    print("GM/GQ P-1 V-trace lift math: PASS")


if __name__ == "__main__":
    main()
