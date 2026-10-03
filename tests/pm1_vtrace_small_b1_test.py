#!/usr/bin/env python3
"""V-trace stage 2 must be able to cover primes q | D with q > B1 (small B1)."""
from math import gcd
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
src = (root / 'src/modes/RunPM1.cpp').read_text()

# Source: the baby table and the planner's baby count share one predicate.
assert 'auto vtrace_needs_baby_j = [&](uint64_t j, uint64_t d)->bool{' in src
assert 'if (vtrace_needs_baby_j(j, d)) ++c;' in src
assert 'if (vtrace_needs_baby_j(j, D)) {' in src
assert not re.search(r'for \(uint64_t j = 1; j <= D / 2; j \+= 2\) \{\s*if \(gcd_u64\(j, D\) == 1\)', src)


def is_prime(n):
    return n > 1 and all(n % p for p in range(2, int(n ** 0.5) + 1))


def babies(D, b1):
    return {j for j in range(1, D // 2 + 1, 2)
            if gcd(j, D) == 1 or (j > b1 and D % j == 0 and is_prime(j))}


# Model: every odd prime q in (B1, B2] is kD +/- j for a stored j.
for D in (30, 210, 2310):
    for b1 in range(1, 12):
        B2 = 3000
        have = babies(D, b1)
        for q in range(b1 + 1, B2 + 1):
            if q % 2 == 0 or not is_prime(q):
                continue
            k = q // D
            j = q - k * D
            if j > D // 2:
                j = D - j
            assert j in have, (D, b1, q, j)
print('PrMers V-trace small-B1 baby coverage test passed')
