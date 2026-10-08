#!/usr/bin/env python3
"""BSGS Stage 2 must cover every prime in (B1, B2] or refuse the parameters.

The giant steps reach a prime q as k*D +- delta with k >= 1 and gcd(delta, D) = 1.
A prime q <= D/2 has k == 0 (or shares a factor with D) and was skipped silently,
while Stage 2 was still reported as complete. The plan builder in
RunGaussianMersenneEcmOptimized.cpp therefore picks the largest D whose half is
below the first Stage 2 prime and rejects parameters for which none exists.

This test re-implements the plan (as the other GM ECM regression tests do the
curve arithmetic) and checks it against the C++ source.
"""
from math import gcd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "src/modes/RunGaussianMersenneEcmOptimized.cpp").read_text()

# Source must contain the coverage guard and use it for explicit and automatic D.
for token in (
    "stage2_d_covers_primes",
    "build_stage2_plan",
    "PRMERS_GM_ECM_BSGS_D",
    "{210ULL, 30ULL, 6ULL, 4ULL}",
):
    assert token in src, token
assert src.count("stage2_d_covers_primes(") >= 4  # definition + explicit + automatic + fallback


def primes_in(low, high):
    out = []
    for n in range(max(2, low + 1), high + 1):
        if all(n % d for d in range(2, int(n ** 0.5) + 1)):
            out.append(n)
    return out


def plan(D, primes):
    """Python port of build_stage2_plan: the primes the giant steps reach."""
    reached = set()
    for q in primes:
        k = (q + D // 2) // D
        if k == 0:
            continue
        kd = k * D
        delta = abs(kd - q)
        if delta == 0 or delta > D // 2 or gcd(delta, D) != 1:
            continue
        reached.add(q)
    return reached


def covers(D, primes):
    return not primes or primes[0] > D // 2


def choose_D(B1, B2):
    """Python port of make_stage2_plan's automatic D selection (None = error)."""
    primes = primes_in(B1, B2)
    D = 30 if (B1 < 100 or B2 <= 10000) else 210
    for cand in (210, 30, 6, 4):
        if cand <= D and covers(cand, primes):
            return cand
    return None


# The old behaviour: with the fixed default D some primes are skipped.
old = plan(210, primes_in(100, 20000))
assert 101 not in old and 103 not in old, "expected the old plan to drop 101 and 103"
assert 3 not in plan(30, primes_in(2, 3))

# The new behaviour: for every B1 >= 2 the chosen D reaches every prime.
checked = 0
for B1 in list(range(2, 140)) + [200, 211, 1000]:
    for B2 in sorted({B1 + 1, B1 + 2, B1 + 10, B1 * 3, 2000}):
        if B2 <= B1:
            continue
        primes = primes_in(B1, B2)
        D = choose_D(B1, B2)
        assert D is not None, (B1, B2)
        assert plan(D, primes) == set(primes), (B1, B2, D, set(primes) - plan(D, primes))
        checked += 1
assert checked > 500

# B1 < 2 cannot be covered by any D >= 4 and must be rejected, not skipped.
assert choose_D(1, 10) is None
assert choose_D(0, 10) is None

print("gaussian mersenne ecm bsgs small primes test passed")
