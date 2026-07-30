#!/usr/bin/env python3
"""Exact source/math audit for the GM/GQ shared trial-factoring path."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
kernel = (root / "kernels/gm_trial_factor.cl").read_text()
host = (root / "src/modes/RunGaussianTrialFactor.cpp").read_text()
main = (root / "src/main.cpp").read_text()

assert "gm_trial_factor" in kernel
assert "mont_mul_u64" in kernel
assert "mul_hi" in kernel
assert "gm_residue" in kernel and "gq_residue" in kernel
assert "One exponentiation" in kernel
assert "tryRunGaussianTrialFactor" in main
assert "hasExplicitNonTfWork" in host
assert "Unable to replace TF checkpoint" in host
assert "GMTF format is" in host
assert '"mode\\\": \\"gm-tf\\\"' in host
assert '"family\\\": \\"gaussian-pair\\\"' in host


def gm_gq(p: int) -> tuple[int, int]:
    middle = (p + 1) // 2
    epsilon = 1 if p % 8 in (1, 7) else -1
    gm = (1 << p) - epsilon * (1 << middle) + 1
    gq = ((1 << p) + epsilon * (1 << middle) + 1) // 5
    return gm, gq

# Known small examples prove that the complementary factor is GQ and that
# the same t=2^((p+1)/2) classifies both residues.
for p, gm_factor, gq_factor in ((7, 113, 29), (13, 53, 1613), (19, 525313, 229)):
    gm, gq = gm_gq(p)
    assert gm % gm_factor == 0
    assert gq % gq_factor == 0
    for q, expected in ((gm_factor, "GM"), (gq_factor, "GQ")):
        t = pow(2, (p + 1) // 2, q)
        a = (t * t * pow(2, -1, q)) % q
        epsilon = 1 if p % 8 in (1, 7) else -1
        gm_residue = (a - epsilon * t + 1) % q
        gq_residue = (a + epsilon * t + 1) % q
        assert (gm_residue == 0) == (expected == "GM")
        assert (gq_residue == 0) == (expected == "GQ")
        assert q == 1 or q == 5 or (q - 1) % (4 * p) == 0


MASK64 = (1 << 64) - 1

def mont_nprime(n: int) -> int:
    inverse = 1
    for _ in range(6):
        inverse = (inverse * (2 - ((n * inverse) & MASK64))) & MASK64
    return (-inverse) & MASK64

def mont_mul(a: int, b: int, n: int, nprime: int) -> int:
    product = a * b
    lo, hi = product & MASK64, product >> 64
    m = (lo * nprime) & MASK64
    mn = m * n
    mn_lo, mn_hi = mn & MASK64, mn >> 64
    carry = ((lo + mn_lo) >> 64) & 1
    true_value = hi + mn_hi + carry
    value = true_value & MASK64
    if true_value >> 64:
        value = (value + ((-n) & MASK64)) & MASK64
    elif value >= n:
        value -= n
    return value

for modulus in (13, 29, (1 << 63) - 25, (1 << 64) - 59):
    nprime = mont_nprime(modulus)
    rinverse = pow(1 << 64, -1, modulus)
    for a in (0, 1, 2, modulus // 2, modulus - 1):
        for b in (0, 1, 2, modulus // 2, modulus - 1):
            assert mont_mul(a, b, modulus, nprime) == (a * b * rinverse) % modulus

print("Gaussian pair shared GM/GQ GPU TF mathematics passed")
