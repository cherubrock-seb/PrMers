#!/usr/bin/env python3
from pathlib import Path
from math import gcd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).name.startswith('gaussian_') else Path('.')

# Source/isolation guards: v99.96 code must remain compiled as a legacy symbol,
# while the v99.97 wrapper owns the public method.
app_h = (ROOT / 'include/core/App.hpp').read_text()
make = (ROOT / 'Makefile').read_text()
fast = (ROOT / 'src/modes/RunGaussianMersenneEcmFast.cpp').read_text()
rename = (ROOT / 'include/core/GmEcmLegacyRename.hpp').read_text()

assert 'int runGaussianMersenneECM();' in app_h
assert 'int runGaussianMersenneECMLegacy();' in app_h
assert 'GmEcmLegacyRename.hpp' in make
assert '#define runGaussianMersenneECM runGaussianMersenneECMLegacy' in rename
assert 'if (!options.edwards) return runGaussianMersenneECMLegacy();' in fast
assert 'twisted Edwards + signed NAF' in fast
assert 'exact legacy Suyama sigma sequence' in fast

# Exact v99.96 seed stream regression for the real factor discovered after
# seven curves at p=21403643, B1=2000.
def splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask

seed = 84_782_075_184
sigmas = [6 + splitmix64(seed + c) % 0x7FFFFFFFFFFFFFF0 for c in range(7)]
assert sigmas[-1] == 3_059_155_915_320_676_093, sigmas[-1]

p = 21_403_643
sigma = sigmas[-1]
factors = [3_253_353_737, 148_455_667_849]
bundle = factors[0] * factors[1]
assert bundle == 482_978_801_775_374_901_713

# Both factors really divide the selected Gaussian-Mersenne norm.  Work only
# modulo q; constructing the 21M-bit norm is unnecessary.
r = p & 7
chi = 1 if r in (1, 7) else -1
m = (p + 1) // 2
for q in factors:
    target_mod_q = (pow(2, p, q) - chi * pow(2, m, q) + 1) % q
    assert target_mod_q == 0, (q, target_mod_q)

# Build K=lcm(1..2000) as prime powers.
def primes_upto(n: int):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[:2] = b'\x00\x00'
    for x in range(2, int(n**0.5) + 1):
        if sieve[x]:
            sieve[x*x:n+1:x] = b'\x00' * (((n - x*x) // x) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]

K = 1
for ell in primes_upto(2000):
    power = ell
    while power <= 2000 // ell:
        power *= ell
    K *= power
assert K.bit_length() == 2878

# Verify the Suyama -> twisted-Edwards conversion and [K]P=O modulo each of
# the two real factors.  This proves the NAF engine is mathematically expected
# to return the same bundle as the legacy Montgomery ladder.
def inv(a, q):
    return pow(a % q, -1, q)

def setup_te(q):
    s = sigma % q
    if s < 6:
        s += 6
    u = (s*s - 5) % q
    v = (4*s) % q
    u2, v2 = u*u % q, v*v % q
    u3, v3 = u2*u % q, v2*v % q
    vu = (v-u) % q
    aplus2 = vu*vu % q * vu % q * (3*u+v) % q * inv(4*u3*v, q) % q
    Bm = u * inv(v, q) % q
    a = aplus2 * inv(Bm, q) % q
    d = (aplus2 - 4) * inv(Bm, q) % q
    x = u2 * v % q * inv((u-v)*(u+v)*(s*s+5), q) % q
    y = (u3-v3) % q * inv(u3+v3, q) % q
    assert (a*x*x + y*y - 1 - d*x*x*y*y) % q == 0
    return a, d, (x, y)

def add(P, Q, a, d, q):
    x1, y1 = P
    x2, y2 = Q
    t = x1*x2 % q * y1 % q * y2 % q
    x3 = (x1*y2 + y1*x2) % q * inv(1 + d*t, q) % q
    y3 = (y1*y2 - a*x1*x2) % q * inv(1 - d*t, q) % q
    return x3, y3

def mul(k, P, a, d, q):
    R = (0, 1)
    Q = P
    while k:
        if k & 1:
            R = add(R, Q, a, d, q)
        Q = add(Q, Q, a, d, q)
        k >>= 1
    return R

for q in factors:
    a, d, P = setup_te(q)
    assert mul(K, P, a, d, q) == (0, 1), q

print('Gaussian ECM v99.97 NAF regression: OK')
