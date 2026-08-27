#!/usr/bin/env python3
from pathlib import Path
from math import gcd

root = Path(__file__).resolve().parents[1]
cli = (root / "src/io/CliParser.cpp").read_text()
app = (root / "src/core/App.cpp").read_text()
fast = (root / "src/modes/RunGaussianMersenneEcmFast.cpp").read_text()
opt = (root / "src/modes/RunGaussianMersenneEcmOptimized.cpp").read_text()

# Isolation/routing guards.
assert '-gm-ecm-special32' in cli
assert 'opts.mode = "gm-ecm-special32"' in cli
assert 'opts.mode != "gm-pm1" && opts.mode != "gm-ecm" && opts.mode != "gm-ecm-special32"' in cli
assert 'options.mode == "gm-ecm-special32"' in app
assert 'options.mode == "gm-ecm-special32"' in fast
assert 'make_special32_opt' in opt
assert 't.special32 = special32' in opt
assert 'deterministic Special32 profiles A/B/C' in opt
assert '? 3ULL' in opt
assert 'make_suyama_opt(t.n, sigma)' in opt  # ordinary Suyama path still exists

def gm_norm(p):
    chi = 1 if p % 8 in (1, 7) else -1
    return (1 << p) - chi * (1 << ((p + 1) // 2)) + 1

def oriented_i(p, n):
    I = (1 << p) % n
    if ((p + 1) // 2) & 1:
        I = (-I) % n
    return I

def special32_setups(p, n):
    I = oriented_i(p, n)
    inv4 = pow(4, -1, n)
    inv8 = pow(8, -1, n)
    inv16 = pow(16, -1, n)
    return (
        ((-1 + I) % n, ((7 + I) * inv8) % n),          # A
        ((-4) % n, ((7 + I) * inv8) % n),              # B
        (((1 - 7 * I) * inv4) % n, ((-1 + 7 * I) * inv16) % n),  # C
    )

def xdbl(P, a24, n):
    X, Z = P
    A = (X + Z) % n
    B = (X - Z) % n
    AA = A * A % n
    BB = B * B % n
    E = (AA - BB) % n
    return AA * BB % n, E * (BB + a24 * E) % n

def xadd(P, Q, D, n):
    X1, Z1 = P
    X2, Z2 = Q
    Xd, Zd = D
    A = (X1 + Z1) % n
    B = (X1 - Z1) % n
    C = (X2 + Z2) % n
    Dm = (X2 - Z2) % n
    DA = Dm * A % n
    CB = C * B % n
    s = (DA + CB) % n
    d = (DA - CB) % n
    return Zd * s * s % n, Xd * d * d % n

def xmul(P, k, a24, n):
    if k == 0:
        return (1, 0)
    if k == 1:
        return P
    R0 = P
    R1 = xdbl(P, a24, n)
    for bit in bin(k)[3:]:
        if bit == "0":
            R1 = xadd(R0, R1, P, n)
            R0 = xdbl(R0, a24, n)
        else:
            R0 = xadd(R0, R1, P, n)
            R1 = xdbl(R1, a24, n)
    return R0

def primes(limit):
    out = []
    for x in range(2, limit + 1):
        if all(x % d for d in range(2, int(x ** 0.5) + 1)):
            out.append(x)
    return out

def stage1_exp(B1):
    E = 1
    for q in primes(B1):
        pp = q
        while pp * q <= B1:
            pp *= q
        E *= pp
    return E

# Core GM identities used by the implementation.
for p in (13, 17, 23, 31, 37, 41, 43):
    n = gm_norm(p)
    I = oriented_i(p, n)
    assert (I * I + 1) % n == 0
    assert pow((1 + I) % n, p, n) == 1
    assert len(special32_setups(p, n)) == 3

# End-to-end CPU emulation of the same Montgomery x/z Stage 1 used by PrMers.
# For each historical small composite, at least one of A/B/C must recover the
# known factor at the listed bound.
golden = (
    (13, 5, 53),
    (17, 5, 137),
    (23, 5, 277),
    (31, 50, 5581),
    (37, 5, 593),
    (41, 100, 181549),
    (43, 5, 173),
)

for p, B1, wanted in golden:
    n = gm_norm(p)
    E = stage1_exp(B1)
    got = []
    for x, a24 in special32_setups(p, n):
        _, Z = xmul((x, 1), E, a24, n)
        g = gcd(Z, n)
        if 1 < g < n:
            got.append(g)
    assert wanted in got, (p, B1, wanted, got)

print("Gaussian ECM Special32 regression: OK")
print("profiles=A/B/C; golden Stage1 cases:", len(golden))
