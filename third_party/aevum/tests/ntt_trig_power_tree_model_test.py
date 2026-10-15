"""Finite-field algebra for the NTT twiddle power tree; not a device test."""
import random

def mul(a, b, p):
    return ((a[0]*b[0] - a[1]*b[1]) % p,
            (a[0]*b[1] + a[1]*b[0]) % p)

def power(a, n, p):
    out = (1, 0)
    while n:
        if n & 1:
            out = mul(out, a, p)
        a = mul(a, a, p)
        n >>= 1
    return out

def square_unit(a, p):
    return ((1-2*a[1]*a[1]) % p, 2*a[0]*a[1] % p)

def cube_unit(sq, w, p):
    return ((w[0]-2*sq[1]*w[1]) % p,
            (2*sq[1]*w[0]-w[1]) % p)

def pair(a, b, p):
    xx, yy, xy, yx = a[0]*b[0], a[1]*b[1], a[0]*b[1], a[1]*b[0]
    return (((xx-yy) % p, (xy+yx) % p),
            ((xx+yy) % p, (yx-xy) % p))

checks = 0
for bits in (31, 61):
    p = (1 << bits)-1
    rng = random.Random(bits)
    parameters = [0, 1, 2, p-1, p//2] + [rng.randrange(p) for _ in range(63)]
    if bits == 31:
        assert 2*p*p < (1 << 63)
    for t in parameters:
        inv = pow(1+t*t, -1, p)
        w = ((1-t*t)*inv % p, 2*t*inv % p)
        assert (w[0]*w[0]+w[1]*w[1]) % p == 1
        # Include redundant zero and the allowed small GF61 weak representatives.
        limit = 0 if bits == 31 else 7
        w = tuple(x+p if x <= limit else x for x in w)
        w2 = square_unit(w, p)
        w3 = cube_unit(w2, w, p)
        w4, w6 = square_unit(w2, p), square_unit(w3, p)
        w7, w5 = pair(w6, w, p)
        tree = [(1, 0), tuple(x % p for x in w), w2, w3, w4, w5, w6, w7]
        for k, value in enumerate(tree):
            assert value == power(w, k, p)
            checks += 2
        data = [(rng.randrange(p+1), rng.randrange(p+1)) for _ in range(8)]
        for k in range(1, 8):
            assert mul(data[k], tree[k], p) == mul(data[k], power(w, k, p), p)
            checks += 2
        plus, minus = pair(data[0], w, p)
        assert plus == mul(data[0], w, p)
        assert minus == mul(data[0], power(w, p, p), p)
        checks += 4
print(f"{checks} scalar field comparisons; GPU validation pending")
