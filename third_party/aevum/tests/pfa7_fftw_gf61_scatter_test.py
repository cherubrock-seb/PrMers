#!/usr/bin/env python3
"""Prove the pfa7 512x7x512 GF61 fftW LDS scatter is a bijective reorder."""

R = 7
WIDTH = 512
SMALL_HEIGHT = 512
BIG_HEIGHT = SMALL_HEIGHT * R
BINARY = 524288
L_INV = 4
STEP = 3211264
NWORDS = WIDTH * SMALL_HEIGHT * R * 2
G_W = 64
NW = 8

def logical(row, binary_index):
    delta = (row + R - binary_index % R) % R
    t = (delta * L_INV) % R
    return binary_index + BINARY * t

def canonical_pair(n):
    pair = n >> 1
    x = pair // BIG_HEIGHT
    line = pair - x * BIG_HEIGHT
    return line * WIDTH + x

for g in range(R * SMALL_HEIGHT):
    row = g // SMALL_HEIGHT
    y = g - row * SMALL_HEIGHT
    even = []
    odd = []

    for me in range(G_W):
        first_binary_pair = me * SMALL_HEIGHT + y
        ne = logical(row, first_binary_pair * 2)
        no = logical(row, first_binary_pair * 2 + 1)
        for _ in range(NW):
            even.append(canonical_pair(ne))
            odd.append(canonical_pair(no))
            ne = (ne + STEP) % NWORDS
            no = (no + STEP) % NWORDS

    even_base = canonical_pair(logical(row, y * 2)) & ~(WIDTH - 1)
    odd_base = canonical_pair(logical(row, y * 2 + 1)) & ~(WIDTH - 1)

    assert len(even) == WIDTH and len(odd) == WIDTH
    assert {p & (WIDTH - 1) for p in even} == set(range(WIDTH))
    assert {p & (WIDTH - 1) for p in odd} == set(range(WIDTH))
    assert {p & ~(WIDTH - 1) for p in even} == {even_base}
    assert {p & ~(WIDTH - 1) for p in odd} == {odd_base}

print("ok: all 3584 pfa7 fftW workgroups map to two complete canonical 512-scalar rows")
