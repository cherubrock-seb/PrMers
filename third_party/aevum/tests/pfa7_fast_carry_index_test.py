#!/usr/bin/env python3
"""Prove the PFA7 fast lazy-carry gather index equals the stock permutation."""

WIDTH = 512
SMALL_HEIGHT = 512
PFA_RADIX = 7
BIG_HEIGHT = SMALL_HEIGHT * PFA_RADIX
CARRY_LEN = 8
NW = 8
G_W = WIDTH // NW
HB = BIG_HEIGHT // CARRY_LEN

def stock(x, gy):
    gx = x // G_W
    me = x - gx * G_W
    prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH)
    return WIDTH * (prev % HB) + prev // HB

def fast(x, gy):
    prev_x = x - 1 if x else WIDTH - 1
    return WIDTH * (gy - 1) + x if gy else WIDTH * (HB - 1) + prev_x

for x in range(WIDTH):
    for gy in range(HB):
        a = stock(x, gy)
        b = fast(x, gy)
        assert a == b, (x, gy, a, b)

print(f"ok: {WIDTH*HB} PFA7 lazy-carry predecessor indices are exactly equivalent")
