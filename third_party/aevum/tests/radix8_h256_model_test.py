"""Index/twiddle model for 256=4*8*8; does not execute a GPU kernel."""
import random

P = 257
W = 3
assert pow(W, 128, P) != 1 and pow(W, 256, P) == 1

def dft(a, w):
    return [sum(x * pow(w, j*k, P) for j, x in enumerate(a)) % P for k in range(len(a))]

def shuffle(u, first):
    slots = [None]*256
    for lane in range(32):
        for i in range(8):
            dst = i//2 + (i%2)*128 + lane*4 if first else i*4 + (lane & ~3)*8 + (lane & 3)
            assert slots[dst] is None
            slots[dst] = u[lane][i]
    assert all(v is not None for v in slots)
    return [[slots[i*32+lane] for i in range(8)] for lane in range(32)]

def candidate(a, w):
    u = [[a[i*32+lane] for i in range(8)] for lane in range(32)]
    for lane in range(32):
        even, odd = dft(u[lane][::2], pow(w, 64, P)), dft(u[lane][1::2], pow(w, 64, P))
        u[lane] = [z for pair in zip(even, odd) for z in pair]
        for i in range(2, 8):
            u[lane][i] = u[lane][i] * pow(w, (i//2)*(lane+(i%2)*32), P) % P
    u = shuffle(u, True)
    for lane in range(32):
        u[lane] = dft(u[lane], pow(w, 32, P))
        for i in range(1, 8):
            u[lane][i] = u[lane][i] * pow(w, i*(lane//4)*4, P) % P
    u = shuffle(u, False)
    u = [dft(row, pow(w, 32, P)) for row in u]
    return [u[lane][i] for i in range(8) for lane in range(32)]

rng = random.Random(256)
vectors = [[int(i == j) for i in range(256)] for j in (0, 1, 31, 32, 127, 255)]
vectors += [[rng.randrange(P) for _ in range(256)] for _ in range(4)]
for root in (W, pow(W, -1, P)):
    for a in vectors:
        assert candidate(a, root) == dft(a, root)
print("256-point index/twiddle model: 20 comparisons with direct DFT; GPU validation pending.")
