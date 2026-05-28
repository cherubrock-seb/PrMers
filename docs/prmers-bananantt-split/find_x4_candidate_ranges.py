#!/usr/bin/env python3
import argparse
from typing import Iterable, List, Optional, Tuple

KNOWN_MERSENNE_PRIME_EXPONENTS = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127,
    521, 607, 1279, 2203, 2281, 3217, 4253, 4423,
    9689, 9941, 11213, 19937, 21701, 23209, 44497,
    86243, 110503, 132049, 216091, 756839, 859433,
    1257787, 1398269, 2976221, 3021377, 6972593,
    13466917, 20996011, 24036583, 25964951, 30402457,
    32582657, 37156667, 42643801, 43112609, 57885161,
    74207281, 77232917, 82589933, 136279841,
]

def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small = [2,3,5,7,11,13,17,19,23,29,31,37]
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def first_primes(lo: int, hi: int, count: int = 6) -> List[int]:
    out: List[int] = []
    x = max(2, lo)
    if x <= 2 <= hi:
        out.append(2)
    if x % 2 == 0:
        x += 1
    while x <= hi and len(out) < count:
        if is_probable_prime(x):
            out.append(x)
        x += 2
    return out

def maxp_for_ln(odd: int, ln: int, power: int, cap: int = 92, head: int = 2) -> Optional[int]:
    n = odd * (1 << ln)
    L = (n - 1).bit_length()
    maxw = (cap - 1 - ((power - 1) * L + head)) // power
    if maxw < 1:
        return None
    return maxw * n

def layout_ln(p: int, odd: int, power: int, cap: int = 92, head: int = 2) -> Optional[int]:
    for ln in range(2, 31):
        n = odd * (1 << ln)
        if n > p:
            continue
        mp = maxp_for_ln(odd, ln, power, cap, head)
        if mp is not None and p <= mp:
            return ln
    return None

def intervals(odd: int, pmax: int, cap: int = 92, head: int = 2):
    bps = {1, pmax + 1}
    for power in (2, 4):
        for ln in range(2, 31):
            n = odd * (1 << ln)
            mp = maxp_for_ln(odd, ln, power, cap, head)
            if mp:
                if 1 <= n <= pmax + 1:
                    bps.add(n)
                if 1 <= mp + 1 <= pmax + 1:
                    bps.add(mp + 1)
    bps = sorted(bps)
    out = []
    for a, b in zip(bps, bps[1:]):
        hi = b - 1
        l2 = layout_ln(a, odd, 2, cap, head)
        l4 = layout_ln(a, odd, 4, cap, head)
        if l2 is None or l4 is None:
            continue
        growth = 1 << (l4 - l2)
        ideal_ratio = 2.0 / (growth * (l4 / max(l2, 1)))
        out.append((a, hi, l2, l4, growth, ideal_ratio))
    merged = []
    for it in out:
        if merged and merged[-1][1] + 1 == it[0] and merged[-1][2:5] == it[2:5]:
            merged[-1] = (merged[-1][0], it[1], *it[2:])
        else:
            merged.append(it)
    return merged

def print_intervals(args) -> None:
    print(f'odd={args.odd} cap={args.capacity}: x4 is most plausible when N_x4/N_square <= {args.max_growth}.')
    print('lo..hi                 ln2 ln4 growth ideal_ratio first prime exponents')
    for lo, hi, l2, l4, g, ratio in intervals(args.odd, args.pmax, args.capacity, args.headroom):
        if g <= args.max_growth:
            ex = first_primes(lo, hi, args.prime_examples)
            print(f'{lo:>9}..{hi:<9}  {l2:>3} {l4:>3} {g:>6} {ratio:>10.3f} {ex}')

def print_known_mersenne(args) -> None:
    print(f'Known Mersenne-prime exponents, odd={args.odd}, cap={args.capacity}')
    print('p            ln2 N_square       ln4 N_x4          growth ideal_ratio expected_x4_half_check')
    for p in KNOWN_MERSENNE_PRIME_EXPONENTS:
        if p < args.min_p or p > args.pmax:
            continue
        l2 = layout_ln(p, args.odd, 2, args.capacity, args.headroom)
        l4 = layout_ln(p, args.odd, 4, args.capacity, args.headroom)
        if l2 is None or l4 is None:
            continue
        n2 = args.odd * (1 << l2)
        n4 = args.odd * (1 << l4)
        g = n4 // n2
        ratio = 2.0 / (g * (l4 / max(l2, 1)))
        expected = '0x51' if (p & 1) else '0x09'
        print(f'{p:>10}  {l2:>3} {n2:>12}  {l4:>3} {n4:>12}  {g:>6} {ratio:>10.3f} {expected}')

def main() -> None:
    ap = argparse.ArgumentParser(description='Theoretical x^4 candidate ranges for BananaNTT CRT layouts.')
    ap.add_argument('--odd', type=int, default=9, choices=[1,3,9])
    ap.add_argument('--pmax', type=int, default=2_000_000)
    ap.add_argument('--min-p', type=int, default=1)
    ap.add_argument('--capacity', type=int, default=92)
    ap.add_argument('--headroom', type=int, default=2)
    ap.add_argument('--max-growth', type=int, default=2, help='show intervals where N_x4 / N_square <= this')
    ap.add_argument('--prime-examples', type=int, default=6)
    ap.add_argument('--known-mersenne-primes', action='store_true', help='print known M_p prime exponents instead of generic intervals')
    args = ap.parse_args()
    if args.known_mersenne_primes:
        print_known_mersenne(args)
    else:
        print_intervals(args)

if __name__ == '__main__':
    main()
