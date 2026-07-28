#!/usr/bin/env python3
"""Generate native PrMers Gaussian-Mersenne worktodo entries."""
from __future__ import annotations

import argparse
from pathlib import Path


def is_prime_64(n: int) -> bool:
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == q:
            return True
        if n % q == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_exponents(start: int, count: int):
    n = max(3, start + 1)
    if n % 2 == 0:
        n += 1
    found = 0
    while found < count:
        if is_prime_64(n):
            yield n
            found += 1
        n += 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="strict lower bound")
    ap.add_argument("--count", type=int, required=True, help="number of prime exponents")
    ap.add_argument("--output", type=Path, default=Path("worktodo-gm.txt"))
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--mode", choices=("chain", "pm1", "ecm", "proth", "prp"), default="chain")
    ap.add_argument("--pm1-b1", type=int, default=100_000)
    ap.add_argument("--pm1-b2", type=int, default=1_000_000)
    ap.add_argument("--base", type=int, default=3)
    ap.add_argument("--ecm-b1", type=int, default=0)
    ap.add_argument("--ecm-b2", type=int, default=0)
    ap.add_argument("--curves", type=int, default=0)
    ap.add_argument("--sigma", type=int, default=0)
    ap.add_argument("--sieve", type=int, default=1_000_000_000_000)
    ap.add_argument("--chunk-bits", type=int, default=262_144)
    args = ap.parse_args()

    if args.count <= 0:
        ap.error("--count must be positive")
    for name in ("pm1_b1", "pm1_b2", "ecm_b1", "ecm_b2", "curves", "sieve", "chunk_bits"):
        if getattr(args, name) < 0:
            ap.error(f"--{name.replace('_', '-')} must be nonnegative")

    lines: list[str] = []
    for p in prime_exponents(args.start, args.count):
        if args.mode == "chain":
            lines.append(
                f"GMCHAIN={p},{args.pm1_b1},{args.pm1_b2},"
                f"{args.ecm_b1},{args.ecm_b2},{args.curves},{args.sieve},{args.chunk_bits}"
            )
        elif args.mode == "pm1":
            lines.append(
                f"GMPMINUS1={p},{args.pm1_b1},{args.pm1_b2},"
                f"{args.base},{args.sieve},{args.chunk_bits}"
            )
        elif args.mode == "ecm":
            lines.append(
                f"GMECM={p},{args.ecm_b1},{args.ecm_b2},{max(1, args.curves)},"
                f"{args.sigma},{args.sieve},{args.chunk_bits}"
            )
        elif args.mode == "proth":
            lines.append(f"GMPROTH={p},{args.sieve}")
        else:
            lines.append(f"GMPRP={p},{args.sieve}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as out:
        if not args.append:
            out.write("# Native PrMers Gaussian-Mersenne worktodo generated automatically\n")
            out.write(f"# start>{args.start}; prime exponents={args.count}; mode={args.mode}\n")
        out.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} entries to {args.output}")
    if lines:
        print(f"First: {lines[0]}")
        print(f"Last : {lines[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
