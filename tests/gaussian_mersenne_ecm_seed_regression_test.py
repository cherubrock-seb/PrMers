#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]

src = (
    root / "src/modes/RunGaussianMersenneFactor.cpp"
).read_text()

cli = (
    root / "src/io/CliParser.cpp"
).read_text()

assert (
    "options.curve_seed != 0 ? options.curve_seed"
    in src
)

assert (
    "opts.curve_seed = std::strtoull"
    in cli
)

assert (
    "options.seed != 0 ? options.seed"
    not in src
)

MASK = (1 << 64) - 1
MOD = 0x7ffffffffffffff0

def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK
    x = ((x ^ (x >> 30)) *
         0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) *
         0x94d049bb133111eb) & MASK
    return (x ^ (x >> 31)) & MASK

seed = 82620261847
sigma = 6 + splitmix64(seed) % MOD

assert sigma == 943228762148138654

print("GM-ECM seed regression test passed")
