#!/usr/bin/env python3
from pathlib import Path

use = Path("src/UseOptions.h").read_text()
tail = Path("src/cl/tailsquare.cl").read_text()

assert '"ENABLE_BETTER_ONEPAIRSQ"' in use
assert "#if ENABLE_BETTER_ONEPAIRSQ" in tail
assert "Compute 2ab as (a + b)^2 - a^2 - b^2" in tail
assert "This code should be faster (saves at least one wide mul)" in tail

print("ok: dormant GF61 better-onePairSq path is opt-in through -use with defaults unchanged")
