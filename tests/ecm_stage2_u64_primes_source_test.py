#!/usr/bin/env python3
"""ECM stage-2 primes are 64-bit values.

`unsigned long` is only 32 bits on Windows (LLP64), so passing a stage-2 prime
to mpz_mul_ui() truncates it once it exceeds 2^32.  The ECM sources must go
through ecm_mpz_mul_u64() instead.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bad = []
present = []
for name in ('src/modes/RunEcm.cpp', 'src/modes/RunEcmTwistedEdwards.cpp'):
    text = (root / name).read_text()
    for no, line in enumerate(text.splitlines(), 1):
        code = line.split('//')[0]
        if re.search(r'mpz_mul_ui\([^;]*primesS2_v', code):
            bad.append(f'{name}:{no}: {line.strip()}')
    present.append('ecm_mpz_mul_u64(Echunk, primesS2_v[' in text)
assert not bad, '\n'.join(bad)
assert all(present)
print('PrMers ECM 64-bit stage-2 prime test passed')
