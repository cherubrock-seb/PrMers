#!/usr/bin/env python3
"""-cmont Stage 2: a curve skipped because Z == 0 (mod N) must not leak the engine
or leave its checkpoints behind (they pin every restart to that curve).

Source-level check: a run-time reproducer needs every prime factor of N to die in
the same Stage 2 chunk of a non-TE (-torsion8/-torsion16) curve, which tiny
exponents do not give.
"""
import re
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / 'src/modes/RunEcm.cpp').read_text()

def block_after(marker, text):
    """Brace-matched body of the first '{' after marker."""
    i = text.index(marker)
    j = text.index('{', i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j:k + 1]
    raise AssertionError('unbalanced block after ' + marker)

def check(body, what):
    assert 'fs::remove(ckpt_file' in body and 'fs::remove(ckpt2' in body, what + ': checkpoints are not removed'
    assert re.search(r'ckpt2 \+ "\.old"', body) and re.search(r'ckpt2 \+ "\.new"', body), what + ': .old/.new left behind'
    assert 'delete eng;' in body, what + ': engine is leaked'

# chunk-boundary path: rz < 0 right after the Stage 2 gcd check
check(block_after('if (rz < 0) {\n                        // Z == 0 (mod N)', src), 'chunk boundary Z == 0')
# the cleanup comes before the flag that sends the loop to the next curve
body = block_after('if (rz < 0) {\n                        // Z == 0 (mod N)', src)
assert body.index('delete eng;') < body.index('next_curve_after_stage2 = true;')

# Stage 2 base setup failure
check(block_after('if (setup_rc < 0) {\n                    std::error_code ec0;', src), 'Stage 2 setup failure')
print('PrMers ECM Stage 2 Z == 0 cleanup test passed')
