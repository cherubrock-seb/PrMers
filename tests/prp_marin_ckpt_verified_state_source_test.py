#!/usr/bin/env python3
"""Marin/Aevum PRP checkpoints must carry the iteration of their verified R4/R5 state, and a
resume must not promote the unverified R0/R1 to verified."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / 'src/modes/RunPrpOrLlMarin.cpp').read_text()

# version 4 is written whenever Gerbicz-Li checking is on and carries block + goodIter
assert 'int version = gl_active ? 4 : (ckpt_block ? 3 : 2);' in src
assert 'if (version > 4) return -2;' in src
assert 'const uint32_t good = static_cast<uint32_t>(goodIter);' in src
assert 'if (good > ri) return -2;' in src

# goodIter must be declared before the save/read lambdas that use it
assert src.index('uint64_t goodIter = 0;') < src.index('auto save_ckpt = [&]')
assert src.count('uint64_t goodIter') == 1

# the unconditional promotion of R0/R1 after loading is gone: it only runs without a recorded goodIter
resume = src[src.index('const bool resumed_verified_state'):src.index('eng->set(RBASE, 3);')]
assert 'if (resumed_verified_state) {' in resume
assert 'goodIter = ckpt_good;' in resume
else_branch = resume[resume.index('} else {'):]
assert 'eng->copy(R4, R0);' in else_branch and 'eng->copy(R5, R1);' in else_branch
assert 'goodIter = ri;' in else_branch
assert 'eng->copy(R4, R0);' not in resume[:resume.index('} else {')]
# no other place re-blesses the loaded state
assert src.count('eng->copy(R4, R0);') == 2  # resume fallback and a passed check

# older checkpoints get an early check, and a restore point that was never verified is reported
assert 'checkpass = checkpasslevel - 1;' in src
assert 'restore_point_unverified' in src
print('PrMers Marin PRP checkpoint verified-state test passed')
