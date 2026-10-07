#!/usr/bin/env python3
"""Marin/Aevum PRP checkpoints must carry the iteration of their verified R4/R5 state, and a
resume must not promote the unverified R0/R1 to verified."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / 'src/modes/RunPrpOrLlMarin.cpp').read_text()

# version 4 is written whenever Gerbicz-Li checking is on and carries block + goodIter, except while
# the restore point is still the never-verified state of an older file
save = src[src.index('auto save_ckpt = [&]'):src.index('const size_t R0 = 0')]
assert 'const bool record_good = gl_active && !restore_point_unverified;' in save
assert 'int version = record_good ? 4 : (ckpt_block ? 3 : 2);' in save
assert save.count('int version') == 1
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

# the flag save_ckpt consults is declared once, before the lambda, and is set from the legacy resume
# before the first save; only a passed full check clears it
assert src.count('bool restore_point_unverified') == 1
decl = src.index('bool restore_point_unverified = false;')
assert decl < src.index('auto save_ckpt = [&]')
set_legacy = src.index('restore_point_unverified = resumed_unverified_legacy;')
assert set_legacy > src.index('const bool resumed_unverified_legacy')
assert set_legacy < src.index('save_ckpt(', src.index('auto save_ckpt = [&]') + 1)
import re
assigns = re.findall(r'restore_point_unverified\s*=(?!=)\s*([^;]+);', src)
assert sorted(assigns) == sorted(['false', 'resumed_unverified_legacy', 'false']), assigns
passed = src[src.index('[Gerbicz Li] Check passed!'):]
passed = passed[:passed.index('//cl_event postEvt;')]
assert 'restore_point_unverified = false;' in passed and 'goodIter = iter + 1;' in passed
print('PrMers Marin PRP checkpoint verified-state test passed')
