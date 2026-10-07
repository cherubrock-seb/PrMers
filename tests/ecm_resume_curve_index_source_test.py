#!/usr/bin/env python3
"""The Montgomery ECM curve loop must start at the resumed curve index.

Checkpoints of finished curves are deleted, so the first curve that still has
one is the curve that was running when the previous run stopped.  Looping from
curve 0 reruns every earlier (completed) curve.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / 'src/modes/RunEcm.cpp').read_text()

assert re.search(r'start_curve\s*=\s*\(!options\.seed && have_resume_seed\)\s*\?\s*resume_curve_idx\s*:\s*0', src), \
    'start_curve is not derived from resume_curve_idx'
assert 'for (uint64_t c = start_curve; c < curves; ++c)' in src, 'curve loop does not start at start_curve'
assert 'for (uint64_t c = 0; c < curves; ++c)\n    {\n        result_factor = 0;' not in src, \
    'main curve loop still starts at curve 0'
print('PrMers ECM resume curve index test passed')
