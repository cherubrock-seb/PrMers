#!/usr/bin/env python3
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root / 'src/modes/RunPM1.cpp').read_text()
assert 'Aevum uses the generic square plus base-3 multiply path; fast3 is disabled.' in s
assert 'Gerbicz-Li disabled because its accumulator requires generic multiplication' not in s
assert 'bool useFast3 = useFast3Candidate && (nextStart == 0) && !aevum_backend;' in s
assert 'eng->square_mul(RSTATE); if (b) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RSTATE, RTMP); }' in s
assert 'Aevum P-1 Stage 1 would require generic multiplication across multiple chunks' not in s
print('PrMers Aevum P-1 GitHub prepared-base policy test passed')
