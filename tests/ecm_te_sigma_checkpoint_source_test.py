#!/usr/bin/env python3
"""Twisted Edwards ECM checkpoints must identify the -sigma curve, never resume a
Montgomery file, validate a loaded state before trusting it, and bound the retries."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / 'src/modes/RunEcmTwistedEdwards.cpp').read_text()

# sigma identity is derived once and written to both checkpoint kinds
assert 'const uint64_t sigma_id' in src
assert 'constexpr int kTeCkptS1Sigma = 2;' in src
assert 'constexpr int kTeCkptS2Sigma = 7;' in src
assert 'int version = kTeCkptS1Sigma;' in src
assert 'int version = kTeCkptS2Sigma;' in src
assert src.count('reinterpret_cast<const char*>(&sigma_id)') == 2

# readers (and the resume probes) compare it; versions that carry none are refused under -sigma
assert src.count('saved_sigma_id != sigma_id') >= 4
assert src.count('does not record which sigma') + src.count('do not record which sigma') >= 4

# Montgomery files (ecm_m_ / ecm2_m_) are not loaded by the Twisted Edwards driver
assert 'ckpt_legacy' not in src and 'ckpt2_legacy' not in src
assert not re.search(r'"ecm2?_m_"', src)

# a resumed stage 1 state must pass the curve invariant, otherwise it is discarded
body = src[src.index('int rr = read_ckpt(start_i'):]
body = body[:body.index('auto t0 = high_resolution_clock::now();')]
assert 'check_invariant()' in body
assert 'fresh_state' in body and 'fs::remove(ckpt_file' in body

# rollback retries are bounded
assert 'kMaxInvariantRetries' in src and 'invariant_fail_streak' in src
assert 'invariant_fail_streak = 0;' in src
print('PrMers ECM TE sigma checkpoint test passed')
