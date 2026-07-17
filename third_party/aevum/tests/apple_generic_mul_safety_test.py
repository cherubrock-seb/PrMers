#!/usr/bin/env python3
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root / 'src/EngineApi.cpp').read_text()
start = s.index('  void mul(size_t dst, size_t src, uint32_t factor) {')
end = s.index('\n  void add(size_t dst, size_t src) {', start)
body = s[start:end]
assert '#if defined(__APPLE__)' in body
assert 'Apple Aevum multiplication requires set_multiplicand/prepare' in body
assert 'gpu_->regMulPrepared(reg(dst), prepared_buffers_[slot], 1);' in body
assert '#else' in body
assert 'else gpu_->regMul(reg(dst), reg(src), 1);' in body
assert 'AEVUM_APPLE_UNSAFE_GENERIC_MUL' not in body
# Apple remains fail-closed for an unprepared generic multiplication, but the
# normal GitHub prepare contract and prepared multiplication are enabled.
prepare_start = s.index('  void prepare(size_t dst, size_t src) {')
prepare_end = s.index('\n  void square_mul', prepare_start)
prepare = s[prepare_start:prepare_end]
assert 'gpu_->regPrepare(prepared_buffers_[slot], reg(dst));' in prepare
assert 'prepared_count = 0' not in s
assert 'small_factor_scratch_ = gpu_->makeBufVector(2);' in s
assert 'regAddWords(small_factor_scratch_[next], small_factor_scratch_[current])' in s
print('Aevum Apple prepared-multiplication safety test passed')
