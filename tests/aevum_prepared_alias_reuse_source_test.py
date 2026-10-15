#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "third_party/aevum/src/EngineApi.cpp").read_text()

assert 'AEVUM_PREPARED_ALIAS_REUSE must be 0 or 1' in src
assert 'prepared_alias_reuse_enabled_ = std::strcmp(alias_value, "1") == 0;' in src
assert 'bool prepared_alias_reuse_enabled_ = false;' in src

assert 'std::vector<size_t> regs;' in src
assert 'const auto& aliases = prepared_slots_[i].regs;' in src
assert 'std::find(aliases.begin(), aliases.end(), index) != aliases.end()' in src
assert 'prepared_slots_[slot].regs.assign(1, index);' in src

copy_start = src.index('  void copy(size_t dst, size_t src) {')
copy_end = src.index('\n  void prepare(size_t dst, size_t src) {', copy_start)
copy_body = src[copy_start:copy_end]
assert copy_body.index('find_prepared(src, true)') < copy_body.index('invalidate_if(dst)')
assert 'prepared_slots_[source_slot].regs.push_back(dst);' in copy_body

prepare_start = copy_end + 1
prepare_end = src.index('\n  void square_mul(size_t index, uint32_t factor) {', prepare_start)
prepare_body = src[prepare_start:prepare_end]
assert 'if (source_slot != no_prepared)' in prepare_body
assert 'prepared_slots_[source_slot].regs.push_back(dst);' in prepare_body
assert 'gpu_->regPrepare(prepared_buffers_[slot], reg(dst));' in prepare_body
assert 'prepared_slots_[slot].regs.push_back(src);' in prepare_body

invalidate_start = src.index('  void invalidate_if(size_t index) {')
invalidate_end = src.index('\n  void multiply_small(', invalidate_start)
invalidate_body = src[invalidate_start:invalidate_end]
assert 'aliases.erase(std::remove(aliases.begin(), aliases.end(), index), aliases.end());' in invalidate_body

assert 'prepared_slots_[i].reg == index' not in src
assert 'prepared_slots_[slot].reg = index' not in src
assert 'slot.reg = no_prepared' not in src

print("Aevum prepared alias reuse source contract: PASS")
