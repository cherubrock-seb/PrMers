#!/usr/bin/env python3
from pathlib import Path
root = Path(__file__).resolve().parents[1]
adapter = (root / "src/aevum/EngineAevum.cpp").read_text()
engine = (root / "third_party/aevum/src/EngineApi.cpp").read_text()
gpu = (root / "third_party/aevum/src/Gpu.cpp").read_text()
fft = (root / "third_party/aevum/src/FFTConfig.cpp").read_text()
version = (root / "include/core/Version.hpp").read_text()

assert 'fields[offset] == "1" || fields[offset] == "4"' in adapter
assert 'requires explicit pfa9' not in adapter
assert 'pending_reg_ = index;' in engine
assert 'AEVUM_REG_LEAD_CACHE' in engine
assert 'gpu_->regSquareStep(reg(index), pending_lead_width_, true);' in engine
assert 'gpu_->regSquareStep(reg(index), lead_in, false);' in engine
assert '!fft.isPfa() && !useLongCarry' in gpu
assert 'fft.shape.fft_type == FFT323161;' in fft
assert '4.20.68-alpha-v99.74-aevum-pow2-type4-lead-cache-exp12' in version
print('PrMers Aevum power-of-two type-4 lead-cache source test passed')
