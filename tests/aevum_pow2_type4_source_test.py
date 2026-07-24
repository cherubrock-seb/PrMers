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
assert 'return !useLongCarry' in gpu
assert 'fft.pfa_radix == 9' in gpu
assert 'throughput:auto' in fft
assert 'AEVUM_AUTO_POW2_TYPE4_COST' in fft
assert 'AEVUM_AUTO_PFA9_COST' in fft
assert '4.20.75-alpha-v99.86-workload-plan-policy-audit-fix' in version
assert 'AEVUM_PFA_LEAD_BRIDGE' in engine
assert 'fftPCarryB' in gpu
print('PrMers Aevum throughput-auto and PFA9 bridge source test passed')
