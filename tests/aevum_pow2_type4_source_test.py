#!/usr/bin/env python3
from pathlib import Path
root = Path(__file__).resolve().parents[1]
adapter = (root / "src/aevum/EngineAevum.cpp").read_text()
engine = (root / "third_party/aevum/src/EngineApi.cpp").read_text()
gpu = (root / "third_party/aevum/src/Gpu.cpp").read_text()
fft = (root / "third_party/aevum/src/FFTConfig.cpp").read_text()
fft_h = (root / "third_party/aevum/src/FFTConfig.h").read_text()
app = (root / "src/core/App.cpp").read_text()
backend = (root / "src/marin/gpu.cpp").read_text()
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
assert "PRMERS_VERSION" in version
assert 'AEVUM_PFA_LEAD_BRIDGE' in engine
assert 'fftPCarryB' in gpu

# v100.08: gpuowl 1K radix-8 is compiled but cannot become the default.
assert 'AEVUM_RADIX1K' in fft_h
assert 'aevumRadix8For1K()' in fft_h
assert 'width == 1024 && !aevumRadix8For1K()' in fft_h
assert 'height == 1024 && !aevumRadix8For1K()' in fft_h
assert 'width == 256 ? 4 : 8' not in fft_h
assert 'AEVUM_RADIX1K must be exactly 4 or 8' in engine
assert 'radix1k=8 explicit-override' in backend
assert 'radix1k=4 safe-default' in backend

# Issue #36 regression barrier.
assert '"throughput:prp"' not in app
assert '"throughput:ll"' not in app
assert '"throughput:pm1"' in app
assert '"throughput:ecm"' in app
assert '? plan_override : ""' in app

print('PrMers Aevum throughput-auto, radix1k opt-in and PFA9 bridge source test passed')
