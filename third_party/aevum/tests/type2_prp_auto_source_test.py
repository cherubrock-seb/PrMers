#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

fft = (ROOT / "src/FFTConfig.cpp").read_text()
api = (ROOT / "src/EngineApi.cpp").read_text()
gpu = (ROOT / "src/Gpu.cpp").read_text()
trig_cpp = (ROOT / "src/Trig.cpp").read_text()
trig_h = (ROOT / "src/Trig.h").read_text()
trig_cl = (ROOT / "src/cl/trig.cl").read_text()
bpw = (ROOT / "src/fftbpw.h").read_text()
prp_use = (ROOT / "src/PrpUseTune.cpp").read_text()

assert "TrigCoefsFP32" in trig_h
assert "trigCoefsFP32" in trig_cpp
assert 'reducedCosSin(int k, float cosBase)' in trig_cl
assert '{ "2:512:8:512", {34.36f, 34.36f, 34.36f, 34.36f, 34.36f, 34.36f}}' in bpw
assert 'const TrigCoefsFP32 coefs = trigCoefsFP32' in gpu

assert "kPrpType2MinExponent = 130000000u" in fft
assert "kPrpType2MaxExponent = 137000003u" in fft
assert "FFTShape{FFT3261, 512u, 8u, 512u}" in fft
assert "type2.maxBpw() * args.fftOverdrive - std::log2(3.0)" in fft

assert "native_type2_prp_auto" in api
assert 'spec = "native-prp:auto";' in api
assert "fft.shape.fft_type == FFT3261 && pseudo_auto_spec" in api
assert "validated_type2" in api
assert "fft.shape.width == 512u" in api
assert "fft.shape.middle == 8u" in api
assert "fft.shape.height == 512u" in api
assert "fft.variant == 202u" in api
assert "#if defined(__APPLE__)" in api
assert "const bool native_type2_prp_auto = false;" in api
assert 'parse("INPLACE=1,TAIL_TRIGS=0")' in api
assert 'report("type2-prp-auto"' in api

assert "AEVUM_EXPERIMENT_FFT3261" in fft
assert "AEVUM_EXPERIMENT_FFT3261" in api
assert 'spec.rfind("2:", 0) == 0' in fft
assert "Type2/Type4/PFA is disabled" in fft
assert "#if !defined(__APPLE__)" in fft
assert "AEVUM_EXPERIMENT_TYPE2_PRP_AUTO" not in fft
assert "AEVUM_EXPERIMENT_TYPE2_PRP_AUTO" not in api

# The unrelated LOADS=5 commit from PR #16 ancestry is deliberately excluded.
assert "44444u" in prp_use
assert "55555u" not in prp_use

print("Aevum validated Type2 PRP AUTO source gate passed")
