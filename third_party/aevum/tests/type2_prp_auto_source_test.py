#!/usr/bin/env python3
from pathlib import Path

api = Path("src/EngineApi.cpp").read_text()
fft = Path("src/FFTConfig.cpp").read_text()
bpw = Path("src/fftbpw.h").read_text()
gpu = Path("src/Gpu.cpp").read_text()
trig_cpp = Path("src/Trig.cpp").read_text()
trig_h = Path("src/Trig.h").read_text()
trig_cl = Path("src/cl/trig.cl").read_text()

assert "AEVUM_EXPERIMENT_TYPE2_PRP_AUTO" not in api
assert "AEVUM_EXPERIMENT_TYPE2_PRP_AUTO" not in fft

assert "native_type2_prp_auto" in api
assert "const bool native_type2_prp_auto = false;" in api
assert 'spec = "native-prp:auto";' in api

assert "validated_type2" in api
assert "fft.shape.fft_type == FFT3261" in api
assert "fft.shape.width == 512u" in api
assert "fft.shape.middle == 8u" in api
assert "fft.shape.height == 512u" in api
assert "fft.variant == 202u" in api

assert 'report("type2-prp-auto"' in api
assert 'parse("INPLACE=1,TAIL_TRIGS=0")' in api

assert "kPrpType2MinExponent = 130000000u" in fft
assert "kPrpType2MaxExponent = 137000003u" in fft
assert "FFTShape{FFT3261, 512u, 8u, 512u}" in fft
assert "std::log2(3.0)" in fft
assert "Aevum native PRP Type2 FFT:" in fft
assert 'spec.rfind("2:", 0) == 0' in fft
assert "Type2/Type4/PFA is disabled" in fft
assert "#if !defined(__APPLE__)" in fft

# Explicit Type2 remains a deliberate developer gate; only production AUTO is ungated.
assert "AEVUM_EXPERIMENT_FFT3261" in api
assert "AEVUM_EXPERIMENT_FFT3261" in fft

# Type2 capacity/trig correctness prerequisites from the validated campaign.
assert "34.36" in bpw
assert "TrigCoefsFP32" in trig_h
assert "trigCoefsFP32" in trig_cpp
assert "TRIG_SCALE" in gpu
assert "float cosBase" in trig_cl

# Unrelated LOADS experiment must remain excluded.
prp = Path("src/PrpUseTune.cpp").read_text()
assert "44444u" in prp
assert "55555u" not in prp


# R86 gfx906 Type2 retained-width implementation policy.
assert 'const bool explicit_unroll_w_use = args_.uses("UNROLL_W");' in api
marker = "const bool gfx906_type2_prp_unroll_w1"
start = api.index(marker)
end = api.index("#endif", start)
unroll = api[start:end]
assert "!explicit_unroll_w_use" in unroll
assert "workload_ == aevum_autotune::Workload::Prp" in unroll
assert "pseudo_auto_spec" in unroll
assert 'device_vendor == "4098"' in unroll
assert 'device_name.rfind("gfx906", 0) == 0' in unroll
assert "exponent_ >= 130000000u" in unroll
assert "exponent_ <= 137000003u" in unroll
assert "!fft.isPfa()" in unroll
assert "fft.shape.fft_type == FFT3261" in unroll
assert 'fft.spec() == "2:512:8:512:202"' in unroll
assert 'pfa_use.emplace_back("UNROLL_W", "1")' in unroll

print("Aevum production Type2 PRP AUTO source gate passed")
