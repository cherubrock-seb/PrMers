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
use_options = (ROOT / "src/UseOptions.h").read_text()
math_cl = (ROOT / "src/cl/math.cl").read_text()

# Corrected FFT3261 prerequisite arithmetic/capacity from the validated branch.
assert "TrigCoefsFP32" in trig_h
assert "trigCoefsFP32" in trig_cpp
assert 'reducedCosSin(int k, float cosBase)' in trig_cl
assert '{ "2:512:8:512", {34.36f, 34.36f, 34.36f, 34.36f, 34.36f, 34.36f}}' in bpw
assert 'const TrigCoefsFP32 coefs = trigCoefsFP32' in gpu

# Production selector: tune data remains authoritative, then the narrow
# validated Type2 window precedes the existing wider Type1 4M window.
assert "kPrpType2MinExponent = 130000000u" in fft
assert "kPrpType2MaxExponent = 137000003u" in fft
assert "FFTShape{FFT3261, 512u, 8u, 512u}" in fft
assert "type2.maxBpw() * args.fftOverdrive - std::log2(3.0)" in fft
assert "native_type2_prp_auto" in api
assert 'spec = "native-prp:auto";' in api
assert "fft.shape.fft_type == FFT3261 && pseudo_auto_spec" in api
assert "const bool explicit_fft_spec = !spec.empty() && !pseudo_auto_spec;" in api
assert "const bool autotune_eligible = !pseudo_auto_spec && autotune_mode != aevum_autotune::Mode::Off" in api
assert "FFTConfig native_fft = FFTConfig::bestFit(args_, exponent_, spec);" in api

# Validated Type2 PRP-use profile is below manual/tune priority.
assert 'parse("INPLACE=1,TAIL_TRIGS=0")' in api
assert 'report("type2-prp-auto"' in api

# Manual Type2 remains explicitly gated; production AUTO itself is not.
assert "AEVUM_EXPERIMENT_FFT3261" in fft
assert "AEVUM_EXPERIMENT_FFT3261" in api
assert "AEVUM_EXPERIMENT_TYPE2_PRP_AUTO" not in fft
assert "AEVUM_EXPERIMENT_TYPE2_PRP_AUTO" not in api

# Do not accidentally import unrelated PR #16 ancestry (LOADS=5 admission).
assert "44444u" in prp_use
assert "55555u" not in prp_use

# PR38 gfx906 Type2 middle-in GF61 cmul policy.
assert '"AEVUM_GF61_CMUL_ALT"' in use_options
assert 'shared.args->uses("AEVUM_GF61_CMUL_ALT")' in api
marker = "const bool gfx906_type2_prp_gf61_cmul_alt0"
start = api.index(marker)
end = api.index('report("type2-prp-auto"', start)
cmul = api[start:end]
assert "type2_device_name" in cmul
assert "type2_device_vendor" in cmul
assert 'type2_device_vendor == "4098"' in cmul
assert 'type2_device_name.rfind("gfx906", 0) == 0' in cmul
assert "exponent >= 130000000u" in cmul
assert "exponent <= 137000003u" in cmul
assert "!fft.isPfa()" in cmul
assert "fft.shape.fft_type == FFT3261" in cmul
assert 'fft.spec() == "2:512:8:512:202"' in cmul
assert 'use.emplace_back("AEVUM_GF61_CMUL_ALT", "0")' in cmul
assert "#elif !defined(AEVUM_GF61_CMUL_ALT) || !AEVUM_GF61_CMUL_ALT" in math_cl
assert "canonical three-product cmul path" in api
assert "explicit -use AEVUM_GF61_CMUL_ALT precedence" in api

print("Aevum validated Type2 PRP AUTO source gate passed")
