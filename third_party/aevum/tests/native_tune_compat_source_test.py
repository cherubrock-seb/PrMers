from pathlib import Path

root = Path(__file__).resolve().parents[1]
tune = (root / "src/TuneEntry.cpp").read_text()
fft = (root / "src/FFTConfig.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

assert "validCurrentVariant" in tune
assert "legacySingleDigitVariant" in tune
assert '"1:" + input' in tune
assert "legacy GPUOwl/PRPLL FP64 tune entries" in tune
assert "normalized %u unprefixed current WMH NTT entries" in tune
assert "Ignoring non-Aevum-FFT3161 tune entry" not in tune
assert 'masterDir / "tune.txt"' in tune

assert "three-digit WMH NTT records" in fft

assert "AEVUM_TUNE_DIR" in api
assert "Aevum GB202 native tune" in api
assert "1:512:8:512:202" in api
assert "RTX 5090" in api and "GB202" in api
assert "AEVUM_GB202_TUNE" in api
assert "LOADS/STORES" in api

print("Aevum native tune compatibility source test passed")
