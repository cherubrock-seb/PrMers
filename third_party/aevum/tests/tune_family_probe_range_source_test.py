from pathlib import Path

root = Path(__file__).resolve().parents[1]
tune = (root / "src/tune.cpp").read_text()

assert "u64 probe_exponent = 141000001;" in tune
assert "probe_exponent < min_exponent || probe_exponent > max_exponent" in tune
assert "primes.prevPrime(min_exponent / 2 + max_exponent / 2)" in tune
assert 'FFTShape::multiSpec(to_string((u32) type) + ":256K-128M")' in tune
assert "bpw >= candidate.minBpw()" in tune
assert "candidate.maxExp() >= probe_exponent" in tune
assert "defaultFFTShape = FFTShape(FFT64, 512, 16, 512);" in tune
assert "defaultNTTShape = FFTShape(FFT3161, 512, 8, 512);" in tune

probe = tune.split("auto smallest =", 1)[1].split("fft = smallest", 1)[0]
assert "FFTShape::allShapes()" not in probe
assert "shape.isFavoredShape()" in probe

print("Aevum range-aware family probe source test passed")
