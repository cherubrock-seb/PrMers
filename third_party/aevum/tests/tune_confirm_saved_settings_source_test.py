from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "src/tune.cpp").read_text()

assert "static void confirmNewSettings" in src
assert "auto const originalFlags = args->flags;" in src
assert "writtenFlags[key] = to_string(value);" in src
assert "timePRP(std::min(quick, 3))" in src
assert "double const a1 = timeWith(originalFlags);" in src
assert "double const b1 = timeWith(writtenFlags);" in src
assert "double const b2 = timeWith(writtenFlags);" in src
assert "double const a2 = timeWith(originalFlags);" in src
assert "after < (1.0 - minGain) * before" in src
assert "double minGain = 0.001" in src
assert "suggestedConfigKeyVals.insert(suggestedConfigKeyVals.begin(), newConfigKeyVals.begin(), newConfigKeyVals.end());" in src
assert "newConfigKeyVals.clear();" in src
assert "args->flags = originalFlags;" in src
assert "confirmNewSettings(primes.prevPrime(fft.maxExp()), shared, fft, quick, originalFlags," in src

a1 = src.index("double const a1 = timeWith(originalFlags);")
b1 = src.index("double const b1 = timeWith(writtenFlags);")
b2 = src.index("double const b2 = timeWith(writtenFlags);")
a2 = src.index("double const a2 = timeWith(originalFlags);")
assert a1 < b1 < b2 < a2

print("Aevum tune saved-settings confirmation source test passed")
