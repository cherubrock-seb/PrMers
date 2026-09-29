from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
header = (ROOT / "include/aevum/AutoPolicy.hpp").read_text(encoding="utf-8")
policy = (ROOT / "src/aevum/AutoPolicy.cpp").read_text(encoding="utf-8")
gpu = (ROOT / "src/marin/gpu.cpp").read_text(encoding="utf-8")

assert "runtime_compare = false" in header
assert "result.runtime_compare = true" in policy
assert "safety=runtime-compare" in policy
assert "result.aevum_transform == kOrdinaryPrpQuarantineWords" in policy

assert "quarantine_words = 524288u" in gpu
assert "transform <= quarantine_words" in gpu
assert "exact_prp_operation_probe" in gpu
assert "set_multiplicand(R2, R0)" in gpu
assert "eng.mul(R1, R2)" in gpu
assert "eng.square_mul(R3, 3u)" in gpu
assert "mpz_cmp(a, b) == 0" in gpu
assert "word-exact differential mismatch" in gpu

assert "measured_square_ns" in gpu
assert "PRMERS_BACKEND_AUTO_MIN_SPEEDUP" in gpu
assert "speedup >= min_speedup" in gpu
assert "cache=hit" in gpu
assert "cache=miss" in gpu
assert "PRMERS_BACKEND_AUTO_CACHE_DIR" in gpu

assert "selected_workload == gpu_workload::prp" in gpu
assert "reg_count == 8u" in gpu
assert "created->get_size() <= kOrdinaryPrpQuarantineWords" in gpu
assert "Aevum ordinary PRP <=512K is temporarily quarantined" in gpu

print("Aevum low-range ordinary PRP measured safety selector: PASS")


fft_h = (ROOT / "third_party/aevum/src/FFTConfig.h").read_text(
    encoding="utf-8")
fft = (ROOT / "third_party/aevum/src/FFTConfig.cpp").read_text(
    encoding="utf-8")
api = (ROOT / "third_party/aevum/src/EngineApi.cpp").read_text(
    encoding="utf-8")
task = (ROOT / "third_party/aevum/src/Task.cpp").read_text(
    encoding="utf-8")

assert "knownUnsafeOrdinaryPrp" in fft_h
assert "kFirstReproducedBadExponent = 19121591u" in fft
assert "shape.width == 256u" in fft
assert "shape.middle == 4u" in fft
assert "shape.height == 256u" in fft
assert '"1:512:4:256:101"' in fft
assert "promoteKnownUnsafeOrdinaryPrp" in fft

assert "workload_ == aevum_autotune::Workload::Prp" in api
assert "cached_fft->knownUnsafeOrdinaryPrp(exponent_)" in api
assert "tuned.fft.knownUnsafeOrdinaryPrp(exponent)" in api
assert "fft->knownUnsafeOrdinaryPrp(exponent)" in api
assert "promoted to validated %s" in api

assert "kind == PRP" in task
assert "fft.knownUnsafeOrdinaryPrp(exponent)" in task
assert "promoteKnownUnsafeOrdinaryPrp" in task
