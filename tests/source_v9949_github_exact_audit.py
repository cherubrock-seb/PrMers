#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
pm1 = (root / "src/modes/RunPM1.cpp").read_text()
api = (root / "third_party/aevum/src/EngineApi.cpp").read_text()
gpu = (root / "third_party/aevum/src/Gpu.cpp").read_text()
fft = (root / "third_party/aevum/src/FFTConfig.cpp").read_text()
adapter = (root / "src/aevum/EngineAevum.cpp").read_text()
version = (root / "include/core/Version.hpp").read_text()
aevum_version = (root / "third_party/aevum/src/version.inc").read_text()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")

require("v99.51-aevum-apple-ffthin-gf61-arg-order" in version, "PrMers v99.51 version")
require('"v0.3.45"' in aevum_version, "Aevum v0.3.45 version")

# Exact GitHub P-1 policy: Aevum never uses fast3 and multiplies by prepared base 3.
require("Aevum uses the generic square plus base-3 multiply path; fast3 is disabled." in pm1,
        "GitHub Aevum P-1 policy banner")
require("bool useFast3 = useFast3Candidate && (nextStart == 0) && !aevum_backend;" in pm1,
        "Aevum exclusion from fast3")
require("eng->square_mul(RSTATE); if (b) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RSTATE, RTMP); }" in pm1,
        "GitHub square + prepared base-3 multiply loop")
for forbidden in (
    "native square_mul(x,3)",
    "safe square_mul(x,3)",
    "carryM, generic multiplication",
    "Gerbicz-Li disabled because its accumulator requires generic multiplication",
    "Apple Aevum safe mode: exact GF31-only",
    "Apple Aevum safe mode: FP32+GF31",
):
    require(forbidden not in pm1, f"removed experimental P-1 path: {forbidden}")

# Engine remains strict FFT3161 and keeps GitHub prepared cache behavior.
require("if (fft.shape.fft_type != FFT3161) throw std::runtime_error(\"Aevum engine requires FFT3161\");" in api,
        "strict FFT3161 engine")
require("size_t prepared_count = std::min<size_t>(register_count_, 2);" in api,
        "GitHub prepared cache default")
require("prepared_count = 0" not in api, "Apple must not disable prepared cache")
require("AEVUM_APPLE_UNSAFE_GENERIC_MUL" not in api, "no unsafe generic-multiply switch")
require("gpu_->regPrepare(prepared_buffers_[slot], reg(dst));" in api, "prepared multiplicand API")
require("small_factor_scratch_ = gpu_->makeBufVector(2);" in api, "Apple two-buffer small-factor scratch")
require("gpu_->regCopy(small_factor_scratch_[next], small_factor_scratch_[current]);" in api,
        "Apple non-aliasing small-factor doubling copy")
require("gpu_->regAddWords(small_factor_scratch_[next], small_factor_scratch_[current]);" in api,
        "Apple non-aliasing small-factor doubling add")

# Apple-only adaptation: complete the invariant transform and consume it through upstream MUL_LOW.
prepare_block = re.search(r"void Gpu::regPrepare\(Buffer<double>& prepared, Buffer<Word>& src\) \{(.*?)\n\}", gpu, re.S)
require(prepare_block is not None, "regPrepare prepared overload")
pb = prepare_block.group(1)
require("#if defined(__APPLE__)" in pb, "Apple guard in regPrepare")
require("fftP(buf1, src);" in pb and "fftMidIn(buf1);" in pb and "fftHin(prepared, buf1);" in pb,
        "Apple full prepared transform")
require("#else" in pb and "fftP(prepared, src);" in pb and "fftMidIn(prepared);" in pb,
        "unchanged non-Apple prepared route")

mul_block = re.search(r"void Gpu::regMulPrepared\(Buffer<Word>& dst, Buffer<double>& prepared, u32 factor\) \{(.*?)\n\}", gpu, re.S)
require(mul_block is not None, "regMulPrepared prepared overload")
mb = mul_block.group(1)
require("#if defined(__APPLE__)" in mb, "Apple guard in regMulPrepared")
for expected in ("fftP(buf1, dst);", "fftMidIn(buf1);", "tailMulLow(buf1, prepared);",
                 "fftMidOut(buf1);", "fftW(buf3, buf1);", "carryA(dst, buf3);", "carryB(dst);"):
    require(expected in mb, f"Apple prepared route contains {expected}")
require("#else" in mb and "mul(dst, prepared, buf1, false);" in mb,
        "unchanged non-Apple prepared multiply")

# The upstream low-tail kernels for BOTH residues must remain compiled.
require('K(ktailMulLowGF31' in gpu and '"tailMulGF31"' in gpu and '-DMUL_LOW=1' in gpu,
        "GF31 upstream tailMulLow")
require('K(ktailMulLowGF61' in gpu and '"tailMulGF61"' in gpu,
        "GF61 upstream tailMulLow")

# Reject alternate arithmetic engines and parser broadening.
combined = "\n".join((api, gpu, fft, adapter, pm1))
for forbidden in ("Apple Aevum safe FFT3231", "Apple Aevum safe FFT31", "FFT_TYPE=50 for exponent", "FFT_TYPE=52 for exponent", "GF31-only arithmetic"):
    require(forbidden not in combined, f"alternate Apple arithmetic selector removed: {forbidden}")
require('fields[0] != "1"' in adapter, "PrMers adapter remains FFT3161-only")

print("PASS: v99.51 preserves the GitHub FFT3161/GF31+GF61 contract and fixes only the Apple fftHinGF61 ABI")
