#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()
config = (root / "src/FFTConfig.cpp").read_text()


def must(x, msg):
    if not x:
        raise SystemExit("FAIL: " + msg)

must("FFT3161" in api and "experimental FFT323161 PFA9" in api,
     "FFT3161 plus explicit FFT323161 PFA9 strictness")
must("prepared_count = 0" not in api, "prepared cache enabled on Apple")
must("fftHin(prepared, buf1);" in gpu, "Apple prepared full-height transform")
must("tailMulLow(buf1, prepared);" in gpu, "Apple prepared route remains selected")
must("runAppleTailMulLowGF61Fft" in gpu, "Apple GF61 MUL_LOW is staged")
must("ktailMulGF61PairSpecialScalarApple(*raw, rawBase, *out, fullBase, *in2, fullBase, bufTrigH);" in gpu,
     "prepared GF61 operand feeds exact special pairing")
must("ktailMulLowGF31" in gpu and "ktailMulLowGF61" in gpu, "both CRT residues retained")
must("-DMUL_LOW=1" in gpu, "upstream MUL_LOW specialization retained")
must("Apple Aevum safe FFT3231" not in config and "Apple Aevum safe FFT31" not in config, "no alternate Apple FFT selection")

# Make sure every modified execution sequence is textually enclosed by an Apple guard.
for signature in (
    r"void Gpu::regPrepare\(Buffer<Word>& src\)",
    r"void Gpu::regPrepare\(Buffer<double>& prepared, Buffer<Word>& src\)",
    r"void Gpu::regMulPrepared\(Buffer<Word>& dst, u32 factor\)",
    r"void Gpu::regMulPrepared\(Buffer<Word>& dst, Buffer<double>& prepared, u32 factor\)",
):
    m = re.search(signature + r" \{(.*?)\n\}", gpu, re.S)
    must(m and "#if defined(__APPLE__)" in m.group(1) and "#else" in m.group(1), signature + " Apple isolation")

print("PASS: Apple prepared FFT3161 route is isolated and keeps GF31+GF61")
