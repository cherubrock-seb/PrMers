#!/usr/bin/env python3
from pathlib import Path
r=Path(__file__).resolve().parents[1]
a=r/'third_party/aevum'
g=(a/'src/Gpu.cpp').read_text()
h=(a/'src/Gpu.h').read_text()
cl=(a/'src/cl/ffthin.cl').read_text()
assert 'fftHinGF61LoadScalarApple' in cl
assert 'fftHinGF61FftRadixApple' in cl
assert 'fftHinGF61FftTwiddleApple' in cl
assert 'fftHinGF61FftShuffleApple' in cl
assert 'fftHinGF61FftFinalApple' in cl
assert 'fftHinGF61ApplePlaceholder' in cl
assert '#if defined(__APPLE__)' in g
assert 'K(kfftHinGF61,           "ffthin.cl",  "fftHinGF61ApplePlaceholder"' in g
assert '#else\n  K(kfftHinGF61,           "ffthin.cl",  "fftHinGF61"' in g
assert 'if (fft.shape.fft_type == FFT3161)' in g
assert 'FFT_TYPE=50' not in g and 'FFT_TYPE=52' not in g
assert 'GF31-only' not in g and 'FP32+GF31' not in g
assert 'fft.shape.fft_type != FFT3161' in (a/'src/EngineApi.cpp').read_text()
assert 'fftHinGF61 uses exact global staging' in (a/'src/EngineApi.cpp').read_text()
print('PrMers v99.51 Apple staged fftHinGF61 source audit passed')
