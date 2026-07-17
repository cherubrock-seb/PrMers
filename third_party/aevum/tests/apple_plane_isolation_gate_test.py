#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fft = (root / "src/FFTConfig.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

for text in (fft, api):
    assert 'AEVUM_APPLE_DIAGNOSTIC_PLANES' in text
    assert '#if defined(__APPLE__)' in text
    assert 'fft.shape.fft_type == FFT31 || fft.shape.fft_type == FFT61' in text

assert 'fft.shape.fft_type != FFT3161 && !apple_diagnostic_plane' in fft
assert 'fft.shape.fft_type != FFT3161 && !apple_diagnostic_plane' in api
assert 'Apple Aevum plane-isolation diagnostic' in api
assert 'Aevum accepts only FFT type 1' in fft
print('Apple plane-isolation compile gate audit passed')
