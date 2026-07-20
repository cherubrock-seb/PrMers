#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fft = (root / "src/FFTConfig.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

for text in (fft, api):
    assert 'AEVUM_APPLE_DIAGNOSTIC_PLANES' in text
    assert '#if defined(__APPLE__)' in text
    assert 'fft.shape.fft_type == FFT31 || fft.shape.fft_type == FFT61' in text
    assert 'supported_aevum_type' in text
    assert 'FFT323161' in text
    assert 'fft.shape.fft_type == FFT323161' in text
    assert '!supported_aevum_type && !apple_diagnostic_plane' in text

assert 'Apple Aevum plane-isolation diagnostic' in api
assert 'FFT type 4 (power-of-two or explicit PFA9)' in fft
print('Apple plane-isolation compile gate and FFT323161 audit passed')
