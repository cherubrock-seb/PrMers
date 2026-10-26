#!/usr/bin/env python3
"""Source wiring gate for LL-SAFE factor-3 FFT capacity policy."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
text = (root / "src/modes/RunLlSafeMarin.cpp").read_text(encoding="utf-8")
start = text.index("int App::runLlSafeMarin()")
body = text[start:text.index("\nint App::", start + 1) if "\nint App::" in text[start + 1:] else len(text)]

assert 'aevum_engine_resolve_factor_safe_fft(' in body
assert 'p, 3u, &safe_transform, &safe_fft, &safe_reason' in body
assert 'requested_fft.empty()' in body
assert 'engine::gpu_backend::auto_select' in body
assert 'engine::gpu_backend::aevum' in body
assert 'engine::configure_gpu_backend(\n                requested_backend, safe_fft, engine::gpu_workload::ll);' in body
assert 'engine::gpu_backend::marin, "", engine::gpu_workload::ll' in body
assert 'Forced Aevum LL-SAFE cannot obtain a factor-3-safe FFT plan' in body
assert 'square_mul(static_cast<engine::Reg>(B), static_cast<uint32_t>(3))' in body

print("LL-SAFE factor-capacity source gate passed")
