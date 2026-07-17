#!/usr/bin/env python3
import ctypes
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lib_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "build-engine/libaevum_engine.so"
lib = ctypes.CDLL(str(lib_path))
lib.aevum_engine_resolve_fft.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
lib.aevum_engine_resolve_fft.restype = ctypes.c_int
lib.aevum_engine_last_error.restype = ctypes.c_char_p
expected = {
    859553: "1:256:2:256:101",
    1362763: "1:256:2:256:101",
    136279841: "1:1K:8:256:101",
    2147483647: "1:4K:16:512:101",
}
for exponent, plan in expected.items():
    out = ctypes.create_string_buffer(128)
    if not lib.aevum_engine_resolve_fft(exponent, b"", out, len(out)):
        raise SystemExit(f"FAIL M{exponent}: {lib.aevum_engine_last_error().decode()}")
    got = out.value.decode()
    if got != plan:
        raise SystemExit(f"FAIL M{exponent}: expected {plan}, got {got}")
    print(f"M{exponent}: {got}")
print("PASS: exact GitHub FFT3161 plans retained")
