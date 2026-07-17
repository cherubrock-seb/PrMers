#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

needle = "void Gpu::regSetU32(Buffer<Word>& dst, u32 value)"
pos = gpu.index(needle)
body = gpu[pos:pos + 900]
assert "#if defined(__APPLE__)" in body
assert "writeIn(dst, makeWords(E, value));" in body
assert "dst.set(static_cast<Word>(value));" in body
assert body.index("writeIn(dst, makeWords(E, value));") < body.index("#else")
assert "set_u32 uses canonical compact-word upload" in api
print("Apple canonical set_u32 routing audit passed")
