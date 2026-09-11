#!/usr/bin/env python3
"""CPU-only explicit-plan gate, retaining the existing device-neutral auto resolver."""
import ctypes,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lib=ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
lib.aevum_engine_resolve_fft.argtypes=[ctypes.c_uint32,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t]
lib.aevum_engine_last_error.restype=ctypes.c_char_p
for p,plan in [(786433,'1:256:1:256:101'),(1362763,'4:256:1:256:202'),
               (21000029,'1:512:1:512:101'),(21000029,'1:1K:1:256:101'),
               (100000007,'4:4K:1:512:202'),(100000007,'1:4K:1:512:101')]:
    out=ctypes.create_string_buffer(128)
    assert lib.aevum_engine_resolve_fft(p,plan.encode(),out,len(out)),lib.aevum_engine_last_error()
    assert out.value.decode()==plan,(plan,out.value)
print('PASS: explicit middle-one PRP plans resolve without an OpenCL device')
