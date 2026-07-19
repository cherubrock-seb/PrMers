#!/usr/bin/env python3
from __future__ import annotations
import ctypes
import sys
from pathlib import Path

libpath = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build-engine/libaevum_engine.so')
lib = ctypes.CDLL(str(libpath))
lib.aevum_engine_resolve_fft.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
lib.aevum_engine_resolve_fft.restype = ctypes.c_int
lib.aevum_engine_last_error.restype = ctypes.c_char_p

def resolve(p: int, spec: str) -> str:
    out = ctypes.create_string_buffer(256)
    if not lib.aevum_engine_resolve_fft(p, spec.encode(), out, len(out)):
        raise RuntimeError(lib.aevum_engine_last_error().decode())
    return out.value.decode()

def expect(p: int, spec: str, prefix: str) -> None:
    got = resolve(p, spec)
    assert got.startswith(prefix), (p, spec, got, prefix)
    print(f'M{p} {spec or "stock"}: {got}')

# Representative stock, automatic, and forced cases.
for p, spec, prefix in [
    (100000019, '', '1:1K:8:256:101'),
    (100000019, 'pfa:auto', 'pfa3:1:512:3:1K:101'),
    (100000019, 'pfa:3', 'pfa3:'),
    (175000001, '', '1:1K:16:256:101'),
    (175000001, 'pfa:auto', 'pfa9:1:512:9:512:101'),
    (175000001, 'pfa:3', 'pfa3:'),
    (175000001, 'pfa:9', 'pfa9:'),
    (142606549, 'pfa:auto', '1:1K:8:256:101'),
]:
    expect(p, spec, prefix)

# Exact pfa:auto ranges documented in the public README files.  The list is
# ordered by exponent and includes both radix 3 and radix 9 windows.
automatic_ranges = [
    (10627319, 15724707, 'pfa3:1:256:3:256:101'),
    (21071135, 31284264, 'pfa3:1:256:3:512:101'),
    (41922069, 46560704, 'pfa9:1:256:9:256:101'),
    (46560705, 62080936, 'pfa3:1:512:3:512:101'),
    (83194017, 92625960, 'pfa9:1:256:9:512:101'),
    (92625961, 123343992, 'pfa3:1:512:3:1K:101'),
    (165507233, 183789168, 'pfa9:1:512:9:512:101'),
    (183789169, 244737648, 'pfa3:1:1K:3:1K:101'),
    (328414017, 365879616, 'pfa9:1:512:9:1K:101'),
    (365879617, 487210368, 'pfa3:1:4K:3:512:101'),
    (653808129, 725153152, 'pfa9:1:1K:9:1K:101'),
    (725153153, 965612672, 'pfa3:1:4K:3:1K:101'),
    (1295872129, 1440869120, 'pfa9:1:4K:9:512:101'),
    (2574967041, 2862863872, 'pfa9:1:4K:9:1K:101'),
]

def expected_auto(p: int) -> str:
    for start, end, plan in automatic_ranges:
        if start <= p <= end:
            return plan
    return '1:'

# Verify every documented transition, including adjacent radix-9/radix-3
# windows where end+1 is another PFA plan rather than stock.
points: set[int] = set()
for start, end, _ in automatic_ranges:
    points.update((start, end))
    if start > 1:
        points.add(start - 1)
    if end < 0xFFFFFFFF:
        points.add(end + 1)
for p in sorted(points):
    expect(p, 'pfa:auto', expected_auto(p))

print('Native PFA radix-3/radix-9 plan policy and documented boundaries passed')
