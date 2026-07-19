#!/usr/bin/env python3
from __future__ import annotations
import ctypes, math, sys
from pathlib import Path

def parse_number(s: str) -> int:
    mult=1
    if s[-1:].lower()=='k': mult=1024; s=s[:-1]
    elif s[-1:].lower()=='m': mult=1024*1024; s=s[:-1]
    return int(float(s)*mult)

def transform_size(spec: str) -> int:
    parts=spec.split(':')
    if parts[0].startswith('pfa'): parts=parts[1:]
    if int(parts[0]) < 60: parts=parts[1:]
    w,m,h=map(parse_number,parts[:3])
    return 2*w*m*h

def reference_power2(p: int) -> int:
    n=256
    while n <= 1<<34:
        if math.log2(n)+2*(p/n+1) < 92: return n
        n*=2
    return 0

def main():
    if len(sys.argv)<3:
        raise SystemExit(f'usage: {sys.argv[0]} libaevum_engine.so exponent')
    lib=ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
    lib.aevum_engine_resolve_fft.argtypes=[ctypes.c_uint32,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t]
    lib.aevum_engine_resolve_fft.restype=ctypes.c_int
    lib.aevum_engine_last_error.restype=ctypes.c_char_p
    p=int(sys.argv[2])
    def resolve(spec: str)->str:
        out=ctypes.create_string_buffer(256)
        if not lib.aevum_engine_resolve_fft(p,spec.encode(),out,len(out)):
            raise RuntimeError(lib.aevum_engine_last_error().decode())
        return out.value.decode()
    stock=resolve('')
    auto=resolve('pfa:auto')
    forced=[]
    for r in (3,9):
        try: forced.append((r,resolve(f'pfa:{r}')))
        except RuntimeError: pass
    ss=transform_size(stock)
    print(f'M{p}')
    print(f'  Aevum stock       : {stock} -> {ss:,} words')
    ref=reference_power2(p)
    if ref: print(f'  table power-of-two: {ref:,} (conservative mersenne2 formula)')
    for r,spec in forced:
        ps=transform_size(spec)
        ratio=ss/ps
        if ratio>=1: note=f'{ratio:.3f}x smaller than Aevum stock'
        else: note=f'{1/ratio:.3f}x larger than Aevum stock'
        print(f'  forced radix-{r:<2}   : {spec} -> {ps:,} words; {note}')
    print(f'  pfa:auto result   : {auto}')
if __name__=='__main__': main()
