#!/usr/bin/env python3
"""Compile actual radix4/8/16 source against weak GF31 host arithmetic.

Check natural output order and modular equivalence against an independent DFT.
The host facade checks 62/63-bit reduction bounds; it does not validate GPU ISA.
"""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
actual = "\n".join(
    re.sub(r'^\s*#(?:include|pragma once).*$', '', (root/'src/cl'/name).read_text(), flags=re.M)
    for name in ('fft4.cl', 'fft8.cl', 'fft16.cl'))
prefix = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
using u32=uint32_t; using i32=int32_t; using u64=uint64_t; using Z31=u32;
constexpr Z31 M31=2147483647U;
struct GF31 { Z31 x,y; };
#define OVERLOAD
#define U2(x,y) GF31{x,y}
#define SWAP(a,b) std::swap(a,b)
#define FFT_FP64 0
#define FFT_FP32 0
#define NTT_GF31 1
#define NTT_GF61 0
#define FFT_TYPE 1
#define FFT3161 1
#define PFA_RADIX 0
#define WIDTH 512
#define MIDDLE 16
#define SMALL_HEIGHT 512
Z31 modM31(u32 a) { assert(a!=UINT32_MAX); return (a&M31)+(a>>31); }
Z31 modM31(i32 a) { assert(a!=INT32_MIN); return (a&M31)+(a>>31); }
Z31 modM31(u64 a, unsigned bits=64) {
  if(bits<64) assert((a>>bits)==0);
  assert(a<0xffffffff7fffffffULL);
  return modM31(Z31(a&M31)+Z31((a>>31)&M31)+Z31(a>>62));
}
Z31 neg(Z31 a) { assert(a<=M31); return M31-a; }
Z31 add(Z31 a,Z31 b) { return modM31(a+b); }
Z31 sub(Z31 a,Z31 b) { return modM31(i32(a-b)); }
GF31 add(GF31 a,GF31 b) { return U2(add(a.x,b.x),add(a.y,b.y)); }
GF31 sub(GF31 a,GF31 b) { return U2(sub(a.x,b.x),sub(a.y,b.y)); }
Z31 shl(Z31 a,unsigned n) { return modM31(u64(a)<<n); }
GF31 cmul(GF31 a,GF31 b) {
  return U2(modM31(u64(a.x)*b.x+u64(a.y)*neg(b.y),63),
            modM31(u64(a.x)*b.y+u64(a.y)*b.x,63));
}
// Same product and range choices as the portable math.cl cmul_const.
GF31 cmul_const(GF31 a,GF31 b) {
  return U2(modM31(u64(a.x)*b.x+u64(a.y)*neg(b.y),(M31-b.y+b.x<=M31)?62:63),
            modM31(u64(a.x)*b.y+u64(a.y)*b.x,(b.x+b.y<=M31)?62:63));
}
GF31 mul_t4(GF31 a) { return U2(neg(a.y),a.x); }
GF31 mul_t8(GF31 a) { return U2(shl(sub(a.x,a.y),15),shl(add(a.x,a.y),15)); }
GF31 mul_3t8(GF31 a) { return U2(shl(neg(add(a.x,a.y)),15),shl(sub(a.x,a.y),15)); }
void X2(GF31& a,GF31& b) { const auto t=a; a=add(t,b); b=sub(t,b); }
void X2t4(GF31& a,GF31& b) { b=mul_t4(b); X2(a,b); }
void X2_mul_t4(GF31& a,GF31& b) { X2(a,b); b=mul_t4(b); }
void X2_mul_t8(GF31& a,GF31& b) { X2(a,b); b=mul_t8(b); }
void X2_mul_3t8(GF31& a,GF31& b) { X2(a,b); b=mul_3t8(b); }
'''
suffix = r'''
GF31 refmul(GF31 a,GF31 b) {
  return U2(Z31((u64(a.x)*b.x%M31+M31-u64(a.y)*b.y%M31)%M31),
            Z31((u64(a.x)*b.y+u64(a.y)*b.x)%M31));
}
GF31 power(GF31 a,unsigned n) { GF31 b{1,0}; for(;n;n>>=1,a=refmul(a,a))if(n&1)b=refmul(a,b); return b; }
int main() {
  const GF31 w{1556715293,978592373};
  assert(power(w,4).x==0 && power(w,4).y==1);
  assert(power(w,8).x==M31-1 && power(w,16).x==1);
  std::mt19937 gen(316116);
  for(unsigned trial=0;trial<1200;++trial) {
    GF31 input[16],a[16];
    for(unsigned k=0;k<16;++k) {
      if(trial<4) input[k]=U2(trial&1?M31:0,trial&2?M31:0);
      else if(trial<68) input[k]=U2(k==(trial-4)%16?((trial-4)/16%2?M31-1:1):0,
                                   k==(trial-4)%16?((trial-4)/16>=2?1:0):M31);
      else if(trial<100) input[k]=U2((k+trial)%2?M31:1,(k+trial)%3?M31-1:0);
      else input[k]=U2(Z31(gen()%(u64(M31)+1)),Z31(gen()%(u64(M31)+1)));
      a[k]=input[k];
    }
    fft16(a);
    for(unsigned j=0;j<16;++j) {
      u64 x=0,y=0;
      for(unsigned k=0;k<16;++k) { const auto v=refmul(input[k],power(w,j*k)); x=(x+v.x)%M31;y=(y+v.y)%M31; }
      assert(a[j].x<=M31 && a[j].y<=M31);
      assert(a[j].x%M31==x && a[j].y%M31==y);
      // The engine implements inverse NTT by exchanging the components.
      a[j]=U2(a[j].y,a[j].x);
    }
    fft16(a);
    for(unsigned k=0;k<16;++k) {
      assert(a[k].x%M31==u64(input[k].y)*16%M31);
      assert(a[k].y%M31==u64(input[k].x)*16%M31);
    }
  }
  std::cout << "GF31 middle16=" << AEVUM_GF31_MIDDLE16 << ": 1200 direct DFT/inverse cases and reduction bounds PASS\n";
}
'''
with tempfile.TemporaryDirectory() as tmp:
    cpp, exe = Path(tmp)/'fft.cpp', Path(tmp)/'fft'
    cpp.write_text(prefix+actual+suffix)
    for mode in (0,1):
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined',
                        '-Wno-unknown-pragmas',f'-DAEVUM_GF31_MIDDLE16={mode}',
                        str(cpp),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
