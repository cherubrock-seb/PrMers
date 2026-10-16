#!/usr/bin/env python3
"""Compile actual PFA7 GF61 cosine-only and dual-base source via C++/UBSan.

Independent direct DFT oracle; no GPU compilation or performance claim.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root/'src/cl/fft-middle.cl').read_text()
start = source.index('#ifndef AEVUM_PFA7_GF61_COSBASE')
end = source.index('\n#endif\n\n#if PFA_RADIX\n// WIDTH', start)
actual = source[start:end]
prefix = r'''
#include <cassert>
#include <cstdint>
#include <random>
#include <iostream>
using u32=uint32_t; using u64=uint64_t; using Z61=u64;
using u128=unsigned __int128;
constexpr Z61 M61=(u64(1)<<61)-1;
struct GF61 { Z61 x,y; };
#define OVERLOAD
#define U2(x,y) GF61{x,y}
#define PFA_RADIX 7
#define FFT_TYPE 1
#define FFT3161 1
#define WIDTH 512
#define MIDDLE 7
#define SMALL_HEIGHT 512
#define AEVUM_GF61_LIMB32 0
#define PFA7_ROOT61 ROOTULL
#define PFA7_INVROOT61 INVROOTULL
#define PFA7_INV7_61 INV7ULL
GF61 operator+(GF61 a,GF61 b) { return U2(a.x+b.x,a.y+b.y); }
GF61 operator-(GF61 a,GF61 b) { return U2(a.x-b.x,a.y-b.y); }
u128 mul64(u64 a,u64 b) { return u128(a)*b; }
u128 mad64(u64 a,u64 b,u64 c) { return u128(a)*b+c; }
u128 mad64(u64 a,u64 b,u128 c) { return u128(a)*b+c; }
Z61 modM61(Z61 a) { return (a&M61)+(a>>61); }
GF61 modM61(GF61 a) { return U2(modM61(a.x),modM61(a.y)); }
GF61 modM61q(GF61 a,u32 count) { return modM61(a+U2(count*M61,count*M61)); }
u64 weakModM61(u128 a,u32 bits) {
  assert((a>>126)==0); // bound for every actual candidate dot
  u64 lo=u64(a),hi=u64(a>>64);
  if(bits<=125) { assert((a>>125)==0);return (lo&M61)+(hi<<3)+(lo>>61); }
  return (lo&M61)+(((hi<<3)+(lo>>61))&M61)+(hi>>58);
}
Z61 mul(Z61 a,Z61 b) { return modM61(weakModM61(mul64(a,b),125)); }
GF61 add(GF61 a,GF61 b) { return modM61(a+b); }
GF61 sub(GF61 a,GF61 b) { return modM61q(a-b,2); }
u64 power(u64 a,u64 n) { u64 v=1; for(;n;n>>=1,a=u128(a)*a%M61)if(n&1)v=u128(v)*a%M61;return v; }
'''
suffix = r'''
int main() {
  const u64 root=power(5,(M61-1)/7),inv=power(root,6);
  assert(root==PFA7_ROOT61 && inv==PFA7_INVROOT61);
  std::mt19937_64 gen(617031);
  for(unsigned trial=0;trial<1200;++trial) {
    GF61 input[7],a[7];
    for(unsigned k=0;k<7;++k) {
      if(trial<4) input[k]=U2(trial&1?M61:0,trial&2?M61:0);
      else if(trial<32) input[k]=U2(k==(trial-4)%7?M61-1:0,k==(trial-4)%7?1:M61);
      else if(trial<112) input[k]=U2((k+trial)%2?M61+(trial%8):1,(k+trial)%3?M61-1:0);
      else input[k]=U2(gen()%(M61+8),gen()%(M61+8));
    }
    for(unsigned inverse=0;inverse<2;++inverse) {
      for(unsigned k=0;k<7;++k)a[k]=input[k];
      if(inverse)pfaInverseMiddle(a);else pfaForwardMiddle(a);
      for(unsigned j=0;j<7;++j) {
        u64 x=0,y=0;
        for(unsigned k=0;k<7;++k) {
          const u64 w=power(inverse?inv:root,j*k);
          x=(x+u128(input[k].x)*w)%M61; y=(y+u128(input[k].y)*w)%M61;
        }
        if(inverse){x=u128(x)*PFA7_INV7_61%M61;y=u128(y)*PFA7_INV7_61%M61;}
        assert(a[j].x<=M61+7 && a[j].y<=M61+7);
        assert(a[j].x%M61==x && a[j].y%M61==y);
      }
    }
    for(unsigned k=0;k<7;++k)a[k]=input[k];
    pfaForwardMiddle(a);pfaInverseMiddle(a);
    for(unsigned k=0;k<7;++k)assert(a[k].x%M61==input[k].x%M61 && a[k].y%M61==input[k].y%M61);
  }
  std::cout << "GF61 cosbase=" << AEVUM_PFA7_GF61_COSBASE << " sinbase=" << AEVUM_PFA7_GF61_SINBASE << " nuss8=" << AEVUM_PFA7_NUSS8
            << ": 1200 direct DFT/inverse/roundtrip cases and wide bounds PASS\n";
}
'''
p=(1<<61)-1
w=pow(5,(p-1)//7,p)
prefix=prefix.replace('INVROOTULL',f'{pow(w,6,p)}ULL').replace('ROOTULL',f'{w}ULL').replace('INV7ULL',f'{pow(7,p-2,p)}ULL')
with tempfile.TemporaryDirectory() as tmp:
    cpp,exe=Path(tmp)/'pfa.cpp',Path(tmp)/'pfa'
    cpp.write_text(prefix+actual+suffix)
    # Keep COSBASE ON for both comparisons: this measures ONLY the new
    # sine-base arithmetic rather than implicitly switching to stock/NUSS8.
    for sinbase,nuss in ((0,0),(1,0),(0,1),(1,1)):
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined',
                        '-Wno-unknown-pragmas','-DAEVUM_PFA7_GF61_COSBASE=1',
                        f'-DAEVUM_PFA7_GF61_SINBASE={sinbase}',
                        f'-DAEVUM_PFA7_NUSS8={nuss}',str(cpp),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
