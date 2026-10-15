#!/usr/bin/env python3
"""Host algebra/orientation gate for the actual PFA5/15 source helpers.

The GF61 primitive wrappers use canonical host arithmetic. This is not a GPU
weak-range, memory, carry, ROE, capacity, or end-to-end validation.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get("CXX") or shutil.which("g++")
if not CXX:
    raise SystemExit("g++ or CXX required")

def function(source, signature):
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]

middle = (ROOT / "src/cl/fft-middle.cl").read_text()
gf = middle[middle.index("#if NTT_GF61"):]
functions = "\n".join(function(gf, signature) for signature in (
    "void OVERLOAD pfaDft3(GF61",
    "void aevumPfaDft5(",
    "void aevumPfaDft515(",
    "void OVERLOAD pfaForwardMiddle(GF61",
    "void OVERLOAD pfaInverseMiddle(GF61",
))
pack = function((ROOT / "src/cl/fftp.cl").read_text(),
                "static inline u32 pfaLogicalIndex(")
HARNESS = r'''
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using u32 = uint32_t; using Z61 = uint64_t;
constexpr Z61 M61 = (Z61(1)<<61)-1;
struct GF61 { Z61 x,y; };
Z61 mul(Z61 a,Z61 b) { return (unsigned __int128)(a%M61)*(b%M61)%M61; }
GF61 add(GF61 a,GF61 b) { return {(a.x%M61+b.x%M61)%M61,(a.y%M61+b.y%M61)%M61}; }
GF61 sub(GF61 a,GF61 b) { return {(a.x%M61+M61-b.x%M61)%M61,(a.y%M61+M61-b.y%M61)%M61}; }
GF61 pfaMulScalar(GF61 a,Z61 b) { return {mul(a.x,b),mul(a.y,b)}; }
bool equal(GF61 a,GF61 b) { return a.x%M61==b.x%M61 && a.y%M61==b.y%M61; }
Z61 power(Z61 a,Z61 e) { Z61 r=1;for(;e;e>>=1,a=mul(a,a))if(e&1)r=mul(r,a);return r; }
struct F2 {float x,y;};
F2 operator+(F2 a,F2 b){return{a.x+b.x,a.y+b.y};}
F2 operator-(F2 a,F2 b){return{a.x-b.x,a.y-b.y};}
F2 operator-(F2 a){return{-a.x,-a.y};}
F2 operator*(F2 a,float b){return{a.x*b,a.y*b};}
F2 operator*(float b,F2 a){return a*b;}
F2 fmaT2(float a,F2 b,F2 c){return{std::fma(a,b.x,c.x),std::fma(a,b.y,c.y)};}
F2 mul_t4(F2 a){return{-a.y,a.x};}
void x2i(F2& a,F2& b){F2 old=a;a=old+b;b=mul_t4(old-b);}
#define X2_mul_t4(a,b) x2i(a,b)
#define fma_addsub(a,b,s,c,d) {F2 tmp=(c)+(s)*(d);(b)=(c)-(s)*(d);(a)=tmp;}
#define SWAP(a,b) std::swap(a,b)
#define OVERLOAD
#define FFT_FP64 0
#define FFT_FP32 1
#include "fft15.cl"
'''
CHECK = r'''
int main(){
  std::mt19937_64 rng(515);
  const int r=PFA_RADIX;
  const Z61 root=power(5,(M61-1)/r);
  for(int sample=0;sample<160;++sample){
    GF61 in[15]{},u[15]{};
    F2 fin[15]{},fu[15]{};
    for(int j=0;j<r;++j){
      if(sample<2*r){
        in[j]={sample==2*j ? 1u:0u,sample==2*j+1 ? 1u:0u};
        fin[j]={float(in[j].x),float(in[j].y)};
      }else{
        in[j]={rng()%(M61+8),rng()%(M61+8)};
        if(sample==2*r)in[j]={M61,M61+7};
        const float scale=std::ldexp(1.0f,sample%3==0?20:sample%3==1?-20:0);
        fin[j]={float(int(rng()%2001)-1000)*scale,float(int(rng()%2001)-1000)*scale};
      }
      u[j]=in[j];fu[j]=fin[j];
    }
    pfaForwardMiddle(u);
    if(r==5)fft5(fu);else fft15(fu);
    long double mass=0;
    for(int j=0;j<r;++j)mass+=std::abs(std::complex<long double>(fin[j].x,fin[j].y));
    for(int k=0;k<r;++k){
      GF61 expected{};
      std::complex<long double> fexpected{};
      for(int j=0;j<r;++j){
        expected=add(expected,pfaMulScalar(in[j],power(root,j*k)));
        const auto angle=2*std::acos(-1.0L)*j*k/r;
        fexpected+=std::complex<long double>(fin[j].x,fin[j].y)*
                   std::complex<long double>(std::cos(angle),std::sin(angle));
      }
      if(!equal(u[k],expected))throw std::runtime_error("GF61 DFT/orientation mismatch");
      if(std::abs(std::complex<long double>(fu[k].x,fu[k].y)-fexpected)>
         1.2e-5L*std::max(mass,1e-30L))
        throw std::runtime_error("FP32 DFT/orientation mismatch");
    }
    pfaInverseMiddle(u);
    for(int j=0;j<r;++j)if(!equal(u[j],in[j]))throw std::runtime_error("GF61 inverse normalization mismatch");
  }
  for(u32 row=0;row<PFA_RADIX;++row)
    for(u32 b : {0u,1u,2u,255u,256u,511u,65535u,PFA_BINARY_LENGTH-2,PFA_BINARY_LENGTH-1}){
      const auto n=pfaLogicalIndex(row,b);
      if(n>=PFA_RADIX*PFA_BINARY_LENGTH || n%PFA_RADIX!=row || n%PFA_BINARY_LENGTH!=b)
        throw std::runtime_error("CRT digit map mismatch");
    }
  std::cout<<"radix "<<r<<": 160 GF61/FP32 source DFT cases, inverse and CRT mapping checked\n";
}
'''
with tempfile.TemporaryDirectory(prefix="pfa515-host-") as d:
    temp = Path(d)
    for name in ("fft3.cl", "fft5.cl", "fft15.cl"):
        shutil.copyfile(ROOT / "src/cl" / name, temp / name)
    prime = (1 << 61) - 1
    for radix in (5, 15):
        root = pow(5, (prime-1)//radix, prime)
        w5 = pow(root, radix//5, prime)
        half = pow(2, -1, prime)
        coef = {}
        for k in (1, 2):
            w, wi = pow(w5, k, prime), pow(w5, -k, prime)
            coef["PFA515_C%d_61" % k] = (w+wi)*half % prime
            coef["PFA515_S%d_61" % k] = (w-wi)*half % prime
        coef["PFA515_INV_RADIX61"] = pow(radix, -1, prime)
        coef["PFA515_ROOT3_61"] = pow(root, radix//3, prime) if radix == 15 else 1
        coef["PFA515_INVROOT3_61"] = pow(coef["PFA515_ROOT3_61"], -1, prime)
        defs = "#define PFA_RADIX %d\n#define MIDDLE %d\n" % (radix, radix)
        defs += "#define PFA_BINARY_LENGTH 131072u\n"
        defs += "#define PFA_L_INV %du\n" % pow(131072, -1, radix)
        defs += "".join("#define %s %dULL\n" % (k, v) for k, v in coef.items())
        src = temp / "gate.cpp"
        src.write_text(defs + HARNESS + functions + "\n" + pack + CHECK)
        binary = temp / ("gate%d" % radix)
        subprocess.run([CXX, "-std=c++17", "-O2", "-Wno-unknown-pragmas",
                        str(src), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
