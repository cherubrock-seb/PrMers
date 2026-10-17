#!/usr/bin/env python3
"""Actual host coefficients + actual FP32 kernel body, with float FMA semantics.

Exhausts the reduced interval for both admitted FFT sizes and checks quadrant
boundaries/random full-circle samples. This is not a native GPU/ROE gate.
"""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "src/cl/trig.cl").read_text()
source = source[source.index("#if FFT_FP32") + len("#if FFT_FP32"):]
source = source[:source.rindex("#endif")]
# Only adapt the constant-table declaration to a runtime test fixture.
source = source.replace("const float S[] = TRIG_SIN;", "const auto& S = sinCoefs;")
source = source.replace("const float C[] = TRIG_COS;", "const auto& C = cosCoefs;")
prefix = r'''
#include "Trig.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
void log(const char*, ...) { std::abort(); }
using std::fma;
struct F2 { float x,y; };
F2 operator-(F2 a) { return {-a.x,-a.y}; }
#define U2(x,y) F2{x,y}
#define SWAP_XY(a) F2{a.y,a.x}
#define OVERLOAD
u32 fftSize;
#define ND fftSize
std::array<float,8> sinCoefs,cosCoefs;
float scale;
#define TRIG_SCALE scale
void install(const TrigCoefsFP32& c) {
  scale=float(c.scale);
  for (unsigned i=0;i<8;++i) { sinCoefs[i]=float(c.sinCoefs[i]); cosCoefs[i]=float(c.cosCoefs[i]); }
}
'''
checks = r'''
unsigned ulps(float a, float b) {
  if (a==b) return 0; // includes signed zero
  assert(std::signbit(a)==std::signbit(b));
  const auto aa=std::bit_cast<uint32_t>(std::abs(a)),bb=std::bit_cast<uint32_t>(std::abs(b));
  return aa>bb?aa-bb:bb-aa;
}
int main() {
  uint64_t cases=0;
  unsigned maxSin=0,maxCos=0,maxFancy=0,maxFull=0;
  std::mt19937 random(0x2ba51340);
  for (u32 n : {0u,1u,2u,6u}) {
    bool caught=false; try{trigCoefsFP32Minimax(n);}catch(const std::invalid_argument&){caught=true;}
    assert(caught);
  }
  for (u32 middle:{8u,16u}) {
    fftSize=512*middle*512;
    const auto c=trigCoefsFP32Minimax(fftSize/2);
    install(c);
    int exponent;
    assert(std::frexp(scale,&exponent)==0.5f);
    assert(sinCoefs[1]!=0);
    for (unsigned i=0;i<5;++i) {
      assert(std::isnormal(sinCoefs[i])); assert(std::isnormal(cosCoefs[i]));
    }
    for (unsigned i=5;i<8;++i) assert(sinCoefs[i]==0 && cosCoefs[i]==0);
    for (u32 k=0;k<=fftSize/8;++k) {
      assert(double(float(k)*scale)==double(k)*double(scale));
      const long double angle=2*std::numbers::pi_v<long double>*k/fftSize;
      const float sine=float(std::sin(angle)),cosine=float(std::cos(angle));
      // Stable reference for cos(a)-1, including the smallest nonzero angle.
      const float fancy=float(-2*std::sin(angle/2)*std::sin(angle/2));
      const auto r=candidate::reducedCosSin(k,1),f=candidate::fancyTrig_N(k);
      maxSin=std::max(maxSin,ulps(r.y,sine));
      maxCos=std::max(maxCos,ulps(r.x,cosine));
      maxFancy=std::max(maxFancy,ulps(f.x,fancy));
      assert(ulps(r.y,sine)<=3 && ulps(r.x,cosine)<=3 && ulps(f.x,fancy)<=3);
      ++cases;
    }
    auto full=[&](u32 k,u32 bound) {
      const auto r=candidate::slowTrig_N(k,bound);
      const u32 reduced=k%fftSize;
      const long double angle=2*std::numbers::pi_v<long double>*reduced/fftSize;
      float sine=float(std::sin(angle)),cosine=float(std::cos(angle));
      if (reduced%(fftSize/4)==0) {
        constexpr float ss[]={0,1,0,-1},cc[]={1,0,-1,0};
        sine=ss[reduced/(fftSize/4)]; cosine=cc[reduced/(fftSize/4)];
      }
      const unsigned error=std::max(ulps(r.y,sine),ulps(r.x,cosine));
      maxFull=std::max(maxFull,error); assert(error<=3); ++cases;
    };
    for (u32 octant=0;octant<=16;++octant) for (int delta=-2;delta<=2;++delta) {
      const int64_t k=int64_t(octant)*(fftSize/8)+delta;
      if (k>=0 && k<2*fftSize) full(u32(k),2*fftSize);
    }
    for (u32 i=0;i<20000;++i) full(random()%(2*fftSize),2*fftSize);
    for (u32 bound:{fftSize/8+1,fftSize/4,fftSize/2,fftSize}) {
      full(0,bound); full(bound-1,bound);
    }
    // Both compiled paths execute; OFF still consumes legacy coefficients.
    install(trigCoefsFP32(fftSize/2));
    for (u32 k:{0u,1u,fftSize/16,fftSize/8}) {
      const auto r=stock::reducedCosSin(k,1);
      assert(std::isfinite(r.x) && std::isfinite(r.y));
    }
  }
  std::cout<<cases<<" actual-kernel FP32 samples: PASS; max ULP sine/cosine/fancy/full = "
           <<maxSin<<'/'<<maxCos<<'/'<<maxFancy<<'/'<<maxFull<<'\n';
}
'''
with tempfile.TemporaryDirectory(prefix="fp32-minimax-") as tmp:
    cpp = Path(tmp) / "test.cpp"
    cpp.write_text(prefix + "\n#define AEVUM_FP32_MINIMAX_TRIG 0\nnamespace stock {\n" + source +
                   "\n}\n#undef AEVUM_FP32_MINIMAX_TRIG\n#define AEVUM_FP32_MINIMAX_TRIG 1\nnamespace candidate {\n" + source + "\n}\n" + checks)
    out = Path(tmp) / "test"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-ffp-contract=off", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    "-I"+str(root/"src"), str(cpp), str(root/"src/Trig.cpp"), "-o", str(out)], check=True)
    subprocess.run([str(out)], check=True)
