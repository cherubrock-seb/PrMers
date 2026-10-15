#!/usr/bin/env python3
"""Execute the actual FP32 fft7/fft9 bodies against an independent O(N^2) DFT.

This host floating-point/orientation gate is not GPU residue, ROE, or performance
validation. It uses scalar std::fma, not CUDA/OpenCL vector code generation.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get("CXX") or shutil.which("g++")
if not CXX:
    raise SystemExit("g++ or CXX is required")

HARNESS = r'''
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
using u32 = uint32_t;
struct F2 { float x, y; };
F2 operator+(F2 a, F2 b) { return {a.x+b.x,a.y+b.y}; }
F2 operator-(F2 a, F2 b) { return {a.x-b.x,a.y-b.y}; }
F2 operator-(F2 a) { return {-a.x,-a.y}; }
F2 operator*(F2 a, float b) { return {a.x*b,a.y*b}; }
F2 operator*(float b, F2 a) { return a*b; }
F2& operator+=(F2& a,F2 b) { a=a+b; return a; }
F2& operator-=(F2& a,F2 b) { a=a-b; return a; }
F2 fmaT2(float a,F2 b,F2 c) { return {std::fma(a,b.x,c.x),std::fma(a,b.y,c.y)}; }
F2 mul_t4(F2 a) { return {-a.y,a.x}; }
void x2(F2& a,F2& b) { F2 old=a; a=old+b; b=old-b; }
void x2i(F2& a,F2& b) { x2(a,b); b=mul_t4(b); }
#define fma_addsub(a,b,s,c,d) { F2 tmp=(c)+(s)*(d); (b)=(c)-(s)*(d); (a)=tmp; }
#define X2(a,b) x2(a,b)
#define X2_mul_t4(a,b) x2i(a,b)
#define FFT_FP64 0
#define FFT_FP32 1
#include "fft7.cl"
#include "fft9.cl"
using C = std::complex<long double>;
void run(F2* u,int n) { if(n==7) fft7(u); else fft9(u); }
int main() {
  std::mt19937 rng(6179);
  std::uniform_real_distribution<float> dist(-1,1);
  int cases=0;
  for(int n : {7,9}) for(int sample=0;sample<160;++sample) {
    F2 in[9]{},u[9]{};
    const float scale=std::ldexp(1.0f, sample%3==0 ? 20 : sample%3==1 ? -20 : 0);
    for(int j=0;j<n;++j) {
      if(sample<2*n) {
        in[j] = {sample==2*j ? scale:0, sample==2*j+1 ? scale:0};
      } else if(sample==2*n) {
        in[j]={scale,-scale};
      } else {
        in[j]={dist(rng)*scale,dist(rng)*scale};
      }
      u[j]=in[j];
    }
    run(u,n);
    long double mass=0;
    for(int j=0;j<n;++j) mass+=std::abs(C(in[j].x,in[j].y));
    const long double tol=8e-6L*std::max(mass,1e-30L);
    for(int k=0;k<n;++k) {
      C expected{};
      for(int j=0;j<n;++j) {
        const long double a=2*std::acos(-1.0L)*j*k/n;
        expected+=C(in[j].x,in[j].y)*C(std::cos(a),std::sin(a));
      }
      if(std::abs(C(u[k].x,u[k].y)-expected)>tol)
        throw std::runtime_error("forward DFT/orientation mismatch");
    }
    // Conjugate-forward-conjugate is an independent inverse orientation gate.
    for(int j=0;j<n;++j) u[j].y=-u[j].y;
    run(u,n);
    for(int j=0;j<n;++j)
      if(std::abs(C(u[j].x/n,-u[j].y/n)-C(in[j].x,in[j].y))>tol)
        throw std::runtime_error("inverse round-trip mismatch");
    ++cases;
  }
  std::cout << cases << " source-body DFT/round-trip cases\n";
}
'''
with tempfile.TemporaryDirectory(prefix="pfa-cosbase-") as d:
    tmp = Path(d)
    for name in ("fft7.cl", "fft9.cl"):
        shutil.copyfile(ROOT / "src" / "cl" / name, tmp / name)
    (tmp / "gate.cpp").write_text(HARNESS)
    for flag in (0, 1):
        binary = tmp / ("gate" + str(flag))
        subprocess.run([CXX, "-std=c++17", "-O2",
                        "-DAEVUM_FP32_PFA_COSBASE=" + str(flag),
                        str(tmp / "gate.cpp"), "-o", str(binary)], check=True)
        print("AEVUM_FP32_PFA_COSBASE=" + str(flag), flush=True)
        subprocess.run([str(binary)], check=True)
