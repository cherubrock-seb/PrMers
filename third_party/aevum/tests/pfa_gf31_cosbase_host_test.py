#!/usr/bin/env python3
"""Compile actual PFA GF31 source; compare both directions with a direct DFT.

Host arithmetic exercises the 0..M31 weak representation and its reduction
bounds. It does not validate GPU codegen, synchronization, or whole residues.
"""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "src/cl/fft-middle.cl").read_text()
start = source.index("GF31 OVERLOAD pfaMulScalar")
end = source.index("\n#endif\n\n#if PFA_RADIX\n// WIDTH", start)
actual = source[start:end]
guards = source[source.index('#ifndef AEVUM_PFA_GF31_COSBASE'):source.index('#ifndef AEVUM_PFA7_NUSS8')]
# All-input bounds for unfolded native PFA7 constants in either direction.
# Reduced sums and differences range independently over 0..M31, a
# conservative superset of the actual paired inputs.
p = (1 << 31) - 1
c, sine = [list(map(int, re.findall(r'(\d+)U', values))) for values in
           re.findall(r'const Z31 (?:c|sine)\[PFA_RADIX\] = \{([^}]+)\};', actual)[:2]]
assert len(c) == len(sine) == 7
for inverse in (False, True):
    for j in (1, 2, 3):
        pc_max = p + p*c[1] + p*sum((c[j*k%7]-c[1])%p for k in (1,2,3) if j*k%7 not in (1,6))
        qs_max = p*sum(sine[(7-j*k%7) if inverse else (j*k%7)] for k in (1,2,3))
        offset = p << 32
        assert pc_max*100 < 37*p*p and qs_max*100 < 178*p*p
        assert qs_max <= offset
        assert pc_max+offset < 0xffffffff7fffffff
        assert pc_max+qs_max < 0xffffffff7fffffff
print('PFA7 all-input unfolded dot and combined reduction bounds PASS')
prefix = r'''
#include <cassert>
#include <cstdint>
#include <random>
#include <iostream>
using u32 = uint32_t; using i32 = int32_t;
using u64 = uint64_t; using Z31 = u32;
constexpr Z31 M31 = 2147483647U;
unsigned reductions32=0,reductions64=0;
struct GF31 { Z31 x, y; };
#define OVERLOAD
#define U2(x,y) GF31{x,y}
#define AEVUM_GF61_LIMB32 0
#define FFT_TYPE 1
#define FFT3161 1
#define WIDTH 512
#define SMALL_HEIGHT 512
#define MIDDLE PFA_RADIX
#define PFA7_ROOT31 894255406U
#define PFA7_INVROOT31 1537170743U
#define PFA7_INV7_31 1840700269U
Z31 modM31(Z31 a) { ++reductions32; assert(a != 0xffffffffU); return (a & M31) + (a >> 31); }
Z31 modM31(i32 a) { ++reductions32; assert(a != INT32_MIN); return (a & M31) + (a >> 31); }
Z31 modM31(u64 a) {
  ++reductions64;
  assert(a < 0xffffffff7fffffffULL);
  return modM31(Z31(a & M31) + Z31((a >> 31) & M31) + Z31(a >> 62));
}
Z31 add(Z31 a, Z31 b) { return modM31(a+b); }
Z31 sub(Z31 a, Z31 b) { return modM31(i32(a-b)); }
GF31 add(GF31 a, GF31 b) { return U2(add(a.x,b.x),add(a.y,b.y)); }
GF31 sub(GF31 a, GF31 b) { return U2(sub(a.x,b.x),sub(a.y,b.y)); }
Z31 mul(Z31 a, Z31 b) { return modM31(u64(a)*b); }
u64 power(u64 a, u64 b) { u64 v=1; for(;b;b>>=1,a=a*a%M31) if(b&1)v=v*a%M31; return v; }
'''
suffix = r'''
int main() {
  const u64 root=power(5,(M31-1)/PFA_RADIX), inv=power(root,PFA_RADIX-1);
  if(PFA_RADIX==7) { assert(root==PFA7_ROOT31); assert(inv==PFA7_INVROOT31); }
  std::mt19937 gen(316179);
  for(unsigned trial=0; trial<1400; ++trial) {
    GF31 a[PFA_RADIX], b[PFA_RADIX];
    for(unsigned k=0;k<PFA_RADIX;++k) {
      if(trial<(1u<<PFA_RADIX)) a[k]=U2(trial&(1u<<k)?M31:0,trial&(1u<<k)?0:M31);
      else if(trial<(1u<<PFA_RADIX)+4*PFA_RADIX) {
        const u32 v[] = {1,M31-1,M31/2,M31};
        const u32 index=trial-(1u<<PFA_RADIX);
        a[k]=U2(k==index%PFA_RADIX?v[index/PFA_RADIX]:0,M31);
      } else if(trial<600) a[k]=U2((k+trial)%2?M31:1,(k+trial)%3?M31-1:0);
      else a[k]=U2(Z31(gen()%(u64(M31)+1)),Z31(gen()%(u64(M31)+1)));
    }
    for(unsigned inverse=0;inverse<2;++inverse) {
      for(unsigned k=0;k<PFA_RADIX;++k)b[k]=a[k];
      reductions32=reductions64=0;
      if(inverse)pfaInverseMiddle(b); else pfaForwardMiddle(b);
      if(PFA_RADIX==7 && AEVUM_PFA_GF31_COSBASE && !inverse) {
        assert(reductions64==12);
        assert(reductions32==(AEVUM_PFA7_GF31_QUICK?30u:42u));
      }
      for(unsigned j=0;j<PFA_RADIX;++j) {
        u64 x=0,y=0;
        for(unsigned k=0;k<PFA_RADIX;++k) {
          const u64 w=power(inverse?inv:root,j*k);
          x=(x+u64(a[k].x)*w)%M31; y=(y+u64(a[k].y)*w)%M31;
        }
        if(inverse) { const u64 n=power(PFA_RADIX,M31-2); x=x*n%M31; y=y*n%M31; }
        assert(b[j].x<=M31 && b[j].y<=M31);
        assert(b[j].x%M31==x && b[j].y%M31==y);
      }
    }
    for(unsigned k=0;k<PFA_RADIX;++k)b[k]=a[k];
    pfaForwardMiddle(b); pfaInverseMiddle(b);
    for(unsigned k=0;k<PFA_RADIX;++k)assert(b[k].x%M31==a[k].x%M31 && b[k].y%M31==a[k].y%M31);
  }
  std::cout << "radix=" << PFA_RADIX << " cosbase=" << AEVUM_PFA_GF31_COSBASE
            << " nuss8=" << AEVUM_PFA7_NUSS8 << " quick=" << AEVUM_PFA7_GF31_QUICK
            << " PASS: 1400 forward/inverse/roundtrip cases\n";
}
'''
# The root constants are checked independently against the host generator.
p = (1 << 31) - 1
w = pow(5, (p-1)//7, p)
prefix = prefix.replace("894255406U", f"{w}U").replace("1537170743U", f"{pow(w,6,p)}U")
with tempfile.TemporaryDirectory() as tmp:
    cpp, exe = Path(tmp)/"test.cpp", Path(tmp)/"test"
    cpp.write_text(prefix + guards + actual + suffix)
    for radix in (7, 9):
        cases = [(0,0,0), (1,0,0), (1,1,0)]
        if radix == 7:
            cases += [(1,0,1), (1,1,1)]
        for mode, nuss, quick in cases:
            subprocess.run(["g++", "-std=c++17", "-O2", "-fsanitize=undefined", "-Wno-unknown-pragmas",
                            f"-DPFA_RADIX={radix}", f"-DAEVUM_PFA_GF31_COSBASE={mode}",
                            f"-DAEVUM_PFA7_GF31_QUICK={quick}",
                            f"-DAEVUM_PFA7_NUSS8={nuss}", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
    for radix, cosbase, quick in ((7,1,2), (9,1,1), (7,0,1)):
        invalid = subprocess.run(["g++", "-std=c++17", "-fsyntax-only",
            f"-DPFA_RADIX={radix}", f"-DAEVUM_PFA_GF31_COSBASE={cosbase}",
            f"-DAEVUM_PFA7_GF31_QUICK={quick}", "-DAEVUM_PFA7_NUSS8=0", str(cpp)], capture_output=True, text=True)
        assert invalid.returncode and '#error AEVUM_PFA7_GF31_QUICK' in invalid.stderr
    print('Kernel flag/scope rejection PASS')
