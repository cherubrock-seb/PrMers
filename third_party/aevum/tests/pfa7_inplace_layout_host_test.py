#!/usr/bin/env python3
"""Exercise actual in-place accessors across a complete odd-middle layout.

Models each 16x16 transpose atomically; this is an index/ownership/bounds test,
not a GPU barrier, transform arithmetic, or full-residue test.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
text = (root / "src/cl/middle.cl").read_text()
actual = text.split("#else                           // New implementation (in-place)", 1)[1]
actual = actual.rsplit("#endif", 1)[0]
prefix = r'''
#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
using u32=uint32_t; using i32=int32_t; using u64=uint64_t;
struct T2 { u64 tag=0; }; struct F2 { u64 tag=0; };
#define FFT_FP64 1
#define FFT_FP32 1
#define NTT_GF31 0
#define NTT_GF61 0
#define INPLACE 1
#define WIDTH 512
#define SMALL_HEIGHT 512
#define MIDDLE 7
#define NW 8
#define NH 8
#define G_W (WIDTH/NW)
#define OVERLOAD
#define P(T) T*
#define CP(T) const T*
#define FFTLOAD(p) (*(p))
#define FFTSTORE(p,v) (*(p)=(v))
'''
suffix = r'''
u64 tag(u32 r,u32 x,u32 y) { return 1+(u64(r)*WIDTH+x)*SMALL_HEIGHT+y; }
template<class V> void run() {
  constexpr size_t count=WIDTH*MIDDLE*SMALL_HEIGHT;
  // Match the current conservative PAD_ADJUST allocation (complex values).
  std::vector<V> buf(count*3/2), copy, canonical(count);
  for(u32 line=0;line<MIDDLE*SMALL_HEIGHT;++line)
    for(u32 me=0;me<G_W;++me) {
      V u[NW];
      for(u32 i=0;i<NW;++i)u[i].tag=tag(line/SMALL_HEIGHT,me+i*G_W,line%SMALL_HEIGHT);
      writeCarryFusedLine(u,buf.data(),line,me);
    }
  assert(std::count_if(buf.begin(),buf.end(),[](V v){return v.tag!=0;})==count);
  // Forward middle: all 256 lanes read before the 16x16 transpose/write.
  for(u32 by=0;by<SMALL_HEIGHT;by+=16)for(u32 bx=0;bx<WIDTH;bx+=16) {
    V tile[256][MIDDLE];
    for(u32 lane=0;lane<256;++lane) {
      readMiddleInLine(tile[lane],buf.data(),by+lane/16,bx+lane%16);
      for(u32 r=0;r<MIDDLE;++r)assert(tile[lane][r].tag==tag(r,bx+lane%16,by+lane/16));
    }
    for(u32 lane=0;lane<256;++lane)
      writeMiddleInLine(buf.data(),tile[(lane%16)*16+lane/16],by+lane/16,bx+lane%16);
  }
  // Every PFA special or ordinary tail owns its original line(s). INPLACE=1
  // makes transPos identity, so the ordinary pair cannot overwrite another
  // workgroup's unread row. Model identity arithmetic to isolate the layout.
  std::vector<unsigned> owners(MIDDLE*WIDTH);
  for(u32 r=0;r<MIDDLE;++r) {
    ++owners[r*WIDTH]; ++owners[r*WIDTH+WIDTH/2];
    for(u32 x=1;x<WIDTH/2;++x){++owners[r*WIDTH+x];++owners[r*WIDTH+WIDTH-x];}
  }
  for(auto n:owners)assert(n==1);
  for(u32 line=0;line<MIDDLE*WIDTH;++line)for(u32 me=0;me<SMALL_HEIGHT/NH;++me) {
    V u[NH]; readTailFusedLine(buf.data(),u,line,me);
    for(u32 i=0;i<NH;++i)assert(u[i].tag==tag(line/WIDTH,line%WIDTH,me+i*(SMALL_HEIGHT/NH)));
    writeTailFusedLine(u,buf.data(),line,me);
  }
  // Inverse middle restores the width layout with a second tile transpose.
  for(u32 bx=0;bx<SMALL_HEIGHT;bx+=16)for(u32 by=0;by<WIDTH;by+=16) {
    V tile[256][MIDDLE];
    for(u32 lane=0;lane<256;++lane) {
      readMiddleOutLine(tile[lane],buf.data(),by+lane/16,bx+lane%16);
      for(u32 r=0;r<MIDDLE;++r)assert(tile[lane][r].tag==tag(r,by+lane/16,bx+lane%16));
    }
    for(u32 lane=0;lane<256;++lane)
      writeMiddleOutLine(buf.data(),tile[(lane%16)*16+lane/16],by+lane/16,bx+lane%16);
  }
  for(u32 line=0;line<MIDDLE*SMALL_HEIGHT;++line)for(u32 me=0;me<G_W;++me) {
    V u[NW]; readCarryFusedLine(buf.data(),u,line,me);
    for(u32 i=0;i<NW;++i)assert(u[i].tag==tag(line/SMALL_HEIGHT,me+i*G_W,line%SMALL_HEIGHT));
  }
  std::cout << "PFA7 in-place: " << count << " values, width/middle/tail roundtrip and ownership PASS\n";
}
int main() { run<T2>(); run<F2>(); }
'''
with tempfile.TemporaryDirectory() as tmp:
    cpp, exe = Path(tmp)/"layout.cpp", Path(tmp)/"layout"
    cpp.write_text(prefix+actual+suffix)
    subprocess.run(["g++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
