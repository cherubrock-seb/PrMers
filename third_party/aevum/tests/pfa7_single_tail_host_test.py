#!/usr/bin/env python3
"""Compare actual single/split tail dispatch and pair-operation routing.

FFT/reversal/arithmetic are recording facades. Exact trace equality validates
row ownership, special flags, table offsets and pair coverage, not GPU maths.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "src/cl/tailsquare.cl").read_text()

def function(name):
    start = source.index(f"KERNEL(G_H) {name}(")
    opening = source.index("{", start)
    # Preprocessor alternatives can contain unmatched braces, so use the
    # unindented function terminator (nested blocks are indented).
    end = source.index("\n}", opening) + 2
    return source[start:end]

prefix = r'''
#include <array>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
using u32=uint32_t; using i32=int32_t; using u64=uint64_t;
struct V { u64 tag=0; };
using T2=V; using GF31=V; using GF61=V;
using Trig=const V*; using TrigGF31=Trig; using TrigGF61=Trig;
#define KERNEL(x) void
#define P(T) T*
#define CP(T) const T*
#define local
#define AEVUM_PRP_TRIG_ARG
#define PFA_RADIX 7
#define WIDTH 512
#define MIDDLE 7
#define SMALL_HEIGHT 512
#define NH 8
#define G_H (SMALL_HEIGHT/NH)
#define ND (WIDTH*MIDDLE*SMALL_HEIGHT)
#define LDS_BYTES (SMALL_HEIGHT*sizeof(V))
#define DISTGF31 0
#define DISTGF61 0
#define DISTHTRIGGF31 0
#define DISTHTRIGGF61 0
#define SINGLE_WIDE 1
#define ZEROHACK_H 0
#define TAIL_TRIGS31 0
#define TAIL_TRIGS61 0
#define TOLOAD(p) (*(p))
const V TAILTGF31{0}, TAILTGF61{0};
u32 group_id=0, local_id=0;
u32 get_group_id(unsigned) { return group_id; }
u32 get_local_id(unsigned) { return local_id; }
void dependentLaunch() {} void dependentLaunchWait() {}
u32 transPos(u32 line,u32 middle,u32 width) { return line/width+line%width*middle; }
using Event=std::array<u64,7>;
std::vector<Event> events;
void readTailFusedLine(const V*, V* u,u32 line,u32 me) {
  assert(line<WIDTH*MIDDLE);
  for(u32 i=0;i<NH;++i)u[i].tag=1+u64(line)*SMALL_HEIGHT+me+i*G_H;
}
void writeTailFusedLine(V*,V*,u32 line,u32 me) {
  assert(line<WIDTH*MIDDLE); events.push_back({1,line,me,0,0,0,0});
}
void fft_HEIGHT1(V*,V*,Trig,u32,u32) {} void fft_HEIGHT2(V*,V*,Trig,u32,u32) {}
void reverse(V*,V*,bool) {} void reverseLine(V*,V*) {}
V cmul(V a,V b) { return V{a.tag^b.tag}; }
void pairSq(u32 n,V* u,V* v,V trig,bool special) {
  events.push_back({0,u[0].tag,v[0].tag,trig.tag,n,u64(special),local_id});
}
'''
suffix = r'''
int main() {
  std::vector<V> trig(SMALL_HEIGHT+(WIDTH/2+1)*G_H);
  for(size_t i=0;i<trig.size();++i)trig[i].tag=i;
  V dummy;
#if !SINGLE_KERNEL
  for(group_id=0;group_id<2*PFA_RADIX;++group_id)
    for(local_id=0;local_id<G_H;++local_id)
      ZERO(&dummy,&dummy,trig.data());
#endif
  for(group_id=0;group_id<PFA_RADIX*(WIDTH/2-(SINGLE_KERNEL?0:1));++group_id)
    for(local_id=0;local_id<G_H;++local_id)
      MAIN(&dummy,&dummy,trig.data());
  std::sort(events.begin(),events.end());
  size_t writes=0,pairs=0,special=0;
  for(const auto& e:events) { writes+=e[0]==1; pairs+=e[0]==0; special+=e[0]==0 && e[5]; }
  assert(writes==PFA_RADIX*WIDTH*G_H);
  assert(pairs==PFA_RADIX*(WIDTH/2+1)*G_H);
  assert(special==PFA_RADIX*G_H);
  // All writes own a different (line,lane), including both self-paired lines.
  for(size_t i=1;i<events.size();++i)if(events[i][0]==1)assert(events[i]!=events[i-1]);
  std::cout.write(reinterpret_cast<const char*>(events.data()),events.size()*sizeof(Event));
}
'''
with tempfile.TemporaryDirectory() as tmp:
    for field in ("GF31", "GF61"):
        cpp, exe = Path(tmp)/"tail.cpp", Path(tmp)/"tail"
        names = (f"tailSquareZero{field}", f"tailSquare{field}")
        cpp.write_text(prefix + "\n".join(map(function, names)) +
                       suffix.replace("ZERO(", names[0]+"(").replace("MAIN(", names[1]+"("))
        traces = []
        for mode in (0, 1):
            subprocess.run(["g++", "-std=c++17", "-O1", "-fsanitize=undefined",
                            f"-DSINGLE_KERNEL={mode}", f"-DAEVUM_PFA7_SINGLE_TAIL={mode}",
                            str(cpp), "-o", str(exe)], check=True)
            traces.append(subprocess.check_output([str(exe)]))
        assert traces[0] == traces[1], f"{field}: split/single pair trace differs"
        print(f"{field}: all 7 rows, 3584 tail lines, exact pair/table/write traces PASS")
