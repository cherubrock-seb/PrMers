#!/usr/bin/env python3
"""Compile actual Gpu selection/square bodies against a recording host facade.

Checks configuration rejection and graph scope on the EngineApi regSquare path.
This does not compile CUDA kernels or establish GPU exactness/performance.
"""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
src = (root / "src/Gpu.cpp").read_text()
queue = (root / "src/Queue.h").read_text()
gate = src[src.index("  cuda_pfa7_graphs = false;"):]
gate = gate[:gate.index('  config.erase("AEVUM_CUDA_PFA7_GRAPHS");') + len('  config.erase("AEVUM_CUDA_PFA7_GRAPHS");')]
square = src[src.index("void Gpu::square("):src.index("u32 Gpu::squareLoop(")]
reg = src[src.index("void Gpu::regSquare("):src.index("void Gpu::regPrepare(")]
scope = queue[queue.index("  class GraphSquareScope {"):]
scope = scope[:scope.index("\n#endif")]

prefix = r'''
#include <cassert>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using u32 = unsigned;
using Word = int;
using namespace std;
enum { FFT3161 = 1, FFT3261 = 2, CARRY_32 = 32, CARRY_64 = 64 };
struct Shape {
  int fft_type=FFT3161, width=512, middle=7, height=512;
  unsigned long long size() const { return 2ull*width*middle*height; }
};
struct FFT {
  Shape shape;
  int pfa_radix=7, variant=202, carry=CARRY_32;
  bool isPfa() const { return pfa_radix != 0; }
};
struct Args {
  int carry=CARRY_32, multi=0;
  bool profile=false;
  int value(const char*, int fallback) const { return multi ? multi : fallback; }
};
bool isNvidiaGpu(int id) { return id != 0; }
u32 getNvidiaComputeCapability(int id) { return unsigned(id); }
void log(const char*, ...) {}
bool select(map<string,string>& config, Args args={}, FFT fft={}, [[maybe_unused]] int id=806,
            unsigned long long E=136279841) {
  bool cuda_pfa7_graphs=true, doLog=false;
'''
facade = r'''
  return cuda_pfa7_graphs;
}
template<class T> struct Buffer {};
enum LEAD_TYPE { LEAD_NONE, LEAD_WIDTH, LEAD_MIDDLE };
struct Queue {
  vector<string> trace;
  bool active=false;
  void beginCudaGraphSquare() { assert(!active); active=true; trace.push_back("begin"); }
  void endCudaGraphSquare() { assert(active); active=false; trace.push_back("launch"); }
  void abortCudaGraphSquare() noexcept { active=false; trace.push_back("abort"); }
'''
gpu = r'''
};
struct Gpu {
  bool cuda_graph_squares_=false, cuda_pfa7_graphs_=false;
  bool prpMiddle1=false, useLongCarry=false, fail=false;
  Queue queue;
  Args args;
  FFT fft;
  Buffer<double> buf1, buf3;
  vector<int> recorded_kernels;
  vector<Buffer<double>*> recorded_kernel_args;
  static constexpr int KPRPMIDDLE1=1;
  void fftP(Buffer<double>&, Buffer<Word>&) { queue.trace.push_back("fftP"); }
  void fftMidIn(Buffer<double>&) { queue.trace.push_back("middleIn"); }
  void tailSquare(Buffer<double>&) { queue.trace.push_back("tail"); }
  void fftMidOut(Buffer<double>&) { queue.trace.push_back("middleOut"); }
  void fftW(Buffer<double>&, Buffer<double>&) {
    queue.trace.push_back("fftW");
    if(fail) throw runtime_error("injected launch failure");
  }
  void carryA(Buffer<Word>&, Buffer<double>&) { queue.trace.push_back("carryA"); }
  void carryLL(Buffer<Word>&, Buffer<double>&) { queue.trace.push_back("carryLL"); }
  void carryM(Buffer<Word>&, Buffer<double>&) { queue.trace.push_back("carryM"); }
  void carryB(Buffer<Word>&) { queue.trace.push_back("carryB"); }
  void fftPCarryB(Buffer<double>&, Buffer<Word>&) { queue.trace.push_back("fftPCarryB"); }
  void carryFusedLL(Buffer<double>&) { queue.trace.push_back("carryFusedLL"); }
  void carryFused(Buffer<double>&) { queue.trace.push_back("carryFused"); }
  void square(Buffer<Word>&, Buffer<Word>&, LEAD_TYPE, LEAD_TYPE, bool=false, bool=false, bool=false);
  void regSquare(Buffer<Word>&, u32=1);
};
'''
checks = r'''
template<class F> void rejects(F f) {
  bool rejected=false;
  try { f(); } catch(const runtime_error&) { rejected=true; }
  assert(rejected);
}
int main() {
  unsetenv("AEVUM_CUDA_PFA7_GRAPHS"); unsetenv("PRPLL_PROFILE");
  const string key="AEVUM_CUDA_PFA7_GRAPHS";
  map<string,string> config;
  assert(!select(config) && !config.count(key));
  config[key]="0"; assert(!select(config) && !config.count(key));
  for(auto v:{"", "2", "-1", "true", "01"}) {
    config[key]=v; rejects([&]{select(config);});
  }
  config[key]="0";
  setenv(key.c_str(),"1",1); rejects([&]{select(config);});
  unsetenv(key.c_str());
  auto on = [&](Args a=Args{}, FFT f=FFT{}, int id=806) {
    map<string,string> c{{key,"1"}};
    bool value=select(c,a,f,id);
    assert(!c.count(key));
    return value;
  };
#ifdef CUDA_BACKEND
  assert(on()); assert(on(Args{},FFT{},705)); assert(on(Args{},FFT{},809));
  rejects([&]{on(Args{},FFT{},704);}); rejects([&]{on(Args{},FFT{},0);});
  setenv(key.c_str(),"1",1); config.clear(); assert(select(config));
  unsetenv(key.c_str());
  Args a; a.profile=true; rejects([&]{on(a);});
  a={}; a.multi=1; rejects([&]{on(a);});
  a={}; a.carry=CARRY_64; rejects([&]{on(a);});
  setenv("PRPLL_PROFILE","0",1); rejects([&]{on();}); unsetenv("PRPLL_PROFILE");
  for(auto v:{"1","2","-1"}) {
    config={{key,"1"},{"MULTI_Q",v}}; rejects([&]{select(config);});
  }
  config={{key,"1"},{"MULTI_Q","0"}}; assert(select(config));
  config={{key,"1"},{"CARRY64","1"}}; rejects([&]{select(config);});
  config={{key,"1"},{"TAIL_KERNELS","2"}}; rejects([&]{select(config);});
  config={{key,"1"}}; rejects([&]{select(config,Args{},FFT{},806,35000000);});
  for(int dimension=0; dimension<7; ++dimension) {
    FFT f;
    switch(dimension) {
      case 0:f.shape.fft_type=FFT3261;break;
      case 1:f.shape.width=1024;break;
      case 2:f.shape.height=256;break;
      case 3:f.shape.middle=9;break;
      case 4:f.pfa_radix=0;break;
      case 5:f.variant=101;break;
      case 6:f.carry=CARRY_64;break;
    }
    rejects([&]{on(Args{},f);});
  }
#else
  rejects([&]{on();});
#endif
  const vector<string> canonical{"fftP","middleIn","tail","middleOut","fftW","carryA","carryB"};
  Gpu g;
  Buffer<Word> word;
  g.regSquare(word); assert(g.queue.trace==canonical);
  g.cuda_pfa7_graphs_=true;
  for(int repeat=0; repeat<2; ++repeat) {
    g.queue.trace.clear(); g.regSquare(word);
#ifdef CUDA_BACKEND
    auto expected=canonical;
    expected.insert(expected.begin(),"begin"); expected.push_back("launch");
    assert(g.queue.trace==expected && !g.queue.active);
#else
    assert(g.queue.trace==canonical);
#endif
  }
  // The real regSquare call leaves prp=false: this path must still batch.
  for(bool ll:{false,true}) {
    g.queue.trace.clear();
    g.square(word,word,LEAD_NONE,LEAD_NONE,!ll,ll,true);
    auto expected=canonical; expected[5]=ll?"carryLL":"carryM";
    assert(g.queue.trace==expected);
  }
  g.queue.trace.clear();
  g.square(word,word,LEAD_WIDTH,LEAD_WIDTH,false,false,true);
  assert(g.queue.trace.front()=="middleIn" && g.queue.trace.back()=="fftP");
  g.cuda_pfa7_graphs_=false; g.cuda_graph_squares_=true; g.fft.pfa_radix=0;
  g.queue.trace.clear(); g.square(word,word,LEAD_WIDTH,LEAD_WIDTH,false,false,true);
#ifdef CUDA_BACKEND
  assert(g.queue.trace.front()=="begin" && g.queue.trace.back()=="launch");
#endif
  g.cuda_graph_squares_=false; g.cuda_pfa7_graphs_=true; g.fail=true;
  g.queue.trace.clear(); rejects([&]{g.regSquare(word);});
#ifdef CUDA_BACKEND
  assert(g.queue.trace.back()=="abort" && !g.queue.active);
#endif
  cout << "PFA7 graph selection and actual regSquare/square scope: PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix="pfa7-graph-host-") as tmp:
    cpp = Path(tmp) / "gate.cpp"
    cpp.write_text(prefix + gate + facade + scope + gpu + square + reg + checks)
    for backend in (False, True):
        out = Path(tmp) / ("cuda" if backend else "opencl")
        command = [os.environ.get("CXX", "g++"), "-std=c++20", "-Wall", "-Wextra",
                   "-fsanitize=undefined", "-fno-sanitize-recover=all"]
        if backend:
            command.append("-DCUDA_BACKEND")
        subprocess.run(command + [str(cpp), "-o", str(out)], check=True)
        subprocess.run([str(out)], check=True)
