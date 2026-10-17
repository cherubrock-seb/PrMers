#!/usr/bin/env python3
"""Exercise the actual cache/load control and PTX helper without a CUDA SDK.
The recording driver tests isolation/failure ordering, not GPU code or timing.
"""
from pathlib import Path
import os
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
compiler = (root / "src/KernelCompiler.cpp").read_text()
backend = (root / "src/cuda/clwrap_cuda.cpp").read_text()
load = compiler[compiler.index("KernelHolder KernelCompiler::loadAux("):compiler.index("std::future<KernelHolder> KernelCompiler::load(")]
jit = backend[backend.index("bool cudaLoadRegLimitedKernel("):backend.index("void cudaGraphBatchBegin(")]
prefix = r'''
#define CUDA_BACKEND
#include "PtxRegisterCache.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>
#include <iostream>
using namespace std;
using CUmodule = unsigned;
using CUjit_option = int;
using CUresult = int;
constexpr int CUDA_SUCCESS=0, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES=1, CU_JIT_ERROR_LOG_BUFFER=2;
const string neutral=R"ptx(.version 8.6
.target sm_86
.address_size 64
// .entry target() .maxnreg 1 { }
.file 1 ".entry target() .maxntid 1 { }"
.visible .entry target_extra(.param .u64 in)
.maxntid 128, 1, 1
{ ret; }
.visible .entry target (
  .param .u64 in
)
// do not touch the workgroup bound
.maxntid 64, 1, 1
{ ret; }
)ptx";
struct _cl_program {
  string ptx=neutral;
  CUmodule module=0;
  bool moduleLoaded=true, kernelExported=false;
};
using cl_program=_cl_program*;
using Program=unique_ptr<_cl_program>;
struct Kernel { unsigned module; };
using KernelHolder=unique_ptr<Kernel>;
unsigned moduleSequence=0;
int compiles=0, saves=0, cacheHits=0, loads=0, unloads=0, kernels=0;
bool failJit=false;
vector<string> compiledArgs, loadedPtx;
map<string,string> cache;
void ensureContextCurrent() {}
CUresult cuModuleLoadDataEx(CUmodule* out, const char* ptx, unsigned, CUjit_option*, void**) {
  ++loads;
  if(failJit) return 999;
  *out=++moduleSequence; loadedPtx.push_back(ptx); return CUDA_SUCCESS;
}
CUresult cuModuleUnload(CUmodule) { ++unloads; return CUDA_SUCCESS; }
void log(const char*, ...) {}
struct Timer { double at() const {return 0;} };
struct SHA3 {
  static array<size_t,1> hash(size_t ctx,const string& f,const string& k,const string& args) {
    return {std::hash<string>{}(to_string(ctx)+f+k+args)};
  }
};
string to_hex(size_t value) { return to_string(value); }
Program loadBinary(int,int,const string& file) {
  auto it=cache.find(file); if(it==cache.end()) return {};
  ++cacheHits;
  auto p=make_unique<_cl_program>(); p->ptx=it->second; p->module=++moduleSequence; return p;
}
void saveBinary(cl_program p,const string& file) {
  assert(p->ptx==neutral); cache[file]=p->ptx; ++saves;
}
Kernel* loadKernel(cl_program p,const char*) {
  ++kernels; p->kernelExported=true; return new Kernel{p->module};
}
struct KernelCompiler {
  int context=0,deviceId=0,verbose=0;
  size_t contextHash=17;
  bool useCache=true;
  string cacheDir="cache";
  Program compile(const string&,const string& args) const {
    ++compiles; compiledArgs.push_back(args);
    auto p=make_unique<_cl_program>(); p->module=++moduleSequence; return p;
  }
  KernelHolder loadAux(const string&,const string&,const string&) const;
};
'''
checks = r'''
template<class F> void rejects(F f) {
  bool rejected=false; try { f(); } catch(const runtime_error&) { rejected=true; }
  assert(rejected);
}
int main() {
  const string marker="--aevum-reg-ptx=1 ";
  const string off="  -DFIELD=61  --maxrregcount=80  ";
  auto unchanged=aevumTakePtxRegisterLimit(off);
  assert(!unchanged.enabled && unchanged.compileArgs==off);
  auto a=aevumTakePtxRegisterLimit(marker+"-DFIELD=61 --maxrregcount=80");
  auto b=aevumTakePtxRegisterLimit(marker+"-DFIELD=61 -DCUDA_MIN_BLOCKS=2");
  assert(a.maxRegs==80 && !a.minBlocks && b.minBlocks==2 && !b.maxRegs);
  assert(a.compileArgs==b.compileArgs && a.compileArgs!=off);
  for(auto bad:{"--maxrregcount=0","--maxrregcount=-1","--maxrregcount=256", "--maxrregcount=8x",
                "--maxrregcount=999999999999999999999", "--maxrregcount", "--maxrregcount=",
                "-DCUDA_MIN_BLOCKS=0", "-DCUDA_MIN_BLOCKS=17", "-DCUDA_MIN_BLOCKS=2u",
                "--maxrregcount=80 -DCUDA_MIN_BLOCKS=2", "--aevum-reg-ptx=1 --maxrregcount=80", ""})
    rejects([&]{aevumTakePtxRegisterLimit(marker+bad);});
  rejects([&]{aevumTakePtxRegisterLimit("--aevum-reg-ptx=0 --maxrregcount=80");});
  string limited=aevumPtxWithRegisterLimit(neutral,"target",80,0);
  auto target=neutral.find(".visible .entry target (");
  assert(limited.substr(0,target)==neutral.substr(0,target));
  assert(limited.find(".maxnreg 80\n.maxntid 64, 1, 1")!=string::npos);
  assert(aevumPtxWithRegisterLimit(neutral,"target",0,2).find(".minnctapersm 2\n.maxntid 64, 1, 1")!=string::npos);
  rejects([&]{aevumPtxWithRegisterLimit(neutral,"missing",80,0);});
  rejects([&]{aevumPtxWithRegisterLimit(neutral,"target",80,2);});
  rejects([&]{aevumPtxWithRegisterLimit(limited,"target",80,0);});
  rejects([&]{aevumPtxWithRegisterLimit(".entry target() { ret; }","target",80,0);});
  rejects([&]{aevumPtxWithRegisterLimit(".entry target()\n.entry other() .maxntid 64 { ret; }","target",80,0);});
  rejects([&]{aevumPtxWithRegisterLimit("/* unfinished","target",80,0);});
  rejects([&]{aevumPtxWithRegisterLimit(neutral+".entry target() .maxntid 64 { ret; }","target",80,0);});
  KernelCompiler compiler;
  vector<unsigned> modules;
  for(auto limit:{"--maxrregcount=64","--maxrregcount=80","-DCUDA_MIN_BLOCKS=2","-DCUDA_MIN_BLOCKS=3"}) {
    auto k=compiler.loadAux("tailsquare.cl","target",marker+"-DFIELD=61 "+limit);
    modules.push_back(k->module);
  }
  assert(compiles==1 && saves==1 && cacheHits==3 && kernels==4 && loads==4 && unloads==4);
  assert(compiledArgs[0]==a.compileArgs && cache.begin()->second==neutral);
  assert(modules[0]!=modules[1]);
  assert(loadedPtx[0].find(".maxnreg 64")!=string::npos);
  assert(loadedPtx[1].find(".maxnreg 80")!=string::npos);
  assert(loadedPtx[2].find(".minnctapersm 2")!=string::npos);
  assert(loadedPtx[3].find(".minnctapersm 3")!=string::npos);
  // Failed JIT cannot expose a baseline kernel or unload its still-live module.
  failJit=true;
  rejects([&]{compiler.loadAux("tailsquare.cl","target",marker+"-DFIELD=61 --maxrregcount=96");});
  assert(compiles==1 && cacheHits==4 && kernels==4 && unloads==4 && saves==1);
  failJit=false;
  auto k=compiler.loadAux("tailsquare.cl","target",off);
  assert(compiles==2 && saves==2 && kernels==5 && loads==5 && unloads==4);
  assert(compiledArgs.back()==off); // OFF follows exactly the original compiler path.
  _cl_program exported; exported.kernelExported=true;
  rejects([&]{cudaLoadRegLimitedKernel(&exported,"target",64,0);});
  assert(loads==5 && unloads==4);
  _cl_program p; p.module=999;
  failJit=true; assert(!cudaLoadRegLimitedKernel(&p,"target",80,0));
  assert(p.module==999 && p.ptx==neutral && p.moduleLoaded);
  cout<<"neutral cache reuse, named limits, OFF identity and transactional JIT gates: PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix="register-ptx-host-") as tmp:
    cpp=Path(tmp)/"gate.cpp"
    cpp.write_text(prefix+jit+load+checks)
    out=Path(tmp)/"gate"
    subprocess.run([os.environ.get("CXX","g++"),"-std=c++20","-Wall","-Wextra","-Werror",
                    "-fsanitize=undefined","-fno-sanitize-recover=all","-I"+str(root/"src/cuda"),
                    str(cpp),"-o",str(out)],check=True)
    subprocess.run([str(out)],check=True)
