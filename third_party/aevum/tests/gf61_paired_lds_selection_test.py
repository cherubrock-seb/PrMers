#!/usr/bin/env python3
"""Compile actual selection, parser, timing identity and GF61 dispatch bodies."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
engine = (root / "src/EngineApi.cpp").read_text()
args = (root / "src/Args.cpp").read_text()
tune = (root / "src/PrpUseTune.cpp").read_text()
fft = (root / "src/cl/fftbase.cl").read_text()
gate = gpu[gpu.index("  // AEVUM_GF61_PAIRED_LDS_GATE_BEGIN"):gpu.index("  // AEVUM_GF61_PAIRED_LDS_GATE_END")]
identity = engine[engine.index("std::string pairedLdsTuneIdentity("):engine.index("void set_error(")]
split = args[args.index("vector<KeyVal> Args::splitUses("):args.index("void Args::readConfig(")]
parse = tune[tune.index("std::vector<KeyVal> parse("):tune.index("std::string normalize(")]
begin = fft.index("void OVERLOAD shufl(local GF61 *lds, GF61 *u, u32 f, u32 numWG, u32 lowMe)")
dispatch = fft[begin:fft.index("\n}\n", begin)+3]
assert "#include \"gf61_paired_lds.cl\"" in fft
prefix = r'''
#include "UseOptions.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using u32=unsigned;
using KeyVal=pair<string,string>;
enum { FFT3161=1,FFT3261=2 };
struct Args {
  map<string,string> flags;
  map<string,vector<KeyVal>> perFftConfig;
  static vector<KeyVal> splitUses(string);
};
struct Shape { unsigned fft_type=1,width=512,height=512,middle=8; };
struct FFTConfig { Shape shape; unsigned variant=202,pfa_radix=0; bool isPfa() const {return pfa_radix!=0;} };
bool isAmdGpu(int id) {return id!=0;}
string getDeviceName(int id) {return id==2?"gfx90a":"gfx906:sramecc+:xnack-";}
unsigned getAmdWavefrontWidth(int id) {return id==3?32:64;}
void log(const char*,...) {}
'''
signature = '''
map<string,string> select(map<string,string> config={},FFTConfig fft={},int id=1) {
  bool doLog=false;
'''
checks = r'''
template<class F> void rejects(F f) {
  bool rejected=false; try{f();}catch(const runtime_error&){rejected=true;} assert(rejected);
}
int main() {
  const char* w="AEVUM_GF61_PAIRED_LDS_W";
  const char* h="AEVUM_GF61_PAIRED_LDS_H";
  for (auto name:{w,h,"AEVUM_PRP_USE","AEVUM_PFA_USE"}) unsetenv(name);
  assert(select().empty());
  assert(select({{w,"0"},{h,"0"}}).empty());
  setenv(w,"0",1); assert(select().empty());
  rejects([&]{select({{w,"1"}});}); unsetenv(w);
  for (auto key:{w,h}) {
    for (auto value:{"","01","2","-1","true"}) {
      rejects([&]{select({{key,value}});});
      rejects([&]{aevum_prp_use::parse(string(key)+'='+value);});
    }
    assert(aevum_prp_use::parse(string(key)+"=1")[0].second=="1");
    rejects([&]{select({{key,"1"}}, {},0);});
    rejects([&]{select({{key,"1"}}, {},2);});
    rejects([&]{select({{key,"1"}}, {},3);});
    for (auto conflict:{"NO_ASM","NOWG2","NOLDS2"})
      rejects([&]{select({{key,"1"},{conflict,"1"}});});
    rejects([&]{select({{key,"1"},{key==w?"SHUFL_BYTES_W":"SHUFL_BYTES_H","4"}});});
#if defined(CUDA_BACKEND) || defined(__APPLE__)
    rejects([&]{select({{key,"1"}});});
#else
    assert(select({{key,"1"}}).at(key)=="1");
    FFTConfig pfa; pfa.pfa_radix=7; pfa.shape.middle=7;
    assert(select({{key,"1"}},pfa).at(key)=="1");
    pfa.pfa_radix=9; pfa.shape.middle=9; rejects([&]{select({{key,"1"}},pfa);});
    pfa.pfa_radix=7; pfa.shape.middle=7; pfa.shape.fft_type=2;
    rejects([&]{select({{key,"1"}},pfa);});
    FFTConfig v; v.variant=101; rejects([&]{select({{key,"1"}},v);});
    v={}; v.shape.width=1024; rejects([&]{select({{key,"1"}},v);});
    v={}; v.shape.height=256; rejects([&]{select({{key,"1"}},v);});
    v={}; v.shape.fft_type=4; rejects([&]{select({{key,"1"}},v);});
    v={}; v.shape.middle=4; rejects([&]{select({{key,"1"}},v);});
    v={}; v.shape.fft_type=2; v.shape.middle=16;
    assert(select({{key,"1"}},v).at(key)=="1");
    auto options=select({{key,"1"},{"LDSPAD_W","1"},{"LDSPAD_H","0"},{"INPLACE","1"}});
    assert(options.at("LDSPAD_W")=="1" && options.at("LDSPAD_H")=="0" && options.at("INPLACE")=="1");
#endif
  }
  Args a;
  assert(pairedLdsTuneIdentity(a).empty());
  setenv(w,"0",1); setenv(h,"0",1); assert(pairedLdsTuneIdentity(a).empty());
  setenv(w,"1",1); auto wi=pairedLdsTuneIdentity(a); assert(!wi.empty());
  setenv(w,"0",1); setenv(h,"1",1); auto hi=pairedLdsTuneIdentity(a); assert(!hi.empty() && wi!=hi);
  setenv(w,"1",1); assert(pairedLdsTuneIdentity(a)!=hi && pairedLdsTuneIdentity(a)!=wi);
  unsetenv(w); unsetenv(h);
  a.flags[w]="0"; a.perFftConfig["plan"]={{h,"0"}}; assert(pairedLdsTuneIdentity(a).empty());
  a.flags[w]="1"; assert(!pairedLdsTuneIdentity(a).empty());
  a.flags.clear(); a.perFftConfig["plan"]={{h,"1"}}; assert(!pairedLdsTuneIdentity(a).empty());
  a.perFftConfig.clear();
  for (auto name:{"AEVUM_PRP_USE","AEVUM_PFA_USE"}) {
    setenv(name,(string(w)+"=0").c_str(),1); assert(pairedLdsTuneIdentity(a).empty());
    setenv(name,(string(w)+"=1").c_str(),1); assert(!pairedLdsTuneIdentity(a).empty());
    unsetenv(name);
  }
  cout<<"literal flags, scope/conflicts and timing identities: PASS\n";
}
'''
dispatch_prefix = r'''
#include <cassert>
using u32=unsigned;
struct GF61 {};
struct T2 {};
#define local
#define OVERLOAD
unsigned paired=0,stock=0;
void aevumGf61PairedLds(GF61*,GF61*,u32,u32,u32){++paired;}
void shufl64(T2*,T2*,u32,u32,u32){++stock;}
'''
dispatch_check = r'''
int main() {
  for (u32 f:{1u,8u,64u}) shufl(nullptr,nullptr,f,1,0);
  assert(paired==EXPECT && stock==3-EXPECT);
}
'''
with tempfile.TemporaryDirectory(prefix="paired-lds-selection-") as tmp:
    cpp=Path(tmp)/"test.cpp"; out=Path(tmp)/"test"
    common=[os.environ.get("CXX","g++"),"-std=c++20","-Wall","-Wextra","-Werror",
            "-fsanitize=undefined","-fno-sanitize-recover=all","-I"+str(root/"src")]
    cpp.write_text(prefix+split+"namespace aevum_prp_use {\n"+parse+"}\n"+identity+signature+gate+"return config; }\n"+checks)
    for backend in ([],["-DCUDA_BACKEND=1"],["-D__APPLE__=1"]):
        subprocess.run(common+backend+[str(cpp),"-o",str(out)],check=True)
        subprocess.run([str(out)],check=True)
    cpp.write_text("#include <initializer_list>\n"+dispatch_prefix+dispatch+dispatch_check)
    for wg,radix,bytes_,width,w,h,expected in [
        (64,8,8,1,0,0,0),(64,8,8,1,1,0,2),(64,8,8,0,1,0,0),
        (64,8,8,0,0,1,2),(64,8,8,1,0,1,0),(64,8,8,0,1,1,2),
        (32,8,8,1,1,1,0),(128,8,8,1,1,1,0),(64,4,8,1,1,1,0),
        (64,8,4,1,1,1,0),(64,8,16,1,1,1,0)]:
        values={"WG":wg,"RADIX":radix,"SHUFL_BYTES":bytes_,"DOING_WIDTH":width,
                "DOING_HEIGHT":1-width,"AEVUM_GF61_PAIRED_LDS_W":w,"AEVUM_GF61_PAIRED_LDS_H":h,"EXPECT":expected}
        subprocess.run(common+[f"-D{k}={v}" for k,v in values.items()]+[str(cpp),"-o",str(out)],check=True)
        subprocess.run([str(out)],check=True)
    print("actual GF61 wrapper, independent W/H dispatch and fallback controls: PASS")
