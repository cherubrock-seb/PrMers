#!/usr/bin/env python3
"""Compile actual gate, parser, timing identity and coefficient emission bodies."""
from pathlib import Path
import os
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
gpu=(root/"src/Gpu.cpp").read_text()
engine=(root/"src/EngineApi.cpp").read_text()
args=(root/"src/Args.cpp").read_text()
tune=(root/"src/PrpUseTune.cpp").read_text()
gate=gpu[gpu.index("  // AEVUM_FP32_MINIMAX_TRIG_GATE_BEGIN"):gpu.index("  // AEVUM_FP32_MINIMAX_TRIG_GATE_END")]
identity=engine[engine.index("std::string fp32MinimaxTuneIdentity("):engine.index("void set_error(")]
split=args[args.index("vector<KeyVal> Args::splitUses("):args.index("void Args::readConfig(")]
parse=tune[tune.index("std::vector<KeyVal> parse("):tune.index("std::string normalize(")]
start=gpu.index("TrigCoefsFP32 const coefs = fp32MinimaxTrig")
finish=gpu.index("defines += toDefine(\"TRIG_COS\", cosCoefs + '}');",start)+len("defines += toDefine(\"TRIG_COS\", cosCoefs + '}');")
emit=gpu[start:finish]
assert engine.count("flags += fp32MinimaxTuneIdentity(*shared.args)")==1
assert engine.count("flags << fp32MinimaxTuneIdentity(args_)")==1
prefix=r'''
#include "UseOptions.h"
#include "Trig.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using KeyVal=pair<string,string>;
enum {FFT3261=2};
struct Args { map<string,string> flags; map<string,vector<KeyVal>> perFftConfig; static vector<KeyVal> splitUses(string); };
struct Shape {
 unsigned fft_type=2,width=512,height=512,middle=8;
 unsigned size() const {return width*height*middle*2;}
};
struct FFTConfig {Shape shape; unsigned variant=202,pfa_radix=0; bool isPfa() const{return pfa_radix!=0;}};
bool isAmdGpu(int id) {return id>=1 && id<=3;}
string getDeviceName(int id) {return id==2?"gfx90a":"gfx906:sramecc+:xnack-";}
unsigned getAmdWavefrontWidth(int id) {return id==3?32:64;}
bool isNvidiaGpu(int id) {return id>=4;}
unsigned getNvidiaComputeCapability(int id) {return id==4?86:id==5?89:70;}
void log(const char*,...) {}
string toDefine(const string& key,const string& value) {return key+'='+value+'\n';}
'''
checks=r'''
template<class F> void rejects(F f) {bool caught=false;try{f();}catch(const runtime_error&){caught=true;}assert(caught);}
int main() {
 const char* key="AEVUM_FP32_MINIMAX_TRIG";
 for(auto name:{key,"AEVUM_PRP_USE","AEVUM_PFA_USE"})unsetenv(name);
 assert(!select().first && select().second.empty());
 assert(select({{key,"0"}}).second.empty());
 setenv(key,"0",1);assert(!select().first);rejects([&]{select({{key,"1"}});});unsetenv(key);
 for(auto value:{"","01","2","-1","true"}) {
  rejects([&]{select({{key,value}});});
  setenv(key,value,1);rejects([&]{select();});unsetenv(key);
  rejects([&]{aevum_prp_use::parse(string(key)+'='+value);});
 }
 assert(aevum_prp_use::parse(string(key)+"=1")[0].second=="1");
#if defined(__APPLE__)
 rejects([&]{select({{key,"1"}});});
#else
#if defined(CUDA_BACKEND)
 const int valid=4;
 for(int id:{0,1,2,3,6})rejects([&]{select({{key,"1"}}, {},id);});
 assert(select({{key,"1"}}, {},5).first);
#else
 const int valid=1;
 for(int id:{0,2,3,4,5,6})rejects([&]{select({{key,"1"}}, {},id);});
#endif
 assert(select({{key,"1"}}, {},valid).first);
 setenv(key,"1",1);assert(select({}, {},valid).first);rejects([&]{select({{key,"0"}}, {},valid);});unsetenv(key);
 FFTConfig f;
 f.pfa_radix=7;rejects([&]{select({{key,"1"}},f,valid);});
 f={};f.variant=101;rejects([&]{select({{key,"1"}},f,valid);});
 f={};f.shape.fft_type=1;rejects([&]{select({{key,"1"}},f,valid);});
 f={};f.shape.width=1024;rejects([&]{select({{key,"1"}},f,valid);});
 f={};f.shape.height=256;rejects([&]{select({{key,"1"}},f,valid);});
 f={};f.shape.middle=4;rejects([&]{select({{key,"1"}},f,valid);});
 f={};f.shape.middle=16;assert(select({{key,"1"}},f,valid).first);
 auto cfg=select({{key,"1"},{"TAIL_TRIGS32","2"},{"INPLACE","1"},{"REGTS32","80"}}, {},valid).second;
 assert(cfg.at("TAIL_TRIGS32")=="2" && cfg.at("INPLACE")=="1" && cfg.at("REGTS32")=="80");
#endif
 Args a;assert(fp32MinimaxTuneIdentity(a).empty());
 setenv(key,"0",1);assert(fp32MinimaxTuneIdentity(a).empty());
 setenv(key,"1",1);assert(!fp32MinimaxTuneIdentity(a).empty());unsetenv(key);
 a.flags[key]="0";a.perFftConfig["plan"]={{key,"0"}};assert(fp32MinimaxTuneIdentity(a).empty());
 a.flags[key]="1";assert(!fp32MinimaxTuneIdentity(a).empty());a.flags.clear();
 a.perFftConfig["plan"]={{key,"1"}};assert(!fp32MinimaxTuneIdentity(a).empty());a.perFftConfig.clear();
 for(auto name:{"AEVUM_PRP_USE","AEVUM_PFA_USE"}) {
  setenv(name,(string(key)+"=0").c_str(),1);assert(fp32MinimaxTuneIdentity(a).empty());
  setenv(name,(string(key)+"=1").c_str(),1);assert(!fp32MinimaxTuneIdentity(a).empty());unsetenv(name);
 }
 for(unsigned middle:{8u,16u}) for(bool active:{false,true}) {
  FFTConfig f;f.shape.middle=middle;
  const auto c=active?trigCoefsFP32Minimax(f.shape.size()/4):trigCoefsFP32(f.shape.size()/4);
  string defs=coefficients(active,f);
  // The actual host formatter's nine significant digits must round-trip.
  auto read=[&](const string& key) {auto first=defs.find(key+'=')+key.size()+1;return defs.substr(first,defs.find('\n',first)-first);};
  assert(stof(read("TRIG_SCALE"))==float(c.scale));
  for(bool sine:{false,true}) {
   string values=read(sine?"TRIG_SIN":"TRIG_COS");values.erase(0,1);
   for(unsigned i=0;i<8;++i) {
    size_t end=values.find(',');const float value=stof(values.substr(0,end));values.erase(0,end+1);
    assert(bit_cast<uint32_t>(value)==bit_cast<uint32_t>(float(sine?c.sinCoefs[i]:c.cosCoefs[i])));
   }
   assert(values=="}");
  }
 }
 assert(coefficients(false,{})!=coefficients(true,{}));
 cout<<"literal/scope/backend gates, cache identities and actual coefficient emission: PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix="fp32-minimax-select-") as tmp:
 cpp=Path(tmp)/"test.cpp";out=Path(tmp)/"test"
 cpp.write_text(prefix+split+"namespace aevum_prp_use {\n"+parse+"}\n"+identity+
  "pair<bool,map<string,string>> select(map<string,string> config={},FFTConfig fft={},[[maybe_unused]] int id=1) { bool doLog=false;\n"+gate+"return {fp32MinimaxTrig,config};}\n"+
  "string coefficients(bool fp32MinimaxTrig, FFTConfig fft) { string defines;\n"+emit+"\nreturn defines;}\n"+checks)
 for backend in ([],["-DCUDA_BACKEND=1"],["-D__APPLE__=1"]):
  subprocess.run([os.environ.get("CXX","g++"),"-std=c++20","-Wall","-Wextra","-Werror",
   "-fsanitize=undefined","-fno-sanitize-recover=all","-I"+str(root/"src"),*backend,
   str(cpp),str(root/"src/Trig.cpp"),"-o",str(out)],check=True)
  subprocess.run([str(out)],check=True)
