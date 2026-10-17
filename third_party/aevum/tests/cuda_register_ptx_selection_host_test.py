#!/usr/bin/env python3
"""Compile the actual option gate, PRP parser and tuning identity on the host."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
gpu = (root / "src/Gpu.cpp").read_text()
engine = (root / "src/EngineApi.cpp").read_text()
args = (root / "src/Args.cpp").read_text()
tune = (root / "src/PrpUseTune.cpp").read_text()
gate = gpu[gpu.index("  cuda_reg_ptx_cache = false;"):gpu.index("  // PRPLL c233c11")]
identity = engine[engine.index("std::string ptxRegisterTuneIdentity("):engine.index("void set_error(")]
split = args[args.index("vector<KeyVal> Args::splitUses("):args.index("void Args::readConfig(")]
parse = tune[tune.index("std::vector<KeyVal> parse("):tune.index("std::string normalize(")]
prefix = r'''
#include "UseOptions.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using KeyVal=pair<string,string>;
struct Args {
  map<string,string> flags;
  map<string,vector<KeyVal>> perFftConfig;
  int value(const string& key,int fallback) const {
    auto it=flags.find(key); return it==flags.end()?fallback:stoi(it->second);
  }
  static vector<KeyVal> splitUses(string);
};
bool isNvidiaGpu(int id) { return id!=0; }
unsigned getNvidiaComputeCapability(int id) { return unsigned(id); }
void log(const char*,...) {}
'''
signature = '''
bool select(map<string,string> config={},int id=806,const Args& args={}) {
  bool cuda_reg_ptx_cache=true,doLog=false; (void)id;
'''
checks = r'''
template<class F> void rejects(F f) {
  bool bad=false; try {f();} catch(const runtime_error&) {bad=true;} assert(bad);
}
int main() {
  const char* key="AEVUM_CUDA_REG_PTX_CACHE";
  unsetenv(key); unsetenv("AEVUM_PRP_USE");
  assert(!select());
  assert(!select({{key,"0"}}));
  setenv(key,"0",1); assert(!select());
  rejects([&]{select({{key,"1"}});});
  unsetenv(key);
  for (auto value:{"", "01", "2", "true", "-1"})
    rejects([&]{select({{key,value}});});
#ifdef CUDA_BACKEND
  assert(select({{key,"1"}},705));
  assert(select({{key,"1"}},806));
  assert(select({{key,"1"}},809));
#else
  rejects([&]{select({{key,"1"}});});
#endif
  rejects([&]{select({{key,"1"}},700);});
  rejects([&]{select({{key,"1"}},0);});
  rejects([&]{select({{key,"1"},{"NOREG","1"}});});
  Args disabled; disabled.flags["NOREG"]="1";
  rejects([&]{select({{key,"1"}},806,disabled);});
  Args a;
  assert(ptxRegisterTuneIdentity(a).empty());
  setenv(key,"0",1); assert(ptxRegisterTuneIdentity(a).empty());
  setenv(key,"1",1); const string env=ptxRegisterTuneIdentity(a); assert(!env.empty());
  rejects([&]{select({{key,"0"}});});
#ifdef CUDA_BACKEND
  assert(select());
#endif
  unsetenv(key);
  a.flags[key]="0"; a.perFftConfig["plan"]={{key,"0"}};
  assert(ptxRegisterTuneIdentity(a).empty());
  a.flags[key]="1"; assert(!ptxRegisterTuneIdentity(a).empty());
  a.flags.clear(); a.perFftConfig["plan"]={{key,"1"}};
  const auto plan=ptxRegisterTuneIdentity(a); assert(!plan.empty() && plan!=env);
  a.perFftConfig.clear();
  setenv("AEVUM_PRP_USE","AEVUM_CUDA_REG_PTX_CACHE=1,REGMI61=80",1);
  assert(!ptxRegisterTuneIdentity(a).empty());
  setenv("AEVUM_PRP_USE","AEVUM_CUDA_REG_PTX_CACHE=0,REGMI61=80",1);
  assert(ptxRegisterTuneIdentity(a).empty());
  for (auto v:{"01","2","-1","true",""})
    rejects([&]{aevum_prp_use::parse(string(key)+'='+v);});
  assert(aevum_prp_use::parse(string(key)+"=1")[0].second=="1");
  assert(aevum_prp_use::parse(string(key)+"=0")[0].second=="0");
  cout<<"literal selection, backend scope, conflict rejection and tuning identity: PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix="ptx-selection-") as tmp:
    cpp = Path(tmp) / "test.cpp"
    cpp.write_text(prefix + split + "namespace aevum_prp_use {\n" + parse + "}\n" +
                   identity + signature + gate +
                   '\nassert(!config.count("AEVUM_CUDA_REG_PTX_CACHE")); return cuda_reg_ptx_cache; }\n' + checks)
    for cuda in (False, True):
        out = Path(tmp) / ("cuda" if cuda else "opencl")
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I" + str(root / "src"),
                        *(["-DCUDA_BACKEND=1"] if cuda else []), str(cpp), "-o", str(out)], check=True)
        subprocess.run([str(out)], check=True)
