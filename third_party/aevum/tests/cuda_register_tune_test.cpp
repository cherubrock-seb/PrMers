#include "CudaRegisterTune.h"
#include "PrpUseTune.h"
#include "UseOptions.h"
#include <cassert>
#include <iostream>
#include <set>

int main() {
  using namespace aevum_cuda_register_tune;
  assert(!enabled(nullptr) && !enabled("0") && enabled("1"));
  for (const char* value : {"", "true", "01", "2", "-1", "1 "}) {
    bool rejected=false;
    try { enabled(value); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
  }
  const std::vector<std::pair<std::string,std::string>> base = {
      {"INPLACE","1"}, {"REGCF3161","2"}, {"REGMO61","-1"},
      {"ZEROHACK_W","0"}, {"LOADS","41042"}};
  const auto unchanged=base;
  for (unsigned type : {1u,2u}) {
    const auto profiles=candidates(type,base);
    std::set<std::string> unique(profiles.begin(),profiles.end()), keys;
    assert(unique.size()==profiles.size());
    assert(profiles.size()==(type==1?61:62));
    for (const auto& profile:profiles) {
      auto eq=profile.find('=');
      assert(eq!=std::string::npos && profile.find(',')==std::string::npos);
      const auto name=profile.substr(0,eq);
      const int value=std::stoi(profile.substr(eq+1));
      assert(aevumRegisterUseKey(name));
      assert(value==-1 || (value>=1 && value<=4) ||
             value==64 || value==80 || value==96 || value==128);
      assert(name!=std::string("REGCF")+(type==1?"3261":"3161"));
      assert(name!="REGMI"+std::string(type==1?"32":"31"));
      keys.insert(name);
    }
    assert(keys.size()==7);
    assert(!unique.count("REGMO61=-1"));
    if (type==1) assert(!unique.count("REGCF3161=2"));
    // All candidates must finish before a cache decision is final, including
    // lists longer than the historical 12-candidate OpenCL neighbourhood.
    aevum_prp_use::SearchProgress progress;
    progress.planned=profiles.size(); progress.screened=12;
    assert(progress.decision(true)==aevum_prp_use::Decision::Deferred);
    progress.screened=progress.planned; progress.finalists=4;progress.finalized=3;
    assert(progress.decision(true)==aevum_prp_use::Decision::Deferred);
    progress.finalized=4;
    assert(progress.decision(true)==aevum_prp_use::Decision::Positive);
    assert(progress.decision(false)==aevum_prp_use::Decision::Negative);
  }
  assert(base==unchanged);
  bool rejected=false;
  try { candidates(4,{}); } catch (const std::runtime_error&) { rejected=true; }
  assert(rejected);
  using aevum_prp_use::key;
  const auto production=key("device","1:512:8:512:202",136279841,"REGCF3161=2");
  const auto tuned=key("cuda-register-neighborhood-v1|device","1:512:8:512:202",136279841,"REGCF3161=2");
  assert(production!=tuned);
  assert(tuned!=key("cuda-register-neighborhood-v1|device","1:512:8:512:202",136279839,"REGCF3161=2"));
  assert(tuned!=key("cuda-register-neighborhood-v1|device","1:512:8:512:202",136279841,"REGCF3161=1"));
  std::cout << "CUDA register neighbourhood, strict flags, completion and cache isolation PASS\n";
}
