#include "PrpUseTune.h"
#include "Args.h"
#include "TuneEntry.h"
#include <fstream>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace aevum_prp_use;
void check(bool v){if(!v)throw std::runtime_error("PRP use regression");}
int main(){
  check(normalize(parse("ZEROHACK_W=0,INPLACE,INPLACE=2"))=="INPLACE=1,ZEROHACK_W=0");
  for(const char* bad:{"FUSED_MUL3=1","UNKNOWN=1","INPLACE=9","LOADS=12345","STORES=99","MODM31=-1"}){
    bool threw=false;try{parse(bad);}catch(...){threw=true;}check(threw);
  }
  check(candidates(true).size()<=12);
  check(candidates(true).front()==normalize(parse("INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0")));
  // Use a unique directory so stale caches from an earlier test run cannot
  // masquerade as a current implementation-profile decision.
  const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
  auto tmp=std::filesystem::temp_directory_path()/("aevum-pass5-use-test-"+std::to_string(nonce));
  std::filesystem::create_directories(tmp);
#if defined(_WIN32)
  _putenv_s("AEVUM_AUTOTUNE_CACHE",(tmp/"shape.tsv").string().c_str());
#else
  setenv("AEVUM_AUTOTUNE_CACHE",(tmp/"shape.tsv").c_str(),1);
#endif
  auto identity=aevum_autotune::makeKey("test","vendor","device","driver","runtime",aevum_autotune::Workload::Prp,4,147800003,"safe");
  auto k=key(identity,"1:512:8:512:202",147800003,"");
  aevum_autotune::Record old{k,"1:512:8:512:202"};aevum_autotune::storeAtomic(aevum_autotune::cachePath(),old);
  check(!load(k)); // v1 shape-only record cannot become an implementation profile
  store(k,"INPLACE=1",1.10,10);check(load(k)->plan=="INPLACE=1");
  check(!load(key(identity,"4:512:8:512:202",147800003,"")));
  check(!load(key(identity,"1:512:8:512:202",147800001,"")));
  // Exhausted/partial budgets never create a final negative record.
  auto deferred_key=key(identity,"1:512:8:512:202",147800005,"");
  SearchProgress progress{.planned=4};
  check(progress.decision(false)==Decision::Deferred);
  check(!persistDecision(deferred_key,progress.decision(false),"",1,0));check(!load(deferred_key));
  progress.screened=3;check(progress.decision(false)==Decision::Deferred);
  progress.screened=4;progress.finalists=1;check(progress.decision(false)==Decision::Deferred);
  progress.finalized=1;progress.interrupted=true;check(progress.decision(false)==Decision::Deferred);
  progress.interrupted=false;check(progress.decision(false)==Decision::Negative);
  check(persistDecision(deferred_key,progress.decision(false),"",1,100));
  check(load(deferred_key)->plan=="defaults"); // A complete negative AUTO cache hit terminates search.
  check(aevum_autotune::load(path(deferred_key),deferred_key)->plan=="defaults-complete");
  check(!persistDecision(deferred_key,Decision::Deferred,"",1,0));check(!load(deferred_key));
  // Deferred searches checkpoint candidate progress separately from the final
  // positive/negative cache. This prevents slow OpenCL compilation from
  // restarting at candidate zero on every AUTO launch.
  ResumeState resume;resume.planned=7;resume.next_screen=3;
  resume.top.push_back({candidates(true)[0],1.04});
  resume.top.push_back({candidates(true)[1],1.035});
  resume.top.push_back({candidates(true)[2],1.03});
  storeProgress(deferred_key,resume);
  auto resumed=loadProgress(deferred_key);check(resumed && resumed->next_screen==3 && resumed->top.size()==3);
  check(resumed->top[0].first==candidates(true)[0]);
  check(resumed->top[2].first==candidates(true)[2]);
  clearProgress(deferred_key);check(!loadProgress(deferred_key));
  storeProgress(deferred_key,resume);
  check(persistDecision(deferred_key,Decision::Negative,"",1,100));check(!loadProgress(deferred_key));
  check(!persistDecision(k,Decision::Deferred,"",1,0));check(load(k)->plan=="INPLACE=1");
  check(progress.decision(true)==Decision::Positive);
  progress.interrupted=true;check(progress.decision(true)==Decision::Deferred);
  // Legacy ambiguous negatives retry, while confirmed positives stay usable.
  auto legacy=*load(k);legacy.plan="defaults";legacy.implementation_speedup=1;
  aevum_autotune::storeAtomic(path(k),legacy);check(!load(k));
  store(k,"INPLACE=1",1.10,10);check(load(k)->plan=="INPLACE=1");
  bool low_gain=false;try{persistDecision(k,Decision::Positive,"INPLACE=1",1.02,100);}catch(...){low_gain=true;}
  check(low_gain);check(load(k)->plan=="INPLACE=1");
  store(k,"",1,10);check(load(k)->plan=="defaults");
  auto bad=*load(k);bad.implementation_speedup=std::numeric_limits<double>::infinity();
  aevum_autotune::storeAtomic(path(k),bad);check(!load(k));
  {std::ofstream f(path(k));f<<"corrupted";}check(!load(k));
  check(aevum_autotune::load(aevum_autotune::cachePath(),k)->plan==old.plan);
  // Issue #36: tune.txt replays the validated implementation flags together
  // with the tuned shape instead of silently dropping -use.
  {std::ofstream f(tmp/"tune.txt");f<<"1.0 1:512:8:512:202 -use INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0 # GB202\n";}
  Args args(true);args.masterDir=tmp;auto entries=TuneEntry::readTuneFile(args);
  check(entries.size()==1 && normalize(entries[0].use)==normalize(parse("INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0")));
  check(entries[0].fft.spec()=="1:512:8:512:202");
  std::filesystem::remove_all(tmp);
  std::cout<<"PRP use parser/cache/tune-entry separation tests passed\n";
}
