#include "PrpUseTune.h"
#include "UseOptions.h"
#include "Args.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <fstream>
#include <iomanip>
namespace aevum_prp_use {
Decision SearchProgress::decision(bool has_winner) const {
  // R5 deliberately finishes every shortlisted finalist before committing a
  // positive result.  Screening order is only a cheap ranking heuristic; the
  // best real-PRP profile may be second or third after the final A/B gate.
  if(!interrupted && planned && screened==planned && finalized==finalists)
    return has_winner ? Decision::Positive : Decision::Negative;
  return Decision::Deferred;
}
bool persistDecision(const std::string& key,Decision decision,const std::string& profile,double gain,uint64_t ms) {
  if(decision==Decision::Deferred) {
    // A forced retune supersedes an old negative. Keep prior positive winners.
    auto old=aevum_autotune::load(path(key),key);
    if(old && (old->plan=="defaults" || old->plan=="defaults-complete")) {
      std::error_code ec;std::filesystem::remove(path(key),ec);
    }
    return false;
  }
  if((decision==Decision::Negative && !profile.empty()) ||
     (decision==Decision::Positive && (profile.empty() || !std::isfinite(gain) || gain<1.03)))
    throw std::runtime_error("inconsistent PRP use decision");
  store(key,profile,gain,ms);
  clearProgress(key);
  return true;
}
std::vector<KeyVal> parse(const std::string& text) {
  // Args owns comma/whitespace/bare-key semantics, just as native -use.
  std::map<std::string,std::string> values;
  for(auto [k,v]:Args::splitUses(text)) {
    if(!aevumUseKey(k))throw std::runtime_error("unrecognized AEVUM_PRP_USE key: "+k);
    if(v.empty() || v.find_first_not_of("0123456789")!=std::string::npos || v.size()>9)
      throw std::runtime_error("AEVUM_PRP_USE requires unsigned integer values: "+k);
    auto n=std::stoul(v);
    if((k=="INPLACE" && n>2) || (k=="MODM31" && n>3) ||
       ((k.rfind("TABMUL_CHAIN",0)==0 || k.rfind("ZEROHACK_",0)==0) && n>1))
      throw std::runtime_error("unsupported AEVUM_PRP_USE value: "+k);
    if(k=="LOADS" || k=="STORES") {
      if(n>(k=="LOADS"?44444u:44u))throw std::runtime_error("invalid load/store policy");
      for(auto x=n;x;x/=10)if(x%10>4)throw std::runtime_error("invalid load/store digit");
    }
    // Native Gpu config is a map constructed from the use vector: first wins.
    values.emplace(k,std::to_string(n));
  }
  return {values.begin(),values.end()};
}
std::string normalize(const std::vector<KeyVal>& v) {
  std::map<std::string,std::string> m(v.begin(),v.end()); std::string result;
  for(const auto& [k,x]:m){if(!result.empty())result+=',';result+=k+'='+x;}
  return result;
}
std::vector<std::string> candidates(bool nvidia_asm) {
  // Bounded, hardware-aware neighbourhood.  The first entries preserve the
  // issue-#36 reporter vector and the independently validated reduced-memory
  // profile.  The rest are one-knob mutations around those seeds rather than a
  // combinatorial grid.  This mirrors the coordinate-wise intent of upstream
  // tune.cpp while keeping first-use tuning finite and resumable.
  const std::string memory=nvidia_asm?"INPLACE=1,LOADS=10040,STORES=22":"INPLACE=1,LOADS=11111,STORES=11";
  const std::string trig="TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0";
  std::vector<std::string> raw={
    memory+','+trig,                         // historical full reporter profile
    memory,                                  // reduced memory profile
    memory+",TABMUL_CHAIN32=1",             // local trig mutation
    memory+",MODM31=2",                    // local mod mutation
    memory+",ZEROHACK_W=0",                // local zero-hack mutation
    trig,                                    // arithmetic-only profile
    "INPLACE=1",                            // isolate in-place execution
    memory+",MODM31=1",                    // neighbouring MODM31 policy
    nvidia_asm ? "INPLACE=0,LOADS=10040,STORES=22"
               : "INPLACE=0,LOADS=11111,STORES=11", // ablate INPLACE only
    nvidia_asm ? "INPLACE=1,LOADS=10041,STORES=22,"+trig
               : "INPLACE=2,LOADS=11111,STORES=11,"+trig,
    memory+",ZEROHACK_W=1",
    memory+",TABMUL_CHAIN32=1,ZEROHACK_W=0"
  };
  std::vector<std::string> out;
  for(auto& profile:raw) {
    profile=normalize(parse(profile));
    if(std::find(out.begin(),out.end(),profile)==out.end())out.push_back(profile);
  }
  return out;
}
aevum_autotune::Mode mode() {
  const char* e=std::getenv("AEVUM_PRP_USE_TUNE");
  if(!e || !*e)return aevum_autotune::modeFromEnvironment();
  std::string v=e;for(auto& c:v)c=char(std::tolower(static_cast<unsigned char>(c)));
  if(v=="off"||v=="0")return aevum_autotune::Mode::Off;
  if(v=="auto"||v=="1")return aevum_autotune::Mode::Auto;
  if(v=="force"||v=="retune"||v=="force-retune")return aevum_autotune::Mode::Retune;
  throw std::runtime_error("AEVUM_PRP_USE_TUNE must be off, auto, or retune");
}
std::string key(const std::string& identity,const std::string& shape,uint32_t p,const std::string& flags) {
  // Exact exponent is deliberate extra protection inside the required band:
  // a near-capacity Type4 result cannot bleed into another workload/band edge.
  return "prp-use-v4|"+identity+"|shape="+shape+"|p="+std::to_string(p)+"|use-flags="+flags;
}
std::filesystem::path path(const std::string& key) {
  uint64_t h=14695981039346656037ULL;for(unsigned char c:key){h^=c;h*=1099511628211ULL;}
  std::ostringstream o;o<<std::hex<<h;
  return std::filesystem::path(aevum_autotune::cachePath().string()+".prp-use-v4")/(o.str()+".tsv");
}
std::filesystem::path progressPath(const std::string& key) {
  auto p=path(key);p.replace_extension(".progress.tsv");return p;
}
std::optional<ResumeState> loadProgress(const std::string& key) {
  std::ifstream in(progressPath(key));if(!in)return {};
  try {
    ResumeState state;std::string line;
    if(!std::getline(in,line))return {};
    // v2 stores the candidate-plan size as well as the resume cursors.  This
    // matters because a cold shape tune deliberately screens fewer -use
    // candidates; the next AUTO invocation may expand the bounded search.
    if(std::count(line.begin(),line.end(),'\t')<6)return {};
    std::istringstream head(line);std::string tag,stored_key;
    if(!std::getline(head,tag,'\t') || tag!="2" || !std::getline(head,stored_key,'\t') || stored_key!=key)return {};
    std::string planned,ns,nf,wg,winner;
    if(!std::getline(head,planned,'\t') || !std::getline(head,ns,'\t') || !std::getline(head,nf,'\t') ||
       !std::getline(head,wg,'\t'))return {};
    std::getline(head,winner); // empty is a valid "no confirmed winner yet" state
    state.planned=std::stoul(planned);state.next_screen=std::stoul(ns);state.next_finalist=std::stoul(nf);
    state.winning_gain=std::stod(wg);state.winner=winner;
    if(!state.winner.empty())state.winner=normalize(parse(state.winner));
    while(std::getline(in,line)) {
      if(line.empty())continue;
      std::istringstream row(line);std::string kind,gain,profile;
      if(!std::getline(row,kind,'\t') || kind!="T" || !std::getline(row,gain,'\t') || !std::getline(row,profile))continue;
      const double g=std::stod(gain);if(!std::isfinite(g) || g<=0)continue;
      state.top.push_back({normalize(parse(profile)),g});
    }
    if(!std::isfinite(state.winning_gain) || state.winning_gain<1.0)return {};
    if(state.top.size()>4)return {};
    return state;
  } catch(...) { return {}; }
}
void storeProgress(const std::string& key,const ResumeState& state) {
  auto file=progressPath(key);std::error_code ec;std::filesystem::create_directories(file.parent_path(),ec);
  if(ec)throw std::runtime_error("cannot create PRP use progress directory: "+ec.message());
  auto tmp=file;tmp += ".tmp."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  {
    std::ofstream out(tmp,std::ios::trunc);if(!out)throw std::runtime_error("cannot write PRP use progress");
    out<<"2\t"<<key<<'\t'<<state.planned<<'\t'<<state.next_screen<<'\t'<<state.next_finalist<<'\t'
       <<std::setprecision(17)<<state.winning_gain<<'\t'<<state.winner<<'\n';
    for(const auto& [profile,gain]:state.top)out<<"T\t"<<std::setprecision(17)<<gain<<'\t'<<profile<<'\n';
    out.flush();if(!out)throw std::runtime_error("cannot flush PRP use progress");
  }
  std::filesystem::rename(tmp,file,ec);
  if(ec) {
    std::error_code rm;std::filesystem::remove(file,rm);ec.clear();std::filesystem::rename(tmp,file,ec);
  }
  if(ec){std::error_code rm;std::filesystem::remove(tmp,rm);throw std::runtime_error("cannot commit PRP use progress: "+ec.message());}
}

void clearProgress(const std::string& key) {
  std::error_code ec;std::filesystem::remove(progressPath(key),ec);
}
std::optional<aevum_autotune::Record> load(const std::string& key) {
  auto r=aevum_autotune::load(path(key),key);if(!r)return {};
  try {
    if(!std::isfinite(r->implementation_speedup) || r->implementation_speedup<1.0)return {};
    // Legacy defaults can mean zero-budget or a cold-first-sample rejection.
    // Retry them once; preserve all previously confirmed positive profiles.
    if(r->plan=="defaults")return {};
    if(r->plan=="defaults-complete")r->plan="defaults";
    else if(r->implementation_speedup<1.03 || normalize(parse(r->plan))!=r->plan)return {};
  }catch(...){return {};}
  return r;
}
void store(const std::string& key,const std::string& profile,double gain,uint64_t ms) {
  aevum_autotune::Record r;
  r.key=key;r.plan=profile.empty()?"defaults-complete":normalize(parse(profile));
  r.implementation_speedup=profile.empty()?1:gain;r.tune_ms=ms;r.created_unix=std::time(nullptr);
  aevum_autotune::storeAtomic(path(key),r);
}
}
