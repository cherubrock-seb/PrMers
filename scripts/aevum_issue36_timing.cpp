// Issue #36 production-engine timing decomposition.
// Drives the public Aevum engine API with the same FFT shape, -use profile,
// exponent, device and eight-register footprint used by PrMers PRP.
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using H = void*;

struct TimingStats {
  uint64_t square_calls{};
  uint64_t pending_flush_calls{};
  uint64_t pending_flush_ns{};
  uint64_t queue_sync_calls{};
  uint64_t queue_sync_ns{};
  uint64_t copy_calls{};
  uint64_t copy_ns{};
  uint64_t equal_calls{};
  uint64_t equal_ns{};
  uint64_t readback_calls{};
  uint64_t readback_ns{};
};

static std::string json_escape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '\\': o += "\\\\"; break;
      case '"': o += "\\\""; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default: o += static_cast<char>(c); break;
    }
  }
  return o;
}

struct Api {
  void* lib{};
  const char* (*error)(){};
  H (*create_ex)(uint32_t,size_t,uint32_t,int,const char*,const char*,uint32_t){};
  H (*create)(uint32_t,size_t,uint32_t,int,const char*,const char*){};
  void (*destroy)(H){};
  size_t (*words)(H){};
  size_t (*transform)(H){};
  int (*sync)(H){};
  int (*set)(H,size_t,uint32_t){};
  int (*put)(H,size_t,const uint32_t*,size_t){};
  int (*get)(H,size_t,uint32_t*,size_t){};
  int (*sq)(H,size_t,uint32_t){};
  int (*copy)(H,size_t,size_t){};
  int (*prepare)(H,size_t,size_t){};
  int (*mul)(H,size_t,size_t,uint32_t){};
  int (*equal)(H,size_t,size_t,int*){};
  int (*timing_reset)(H){};
  int (*timing_get)(H,TimingStats*){};

  template<class T> void load(T& p,const char* n,bool optional=false) {
    dlerror(); p=reinterpret_cast<T>(dlsym(lib,n));
    const char* e=dlerror();
    if ((!p || e) && !optional) throw std::runtime_error(std::string("missing symbol ")+n+(e?std::string(": ")+e:""));
  }
  explicit Api(const char* path) {
    lib=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!lib) throw std::runtime_error(dlerror());
    load(error,"aevum_engine_last_error");
    load(create,"aevum_engine_create");
    load(create_ex,"aevum_engine_create_ex",true);
    load(destroy,"aevum_engine_destroy");
    load(words,"aevum_engine_word_count"); load(transform,"aevum_engine_transform_size");
    load(sync,"aevum_engine_sync"); load(set,"aevum_engine_set_u32");
    load(put,"aevum_engine_set_words"); load(get,"aevum_engine_get_words");
    load(sq,"aevum_engine_square_mul"); load(copy,"aevum_engine_copy");
    load(prepare,"aevum_engine_prepare"); load(mul,"aevum_engine_mul");
    load(equal,"aevum_engine_equal");
    load(timing_reset,"aevum_engine_timing_reset"); load(timing_get,"aevum_engine_timing_get");
  }
  ~Api(){ if(lib) dlclose(lib); }
  void ok(int rc) const { if(!rc) throw std::runtime_error(error && error()?error():"Aevum call failed"); }
};

struct Engine {
  Api& api; H h{}; uint32_t p; size_t nwords{}; double create_s{};
  Engine(Api& a,uint32_t exponent,uint32_t device,const std::string& plan,const std::string& tune)
      : api(a),p(exponent) {
    const auto t0=Clock::now();
    h=api.create_ex ? api.create_ex(p,8,device,0,plan.c_str(),tune.c_str(),1u)
                    : api.create(p,8,device,0,plan.c_str(),tune.c_str());
    create_s=std::chrono::duration<double>(Clock::now()-t0).count();
    if(!h) throw std::runtime_error(api.error && api.error()?api.error():"create failed");
    nwords=api.words(h); if(!nwords || !api.transform(h)) throw std::runtime_error("invalid transform");
  }
  ~Engine(){ if(h) api.destroy(h); }
  void init(const std::vector<uint32_t>& seed) {
    api.ok(api.put(h,0,seed.data(),seed.size()));
    for(size_t r=1;r<8;++r) api.ok(api.set(h,r,static_cast<uint32_t>(r+1)));
    api.ok(api.sync(h));
  }
  std::vector<uint32_t> read(size_t r=0) {
    std::vector<uint32_t> v(nwords); api.ok(api.get(h,r,v.data(),v.size())); return v;
  }
  void reset_timing(){ api.ok(api.timing_reset(h)); }
  TimingStats stats(){ TimingStats s{}; api.ok(api.timing_get(h,&s)); return s; }
};

static std::vector<uint32_t> seed_words(size_t n,uint32_t p) {
  std::vector<uint32_t> v(n); uint32_t x=0x12345678u;
  for(auto& w:v){x^=x<<13;x^=x>>17;x^=x<<5;w=x;}
  if(p%32) v.back() &= (uint32_t(1)<<(p%32))-1u;
  return v;
}

static double ns(Clock::duration d){return std::chrono::duration<double,std::nano>(d).count();}

int main(int argc,char** argv) {
  try {
    if(argc!=11) throw std::runtime_error(
      "usage: timing LIB DEVICE EXP PLAN TUNE PROFILE SHORT_ITERS WINDOW_ITERS WINDOWS OUTDIR");
    const char* lib=argv[1]; const uint32_t device=std::stoul(argv[2]); const uint32_t p=std::stoul(argv[3]);
    const std::string plan=argv[4],tune=argv[5],profile=argv[6];
    const unsigned short_iters=std::stoul(argv[7]),window_iters=std::stoul(argv[8]),windows=std::stoul(argv[9]);
    const fs::path out=argv[10]; fs::create_directories(out);
    if(!short_iters||!window_iters||!windows) throw std::runtime_error("iteration counts must be positive");

    // Exact fixed implementation: no shape/use search, no AEVUM_TUNE_DIR injection.
    setenv("AEVUM_AUTOTUNE","off",1); setenv("AEVUM_PRP_USE_TUNE","off",1);
    setenv("AEVUM_PRP_USE",profile.c_str(),1); unsetenv("AEVUM_TUNE_DIR");
    Api api(lib);

    const uint64_t total=uint64_t(window_iters)*windows;
    std::vector<uint32_t> seed,expected;

    // A-short first: this is the closest analogue to the tuner burst. Kernel
    // construction/compilation and 64 warmup operations are outside timing.
    double short_ns=0.0; TimingStats short_stats{}; size_t nwords=0; size_t transform=0;
    {
      setenv("AEVUM_TIMING_DECOMP","1",1);
      Engine short_run(api,p,device,plan,tune);
      nwords=short_run.nwords; transform=api.transform(short_run.h);
      seed=seed_words(short_run.nwords,p); short_run.init(seed);
      for(unsigned i=0;i<64;++i) { api.ok(api.sq(short_run.h,0,1)); }
      api.ok(api.sync(short_run.h)); short_run.init(seed);
      short_run.reset_timing();
      const auto s0=Clock::now();
      for(unsigned i=0;i<short_iters;++i) { api.ok(api.sq(short_run.h,0,1)); }
      api.ok(api.sync(short_run.h));
      const auto s1=Clock::now(); short_ns=ns(s1-s0); short_stats=short_run.stats();
    }

    // Timing-disabled sustained control: same exact seed/shape/profile/device.
    // Besides giving the canonical expected residue, it quantifies any
    // perturbation caused by the opt-in timing counters themselves.
    double control_ns=0.0;
    {
      setenv("AEVUM_TIMING_DECOMP","0",1);
      Engine reference(api,p,device,plan,tune); reference.init(seed);
      for(unsigned i=0;i<64;++i) { api.ok(api.sq(reference.h,0,1)); }
      api.ok(api.sync(reference.h)); reference.init(seed);
      const auto c0=Clock::now();
      for(uint64_t i=0;i<total;++i) { api.ok(api.sq(reference.h,0,1)); }
      api.ok(api.sync(reference.h));
      const auto c1=Clock::now(); control_ns=ns(c1-c0);
      expected=reference.read();
    }

    // A-long: sustained production EngineApi square scheduler. Progress I/O is
    // deliberately timed after each compute window and excluded from long_ns.
    setenv("AEVUM_TIMING_DECOMP","1",1);
    Engine long_run(api,p,device,plan,tune); long_run.init(seed);
    for(unsigned i=0;i<64;++i) { api.ok(api.sq(long_run.h,0,1)); }
    api.ok(api.sync(long_run.h)); long_run.init(seed);
    long_run.reset_timing();
    std::vector<double> window_us; window_us.reserve(windows);
    uint64_t progress_io_ns=0,progress_bytes=0;
    double long_ns=0.0;
    std::ofstream progress(out/"progress-probe.log",std::ios::trunc);
    for(unsigned w=0;w<windows;++w) {
      const auto w0=Clock::now();
      for(unsigned i=0;i<window_iters;++i) { api.ok(api.sq(long_run.h,0,1)); }
      api.ok(api.sync(long_run.h));
      const auto w1=Clock::now();
      const double compute_ns=ns(w1-w0);
      long_ns += compute_ns;
      window_us.push_back(compute_ns/1000.0/window_iters);
      const auto io0=Clock::now();
      const std::string line="window="+std::to_string(w)+" us_per_iter="+std::to_string(window_us.back())+"\n";
      progress<<line; progress.flush(); progress_bytes+=line.size();
      progress_io_ns += static_cast<uint64_t>(ns(Clock::now()-io0));
    }
    const auto long_stats=long_run.stats();
    const auto actual=long_run.read(); const bool exact=(actual==expected);
    { std::ofstream f(out/"long-residue.bin",std::ios::binary); f.write(reinterpret_cast<const char*>(actual.data()),actual.size()*sizeof(uint32_t)); }

    // C: production Gerbicz-Li schedule from RunPrpOrLlMarin.cpp.
    // A cheap accumulation boundary occurs every B=floor(sqrt(p)) iterations;
    // the expensive replay/readback check occurs every checkpasslevel boundaries.
    const uint64_t B=std::max<uint64_t>(1,static_cast<uint64_t>(std::sqrt(static_cast<double>(p))));
    const uint64_t checkpasslevel=std::max<uint64_t>(1,static_cast<uint64_t>((1000.0*600.0)/double(B)));
    long_run.reset_timing();
    const auto gb0=Clock::now();
    api.ok(api.copy(long_run.h,3,1));
    api.ok(api.prepare(long_run.h,2,0));
    api.ok(api.mul(long_run.h,1,2,1));
    api.ok(api.sync(long_run.h));
    const auto gb1=Clock::now(); const auto gerbicz_boundary_stats=long_run.stats();

    long_run.reset_timing();
    const uint64_t modB=(p%B==0?B:p%B);
    const uint64_t replay_prefix=(B>modB?B-modB-1:0);
    const auto gc0=Clock::now();
    for(uint64_t z=0;z<replay_prefix;++z) { api.ok(api.sq(long_run.h,3,1)); }
    if(p%B==0) {
      api.ok(api.set(long_run.h,6,3));
      api.ok(api.prepare(long_run.h,7,6));
      api.ok(api.mul(long_run.h,3,7,1));
    } else {
      api.ok(api.sq(long_run.h,3,3));
    }
    for(uint64_t z=0;z<modB;++z) { api.ok(api.sq(long_run.h,3,1)); }
    std::vector<uint32_t> gerbicz_r3(nwords),gerbicz_r1(nwords);
    api.ok(api.get(long_run.h,3,gerbicz_r3.data(),gerbicz_r3.size()));
    api.ok(api.get(long_run.h,1,gerbicz_r1.data(),gerbicz_r1.size()));
    api.ok(api.copy(long_run.h,4,0));
    api.ok(api.copy(long_run.h,5,1));
    api.ok(api.sync(long_run.h));
    const auto gc1=Clock::now(); const auto gerbicz_check_stats=long_run.stats();
    const double gerbicz_boundary_ns=ns(gb1-gb0);
    const double gerbicz_check_ns=ns(gc1-gc0);
    const double gerbicz_amortized_ns=gerbicz_boundary_ns/double(B) +
        gerbicz_check_ns/double(B*checkpasslevel);

    // GPU-side equality primitive is not used by the current host mpz compare,
    // but is measured separately because it is a candidate future check path.
    long_run.reset_timing(); const unsigned equality_reps=32; int eq=0;
    const auto ge0=Clock::now();
    for(unsigned i=0;i<equality_reps;++i) {
      api.ok(api.copy(long_run.h,2,0));
      api.ok(api.equal(long_run.h,2,0,&eq));
      if(!eq) throw std::runtime_error("copy/equality exactness failure");
    }
    api.ok(api.sync(long_run.h));
    const auto ge1=Clock::now(); const auto equality_stats=long_run.stats();

    // D: canonical residue, proof-like one-register export and complete
    // eight-register checkpoint export. Readback and file I/O are both kept.
    long_run.reset_timing();
    const auto r0=Clock::now(); const auto residue=long_run.read(0); const auto r1=Clock::now();
    const auto rf0=Clock::now();
    { std::ofstream f(out/"residue-export.bin",std::ios::binary); f.write(reinterpret_cast<const char*>(residue.data()),residue.size()*sizeof(uint32_t)); f.flush(); }
    const auto rf1=Clock::now();
    const auto p0=Clock::now(); const auto proof_words=long_run.read(0);
    { std::ofstream f(out/"proof-export-proxy.bin",std::ios::binary); f.write(reinterpret_cast<const char*>(proof_words.data()),proof_words.size()*sizeof(uint32_t)); f.flush(); }
    const auto p1=Clock::now();
    const auto c0=Clock::now();
    { std::ofstream f(out/"checkpoint-export.bin",std::ios::binary);
      for(size_t r=0;r<8;++r) { const auto v=long_run.read(r); f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(uint32_t)); }
      f.flush(); }
    const auto c1=Clock::now(); const auto export_stats=long_run.stats();

    std::ofstream j(out/"timing-decomposition.json",std::ios::trunc); if(!j)throw std::runtime_error("cannot write JSON");
    j<<std::setprecision(12);
    j<<"{\n";
    j<<"  \"schema\": \"prmers-issue36-timing-v1\",\n";
    j<<"  \"exponent\": "<<p<<", \"device\": "<<device<<",\n";
    j<<"  \"shape\": \""<<json_escape(plan)<<"\",\n";
    j<<"  \"profile\": \""<<json_escape(profile)<<"\",\n";
    j<<"  \"register_count\": 8, \"transform\": "<<transform<<", \"word_count\": "<<nwords<<",\n";
    j<<"  \"exact_timing_on_vs_off\": "<<(exact?"true":"false")<<",\n";
    j<<"  \"A_raw_square_hot_path\": {\"short_iterations\": "<<short_iters<<", \"short_us_per_iter\": "<<(short_ns/1000.0/short_iters)
     <<", \"control_long_us_per_iter\": "<<(control_ns/1000.0/total)
     <<", \"timing_on_over_control_ratio\": "<<(long_ns/control_ns)
     <<", \"long_iterations\": "<<total<<", \"long_us_per_iter\": "<<(long_ns/1000.0/total)<<", \"long_windows_us_per_iter\": [";
    for(size_t i=0;i<window_us.size();++i){if(i)j<<",";j<<window_us[i];} j<<"]},\n";
    j<<"  \"B_pending_flush\": {\"calls\": "<<long_stats.pending_flush_calls<<", \"ns\": "<<long_stats.pending_flush_ns
     <<", \"ns_per_iteration\": "<<(double(long_stats.pending_flush_ns)/total)<<"},\n";
    j<<"  \"C_gerbicz\": {\"B\": "<<B<<", \"checkpasslevel\": "<<checkpasslevel
     <<", \"boundary_wall_ns\": "<<uint64_t(gerbicz_boundary_ns)
     <<", \"full_check_wall_ns\": "<<uint64_t(gerbicz_check_ns)
     <<", \"estimated_amortized_ns_per_prp_iteration\": "<<gerbicz_amortized_ns
     <<", \"boundary_copy_calls\": "<<gerbicz_boundary_stats.copy_calls
     <<", \"full_check_square_calls\": "<<gerbicz_check_stats.square_calls
     <<", \"full_check_readback_calls\": "<<gerbicz_check_stats.readback_calls
     <<", \"gpu_equal_probe\": {\"repetitions\": "<<equality_reps
     <<", \"wall_ns\": "<<uint64_t(ns(ge1-ge0))
     <<", \"copy_ns\": "<<equality_stats.copy_ns<<", \"equal_ns\": "<<equality_stats.equal_ns<<"}},\n";
    j<<"  \"D_export\": {\"residue_readback_wall_ns\": "<<uint64_t(ns(r1-r0))
     <<", \"residue_file_wall_ns\": "<<uint64_t(ns(rf1-rf0))
     <<", \"proof_single_register_proxy_wall_ns\": "<<uint64_t(ns(p1-p0))
     <<", \"checkpoint8_wall_ns\": "<<uint64_t(ns(c1-c0))
     <<", \"readback_calls\": "<<export_stats.readback_calls<<", \"readback_ns\": "<<export_stats.readback_ns<<"},\n";
    j<<"  \"E_queue_waits\": {\"calls\": "<<long_stats.queue_sync_calls<<", \"ns\": "<<long_stats.queue_sync_ns
     <<", \"ns_per_iteration\": "<<(double(long_stats.queue_sync_ns)/total)<<"},\n";
    j<<"  \"F_progress_io_probe\": {\"events\": "<<windows<<", \"bytes\": "<<progress_bytes<<", \"ns\": "<<progress_io_ns
     <<", \"ns_per_iteration\": "<<(double(progress_io_ns)/total)<<"},\n";
    j<<"  \"G_short_vs_long\": {\"long_over_short_ratio\": "<<((long_ns/total)/(short_ns/short_iters))
     <<", \"external_clock_thermal_file\": \"gpu-clock-thermal.csv\"},\n";
    j<<"  \"raw_counters_short\": {\"square_calls\": "<<short_stats.square_calls<<", \"pending_flush_ns\": "<<short_stats.pending_flush_ns
     <<", \"queue_sync_ns\": "<<short_stats.queue_sync_ns<<"},\n";
    j<<"  \"notes\": [\"A/B/E use the production EngineApi pending-square scheduler; no Gerbicz/proof/logging work is inside A\","
       "\"C reproduces the current RunPrpOrLlMarin Gerbicz boundary/replay cadence and reports an amortized per-PRP-iteration estimate; gpu_equal_probe is diagnostic only\","
       "\"D measures canonical residue, a one-register proof-export proxy, and eight-register checkpoint export separately\","
       "\"F is a host I/O floor; production spinner/log cadence must be interpreted from a real PRP run\"]\n";
    j<<"}\n"; j.close();
    if(!exact) throw std::runtime_error("timing instrumentation changed canonical residue");
    std::cout<<"AEVUM_TIMING_DECOMP "<<(out/"timing-decomposition.json")<<"\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
