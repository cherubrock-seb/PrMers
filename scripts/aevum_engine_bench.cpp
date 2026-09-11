// Tight-loop AEVUM benchmark and boundary/alias regression; no math algorithms changed.
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Api {
  void* lib;
  using H = void*;
  H h{};
  const char* (*error)();
  H (*create)(uint32_t,size_t,uint32_t,int,const char*,const char*);
  H (*create_ex)(uint32_t,size_t,uint32_t,int,const char*,const char*,uint32_t){};
  void (*destroy)(H);
  size_t (*words)(H);
  size_t (*transform)(H);
  int (*sync)(H);
  int (*set)(H,size_t,uint32_t);
  int (*put)(H,size_t,const uint32_t*,size_t);
  int (*get)(H,size_t,uint32_t*,size_t);
  int (*sq)(H,size_t,uint32_t);
  int (*sub)(H,size_t,uint32_t);
  int (*copy)(H,size_t,size_t);
  int (*prepare)(H,size_t,size_t);
  int (*mul)(H,size_t,size_t,uint32_t);
  int (*add)(H,size_t,size_t);
  int (*subreg)(H,size_t,size_t);
  int (*equal)(H,size_t,size_t,int*);
  int (*profile)(H,int);
  template<class T> void load(T& p, const char* name) {
    p = reinterpret_cast<T>(dlsym(lib,name));
    if (!p) throw std::runtime_error(name);
  }
  Api(const char* path, uint32_t p, uint32_t d, const char* plan, const char* tune, uint32_t workload) {
    lib=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!lib) throw std::runtime_error(dlerror());
#define SYM(field, name) load(field,"aevum_engine_" name)
    SYM(error,"last_error"); SYM(create,"create");
    create_ex=reinterpret_cast<decltype(create_ex)>(dlsym(lib,"aevum_engine_create_ex"));
    SYM(destroy,"destroy");
    SYM(words,"word_count"); SYM(transform,"transform_size"); SYM(sync,"sync");
    SYM(set,"set_u32"); SYM(put,"set_words"); SYM(get,"get_words");
    SYM(sq,"square_mul"); SYM(sub,"sub_u32"); SYM(copy,"copy");
    SYM(prepare,"prepare"); SYM(mul,"mul"); SYM(add,"add");
    SYM(subreg,"sub_reg"); SYM(equal,"equal"); SYM(profile,"profile_report");
#undef SYM
    h=create_ex ? create_ex(p,4,d,1,plan,tune,workload) : create(p,4,d,1,plan,tune);
    if(!h) throw std::runtime_error(error());
  }
  ~Api(){ if(h) destroy(h); if(lib) dlclose(lib); }
  void ok(int rc){ if(!rc) throw std::runtime_error(error()); }
};
int main(int argc,char**argv) {
  try {
    if(argc!=9 && argc!=10) throw std::runtime_error("usage: bench LIB DEVICE EXP PLAN TUNE MODE ITERS OUTPUT.bin [WORKLOAD_ID]");
    uint32_t p=std::stoul(argv[3]); unsigned iters=std::stoul(argv[7]);
    std::string mode=argv[6]; if(!iters) throw std::runtime_error("ITERS must be positive");
    uint32_t workload = argc == 10 ? static_cast<uint32_t>(std::stoul(argv[9])) : 0u;
    Api a(argv[1],p,std::stoul(argv[2]),argv[4],argv[5],workload);
    std::vector<uint32_t> seed(a.words(a.h)); uint32_t rng=0x12345678u;
    for(auto& w:seed) {rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; w=rng;}
    if(p%32) seed.back() &= (1u<<(p%32))-1u;
    auto init=[&]{a.ok(a.put(a.h,0,seed.data(),seed.size())); a.ok(a.set(a.h,1,7));
      a.ok(a.prepare(a.h,1,1)); a.ok(a.sync(a.h));};
    std::ofstream out(argv[8],std::ios::binary); if(!out) throw std::runtime_error("output open failed");
    auto snap=[&](size_t reg){ std::vector<uint32_t> v(seed.size());
      a.ok(a.get(a.h,reg,v.data(),v.size()));
      out.write(reinterpret_cast<char*>(v.data()),v.size()*sizeof(uint32_t));
      if(!out) throw std::runtime_error("snapshot write failed");};
    init();
    if(mode=="check") {
      snap(0); // Dense initial residue; Python validates this entire trace mod 2^p-1.
      a.ok(a.sq(a.h,0,1)); a.ok(a.sub(a.h,0,2)); snap(0);
      for(int i=0;i<5;++i){a.ok(a.sq(a.h,0,1));a.ok(a.sub(a.h,0,2));} snap(0);
      a.ok(a.sq(a.h,0,1)); a.ok(a.sub(a.h,0,2)); a.ok(a.sub(a.h,0,2)); snap(0);
      a.ok(a.sq(a.h,0,3)); snap(0);
      a.ok(a.sq(a.h,0,2)); snap(0);
      a.ok(a.sq(a.h,0,1)); a.ok(a.sub(a.h,0,2)); a.ok(a.copy(a.h,2,0)); snap(2);
      a.ok(a.sq(a.h,2,1)); a.ok(a.sq(a.h,0,1)); snap(0); snap(2);
      a.ok(a.prepare(a.h,1,1)); a.ok(a.mul(a.h,0,1,1)); snap(0);
      a.ok(a.add(a.h,0,0)); snap(0); // aliased add
      a.ok(a.prepare(a.h,2,0)); a.ok(a.mul(a.h,0,2,3)); snap(0);
      a.ok(a.subreg(a.h,0,2)); snap(0);
      a.ok(a.copy(a.h,3,0)); int eq=0; a.ok(a.equal(a.h,3,0,&eq));
      if(!eq) throw std::runtime_error("equal(copy) failed");
      a.ok(a.set(a.h,0,0)); a.ok(a.sq(a.h,0,1)); a.ok(a.sub(a.h,0,2)); snap(0);
      a.ok(a.set(a.h,0,4)); a.ok(a.sq(a.h,0,1)); a.ok(a.sub(a.h,0,2));
      a.ok(a.sync(a.h)); snap(0);
      std::cout<<"AEVUM_CHECK snapshots=15 words="<<seed.size()<<"\n";
      return 0;
    }
    auto step=[&]{
      a.ok(a.sq(a.h,0,mode=="mul3"?3:1));
      if(mode=="ll") a.ok(a.sub(a.h,0,2));
      if(mode=="mixed") a.ok(a.mul(a.h,0,1,1));
    };
    if(mode!="prp" && mode!="ll" && mode!="mul3" && mode!="mixed") throw std::runtime_error("bad mode");
    for(int i=0;i<64;++i) step();
    a.ok(a.sync(a.h));
    init(); a.ok(a.profile(a.h,0));
    auto start=std::chrono::steady_clock::now();
    for(unsigned i=0;i<iters;++i) step();
    a.ok(a.sync(a.h));
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    a.ok(a.profile(a.h,1)); // Outside timing; reports only measured operations.
    snap(0);
    std::cout.precision(12);
    std::cout<<"AEVUM_BENCH {\"seconds\":"<<seconds<<",\"iterations\":"<<iters
      <<",\"transform\":"<<a.transform(a.h)<<",\"words\":"<<seed.size()<<"}\n";
  } catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
