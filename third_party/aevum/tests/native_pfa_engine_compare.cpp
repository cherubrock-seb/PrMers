#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using H=void*;
struct Api {
 void* so{};
 const char*(*last_error)(){};
 int(*resolve)(uint32_t,const char*,char*,size_t){};
 H(*create)(uint32_t,size_t,uint32_t,int,const char*,const char*){};
 void(*destroy)(H){};
 size_t(*tsize)(H){}; size_t(*wcount)(H){};
 int(*set_u32)(H,size_t,uint32_t){}; int(*get_words)(H,size_t,uint32_t*,size_t){};
 int(*prepare)(H,size_t,size_t){}; int(*square_mul)(H,size_t,uint32_t){}; int(*mul)(H,size_t,size_t,uint32_t){};
 template<class T> T sym(const char* n){auto p=dlsym(so,n);if(!p)throw std::runtime_error(std::string("missing symbol ")+n);return reinterpret_cast<T>(p);} 
 explicit Api(const char* path){so=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!so)throw std::runtime_error(dlerror());
#define S(x) x=sym<decltype(x)>("aevum_engine_" #x)
 S(last_error); resolve=sym<decltype(resolve)>("aevum_engine_resolve_fft"); S(create);S(destroy);
 tsize=sym<decltype(tsize)>("aevum_engine_transform_size");
 wcount=sym<decltype(wcount)>("aevum_engine_word_count");
 S(set_u32);S(get_words);S(prepare);S(square_mul);S(mul);
#undef S
 }
 ~Api(){if(so)dlclose(so);} 
};
static void ok(Api&a,int rc,const char*what){if(!rc)throw std::runtime_error(std::string(what)+": "+(a.last_error()?a.last_error():"unknown"));}
struct Run {size_t transform{}; std::string resolved; std::vector<std::vector<uint32_t>> outputs;};
static Run execute(Api&a,uint32_t p,uint32_t dev,const char*spec,unsigned iters){
 char resolved[256]{};ok(a,a.resolve(p,spec,resolved,sizeof(resolved)),"resolve");
 H h=a.create(p,2,dev,1,spec,".");if(!h)throw std::runtime_error(a.last_error());
 Run r; r.transform=a.tsize(h);r.resolved=resolved;const size_t wc=a.wcount(h);
 auto grab=[&](size_t reg){std::vector<uint32_t> v(wc);ok(a,a.get_words(h,reg,v.data(),v.size()),"get_words");r.outputs.push_back(std::move(v));};
 ok(a,a.set_u32(h,0,3),"set square seed");
 for(unsigned i=0;i<iters;++i){ok(a,a.square_mul(h,0,(i&1)?3u:1u),"square_mul");grab(0);} 
 ok(a,a.set_u32(h,0,7),"set multiplicand dst");ok(a,a.set_u32(h,1,5),"set multiplicand src");
 ok(a,a.prepare(h,1,1),"prepare");ok(a,a.mul(h,0,1,3),"mul prepared");grab(0);
 a.destroy(h); return r;
}
static uint64_t hashv(const std::vector<uint32_t>&v){uint64_t h=1469598103934665603ULL;for(uint32_t x:v){h^=x;h*=1099511628211ULL;}return h;}
int main(int argc,char**argv){try{
 if(argc<5){std::cerr<<"usage: "<<argv[0]<<" libaevum_engine.so device exponent pfa:3|pfa:9 [iters]\n";return 2;}
 Api a(argv[1]);uint32_t dev=std::stoul(argv[2]),p=std::stoul(argv[3]);unsigned iters=argc>5?std::stoul(argv[5]):1;
 Run stock=execute(a,p,dev,"",iters);Run pfa=execute(a,p,dev,argv[4],iters);
 if(stock.outputs.size()!=pfa.outputs.size())throw std::runtime_error("output count mismatch");
 for(size_t i=0;i<stock.outputs.size();++i){
   if(stock.outputs[i]!=pfa.outputs[i]){size_t j=0;while(j<stock.outputs[i].size()&&stock.outputs[i][j]==pfa.outputs[i][j])++j;
     std::cerr<<"MISMATCH output="<<i<<" word="<<j<<" stock_hash="<<std::hex<<hashv(stock.outputs[i])<<" pfa_hash="<<hashv(pfa.outputs[i])<<std::dec<<"\n";
     const size_t begin=j>3?j-3:0, end=std::min(j+5,stock.outputs[i].size());
     for(size_t k=begin;k<end;++k) std::cerr<<"  word["<<k<<"] stock="<<stock.outputs[i][k]<<" pfa="<<pfa.outputs[i][k]<<(k==j?"  <-- first":" ")<<"\n";
     return 1;}
   std::cout<<"exact output "<<i<<" OK hash=0x"<<std::hex<<hashv(stock.outputs[i])<<std::dec<<"\n";
 }
 std::cout<<"stock="<<stock.resolved<<" size="<<stock.transform<<"\n";
 std::cout<<"pfa="<<pfa.resolved<<" size="<<pfa.transform<<" reduction=x"<<double(stock.transform)/pfa.transform<<"\n";
 std::cout<<"NATIVE AEVUM PFA DIFFERENTIAL TEST PASSED\n";return 0;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}
