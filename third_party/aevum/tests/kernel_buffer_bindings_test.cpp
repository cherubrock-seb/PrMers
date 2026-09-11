#include "../src/KernelBufferBindings.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

static std::unordered_map<cl_mem,int> refs;
static int sets=0;
static bool fail=false;
extern "C" int clRetainMemObject(cl_mem m) {++refs[m];return 0;}
extern "C" int clReleaseMemObject(cl_mem m) {assert(refs[m]>0);--refs[m];return 0;}
extern "C" int clSetKernelArg(cl_kernel,unsigned,size_t,const void*) {++sets;return fail?-1:0;}
void check(int err,const char*,int,const char*,string_view) {if(err) throw std::runtime_error("mock set failure");}
int main() {
  auto a=reinterpret_cast<cl_mem>(1),b=reinterpret_cast<cl_mem>(2);
  auto k=reinterpret_cast<cl_kernel>(3);
  {
    KernelBufferBindings cache;
    cache.bind(k,0,a,"k"); cache.bind(k,0,a,"k");
    assert(sets==1 && refs[a]==1);
    fail=true;
    try {cache.bind(k,0,b,"k"); assert(false);} catch(const std::runtime_error&){}
    fail=false;
    assert(refs[a]==1 && refs[b]==0);
    cache.bind(k,0,a,"k"); assert(sets==2); // failed rebind kept previous cache
    cache.bind(k,0,b,"k"); assert(refs[a]==0 && refs[b]==1);
    cache.bind(k,1,a,"k"); assert(refs[a]==1); // independent argument slot
    cache.bind(k,0,nullptr,"k"); assert(refs[b]==0);
    cache.bind(k,0,a,"k"); assert(refs[a]==2);
    cache.bind(k,32,b,"k"); assert(refs[b]==0); // uncached bounds-safe fallback
  }
  assert(refs[a]==0 && refs[b]==0);
  std::cout<<"Kernel buffer binding cache: rebinding, failure and lifetime tests passed\n";
}
