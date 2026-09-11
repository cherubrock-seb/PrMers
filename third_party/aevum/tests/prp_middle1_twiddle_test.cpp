#include "../src/common.h"
#include <iostream>
#include <stdexcept>
#include <cstdint>

std::vector<uint2> genMiddleTrigGF31(u32,u32,u32);
std::vector<ulong2> genMiddleTrigGF61(u32,u32,u32);
using Wide=unsigned __int128;
using Pair=std::pair<uint64_t,uint64_t>;
static Pair mul(Pair a,Pair b,uint64_t p) {
  uint64_t ac=(Wide(a.first)*b.first)%p,bd=(Wide(a.second)*b.second)%p;
  return {(ac+p-bd)%p,uint64_t((Wide(a.first)*b.second+Wide(a.second)*b.first)%p)};
}
static Pair power(Pair a,uint64_t n,uint64_t p) {
  Pair r={1,0}; for(;n;n>>=1,a=mul(a,a,p)) if(n&1)r=mul(r,a,p); return r;
}
template<class T> void check(const std::vector<T>& tab,u32 w,u32 h,Pair generator,unsigned order,unsigned bits) {
  uint64_t p=(uint64_t(1)<<bits)-1;
  if(tab.size()!=w+h)throw std::runtime_error("incomplete middle-one twiddle table");
  Pair rw=power(generator,(uint64_t(1)<<order)/w,p);
  Pair rn=power(generator,(uint64_t(1)<<order)/(uint64_t(w)*h),p);
  for(u32 i=0;i<w+h;++i) {
    Pair expected=i<w?power(rw,i,p):power(rn,i-w,p);
    if(Pair(tab[i].first,tab[i].second)!=expected)throw std::runtime_error("twiddle differs from independent modular oracle");
  }
  if(power(rn,uint64_t(w)*h,p)!=Pair(1,0))throw std::runtime_error("root order mismatch");
}
int main() {
  for(auto [w,h]:{std::pair{256u,256u},{512u,512u},{1024u,256u},{4096u,512u}}) {
    check(genMiddleTrigGF31(h,1,w),w,h,{7735,748621},32,31);
    check(genMiddleTrigGF61(h,1,w),w,h,{264036120304204ull,4677669021635377ull},62,61);
  }
  std::cout<<"PASS: middle-one GF31/GF61 twiddles match independent uint128 modular oracle\n";
}
