#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using u64=uint64_t; using u32=uint32_t;
#include "../src/cl/gf61limb.cl"
int main() {
  constexpr u64 m=(1ULL<<61)-1;
  auto check=[](u64 a,u64 b) {
    const u64 expected=static_cast<u64>((static_cast<__uint128_t>(a)*b)%m);
    if(aevumMul61(a,b)!=expected) throw std::runtime_error("limb product mismatch");
  };
  std::vector<u64> edge={0,1,2,m-1,m,m+1,2*m-1,2*m,2*m+1,~0ULL};
  for(int i=1;i<64;++i){edge.push_back(1ULL<<i);edge.push_back((1ULL<<i)-1);}
  for(u64 a:edge) for(u64 b:edge) check(a,b);
  u64 seed=0xfedcba9876543210ULL;
  auto next=[&] {seed^=seed<<13; seed^=seed>>7; seed^=seed<<17;return seed;};
  for(int i=0;i<1000000;++i) {
    const u64 a=next(),b=next(); check(a,b);
    const u64 ax=a%m,ay=b%m,bx=next()%m,by=next()%m;
    const u64 p=aevumMul61Canonical(ax,bx),q=aevumMul61Canonical(ay,by);
    const u64 r=aevumMul61Canonical(aevumCanonical61(ax+ay),aevumCanonical61(bx+by));
    const u64 real=static_cast<u64>((static_cast<__uint128_t>(ax)*bx+
                                  static_cast<__uint128_t>(m)*m-static_cast<__uint128_t>(ay)*by)%m);
    const u64 imag=static_cast<u64>((static_cast<__uint128_t>(ax)*by+static_cast<__uint128_t>(ay)*bx)%m);
    if(aevumCmul61Real(p,q)!=real||aevumCmul61Imag(p,q,r)!=imag)
      throw std::runtime_error("complex product mismatch");
  }
  std::cout<<"GF61 limb products: adversarial boundaries and 1,000,000 random scalar/complex cases PASS\n";
}
