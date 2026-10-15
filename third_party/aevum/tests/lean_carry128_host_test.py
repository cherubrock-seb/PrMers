#!/usr/bin/env python3
"""Compile actual candidate CRT/carry fragments against independent host arithmetic.

This is an algebra/source gate, not a GPU, compiler-resource, ROE or residue gate.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get("CXX") or shutil.which("g++")
if not CXX:
    raise SystemExit("g++ or CXX required")
source = (ROOT / "src/cl/carryutil.cl").read_text()

def function(signature):
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]

begin = source.index("#if AEVUM_EXPERIMENT_LEAN_CARRY128\n  // PRPLL")
end = source.index("\n#else\n  // Compute x =", begin)
crt = source[begin:end] + "\n#endif"
functions = "\n".join(function(signature) for signature in (
    "i64 aevumCarry128Window(",
    "Word OVERLOAD carryStep(i128",
    "Word OVERLOAD carryStepUnsignedSloppy(i128",
    "Word OVERLOAD carryStepSignedSloppy(i128",
))
HARNESS = r'''
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
using u32=uint32_t;using i32=int32_t;using u64=uint64_t;using i64=int64_t;
using Word=i64;
struct i128 {unsigned __int128 bits;};
u32 lo32(u64 x){return x;}u32 hi32(u64 x){return x>>32;}
i64 make_i64(i32 hi,u32 lo){return (i64)((u64(u32(hi))<<32)|lo);}
i128 make_i128(i64 hi,u64 lo){return{(unsigned __int128(u64(hi))<<64)|lo};}
i128 add(i128 a,i64 b){return{a.bits+(unsigned __int128)(__int128)b};}
i128 add(i128 a,u64 b){return{a.bits+b};}
i128 sub(i128 a,i64 b){return{a.bits-(unsigned __int128)(__int128)b};}
u64 i128_lo64(i128 x){return x.bits;}
u64 i128_hi64(i128 x){return x.bits>>64;}
u64 i128_shrlo64(i128 x,u32 n){return (__int128)x.bits>>n;}
i128 i128_masklo64(i128 x,u64 mask){return{(x.bits&~(unsigned __int128)UINT64_MAX)|(u64(x.bits)&mask)};}
i64 lowBits(u64 x,u32 n){u64 sign=u64(1)<<(n-1);return i64((x&((sign<<1)-1))^sign)-i64(sign);}
i32 lowBits(i32 x,u32 n){u32 sign=u32(1)<<(n-1);return i32((u32(x)&((sign<<1)-1))^sign)-i32(sign);}
u64 ulowFixedBits(u64 x,u32 n){return x&((u64(1)<<n)-1);}
i32 xtract32(i64 x,u32 n){return u64(x)>>n;}
u32 bitlen(bool big){return EXP+big;}
#define OVERLOAD
#define AEVUM_EXPERIMENT_LEAN_CARRY128 1
#define NWORDS 1
#define MAXBPW 10000
#define SLOPPY_MAXBPW 9890
#define ACTUAL_BPW (EXP*100)
'''
CHECK = r'''
int main(){
 std::mt19937_64 rng(128+EXP+MUL3);
 const __int128 m31=(__int128(1)<<31)-1,m61=(__int128(1)<<61)-1;
 const i32 edges[]={-(1<<22),-(1<<21),-1,0,1,(1<<21)-1,(1<<22)-1};
 for(int i=0;i<4096;++i){
  i32 nF2=i<28?edges[i%7]:i32(rng()%(1<<23))-(1<<22);
  u64 n61=i<28?(i%4==0?0:i%4==1?1:i%4==2?u64(m61)-1:u64(m61)):rng()%u64(m61);
  u32 n31=i%17==0?u32(m31):u32(rng()%u64(m31));
  auto got=assemble(nF2,n61,n31).bits;
  auto want=(unsigned __int128)(((__int128)nF2*m31*m61+(__int128)n61*m31+n31)*(MUL3?3:1));
  if(got!=want)throw std::runtime_error("CRT mismatch");
 }
 for(int i=0;i<4096;++i){
  // Wide signed values with carries inside i64 for this word size.
  i64 high=i64(rng()&((u64(1)<<(EXP-2))-1));
  if(i&1)high=-high;
  u64 lo=i<64?(u64(1)<<i):rng();
  i128 x=make_i128(high,lo);
  for(bool big:{false,true}){
   u32 n=EXP+big;u64 sign=u64(1)<<(EXP);
   i64 carry;
   auto w=carryStep(x,&carry,big);
   auto expectedWord=lowBits(lo,n);
   u64 expectedCarry=u64((__int128)x.bits>>n)+(expectedWord<0);
   if(w!=expectedWord||u64(carry)!=expectedCarry)throw std::runtime_error("exact carry mismatch");
   w=carryStepUnsignedSloppy(x,&carry,big);
   auto masked=i128_masklo64(x,~sign);
   if(u64(w)!=(lo&((sign<<1)-1))||u64(carry)!=u64((__int128)masked.bits>>n))
     throw std::runtime_error("unsigned sloppy mismatch");
   w=carryStepSignedSloppy(x,&carry,big);
   auto top=lo&sign;
   auto expectedSloppy=i64(lo&(sign-1))-i64(top);
   expectedCarry=u64(((__int128)x.bits+top)>>n);
   if(w!=expectedSloppy||u64(carry)!=expectedCarry)throw std::runtime_error("signed sloppy mismatch");
  }
 }
 std::cout<<"word bits "<<EXP<<"/"<<EXP+1<<", MUL3="<<MUL3<<": 4096 CRT and 8192 carry cases checked\n";
}
'''
# A type-id cast with two words needs a C-style cast in C++.
HARNESS = HARNESS.replace("unsigned __int128(u64(hi))", "(unsigned __int128)u64(hi)")
with tempfile.TemporaryDirectory(prefix="lean-carry128-") as temp:
    temp = Path(temp)
    unit = temp / "check.cpp"
    unit.write_text(HARNESS + "\ni128 assemble(i32 nF2,u64 n61,u32 n31){\n" +
                    crt + "\nreturn v;\n}\n" + functions + CHECK)
    for bits, mul3 in ((31,0),(32,1),(33,0),(47,1),(55,0)):
        subprocess.run([CXX,"-std=c++17","-O2",f"-DEXP={bits}",f"-DMUL3={mul3}",
                        str(unit),"-o",str(temp/"check")], check=True)
        subprocess.run([str(temp/"check")], check=True)
