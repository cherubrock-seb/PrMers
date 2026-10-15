#!/usr/bin/env python3
"""Check actual 2K padded LDS index helpers against the exchange contract.

This is a host permutation/bounds gate, not GPU compilation or FFT arithmetic.
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
source = (ROOT / "src/cl/fftbase.cl").read_text()

def function(signature):
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]

helpers = "\n".join(function(sig) for sig in (
    "u32 aevum2KWriteIndex(", "u32 aevum2KReadIndex("))
harness = r'''
#include <cstdint>
#include <iostream>
#include <stdexcept>
using u32=uint32_t;
'''
check = r'''
int main() {
 for(u32 f : {1u,16u}) {
  int lds[2174];for(auto& x:lds)x=-1;
  u32 maxIndex=0;
  for(u32 lane=0;lane<256;++lane)for(u32 i=0;i<8;++i){
   // Logical producer layout: fft8 outputs or half-radix16 outputs.
   u32 logical=f==1 ? lane*8+i :
      ((lane%128)/8)*128+i*16+(lane/128)*8+(lane&7);
   u32 address=aevum2KWriteIndex(i,f,lane);
   if(address>=2174||lds[address]!=-1)throw std::runtime_error("LDS overflow/producer collision");
   lds[address]=logical;
   if(address>maxIndex)maxIndex=address;
  }
  for(u32 lane=0;lane<256;++lane)for(u32 i=0;i<16;++i){
   u32 address=aevum2KReadIndex(i,f,lane);
   if(address>=2174||lds[address]!=int(i*128+lane%128))
    throw std::runtime_error("exchange consumer pairing mismatch");
  }
  std::cout<<"f="<<f<<": 2048 unique writes, 4096 consumer reads, max padded index "<<maxIndex<<"\n";
 }
}
'''
with tempfile.TemporaryDirectory(prefix="width2k-layout-") as temp:
    temp = Path(temp)
    unit = temp / "check.cpp"
    unit.write_text(harness + helpers + check)
    subprocess.run([CXX,"-std=c++17","-O2",str(unit),"-o",str(temp/"check")], check=True)
    subprocess.run([str(temp/"check")], check=True)
