#!/usr/bin/env python3
"""Run the actual exchange with cooperative host lanes and real memory guards.

Only the return type and barrier primitive are adapted to C++ coroutines; the
address expressions, memory operations and loops come from the kernel source.
This tests the permutation/partition protocol, not AMD codegen or GPU fences.
"""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "src/cl/gf61_paired_lds.cl").read_text()
assert source.count("void aevumGf61PairedLds(") == 1
source = source.replace("void aevumGf61PairedLds(", "Task aevumGf61PairedLds(")
prefix = r'''
#include <algorithm>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <vector>
using u32=uint32_t;
using u64=uint64_t;
struct GF61 { u64 x,y; bool operator==(const GF61&) const = default; };
#define local
#define NTT_GF61 1
#define WG 64
#define RADIX 8
#define SHUFL_BYTES 8
#define DOING_WIDTH 1
#define DOING_HEIGHT 0
#define AEVUM_GF61_PAIRED_LDS_W 1
#define AEVUM_GF61_PAIRED_LDS_H 0
#define HAS_ASM 0
u32 stride, lane;
#define LDS_BYTES stride
u32 get_local_id(int) { return lane; }
#define bar(n) co_await std::suspend_always{}
struct Task {
  struct promise_type {
    Task get_return_object() { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() { std::terminate(); }
  };
  std::coroutine_handle<promise_type> h;
};
'''
checks = r'''
int main() {
  constexpr u64 sentinel=0xcafebabedead0123ull;
  constexpr size_t guard=32;
  std::mt19937_64 random(0x4fe779ca);
  unsigned cases=0;
  for (u32 bytes : {4096u,4544u}) for (u32 groups : {1u,2u,3u,4u,8u}) {
    stride=bytes;
    std::vector<u64> lds(guard+groups*stride/8+guard,sentinel);
    auto* base=reinterpret_cast<GF61*>(lds.data()+guard);
    for (unsigned pattern=0;pattern<3;++pattern) {
      std::vector<GF61> input(groups*64*8),expected(input.size());
      for (size_t i=0;i<input.size();++i) {
        input[i]=pattern==0?GF61{u64(i),~u64(i)}:
                 pattern==1?GF61{random(),random()}:GF61{u64(1)<<(i%64),~(u64(1)<<(i%64))};
      }
      // Repeat without clearing LDS: models reuse of the same allocation.
      for (u32 f : {1u,8u,1u,8u}) {
        auto actual=input;
        std::vector<GF61> canonical(512);
        for (u32 g=0;g<groups;++g) {
          const u32 start=g*512;
          for (u32 me=0;me<64;++me) for (u32 i=0;i<8;++i)
            canonical[i*f+(me&~(f-1))*8+(me&(f-1))]=input[start+me*8+i];
          for (u32 me=0;me<64;++me) for (u32 i=0;i<8;++i)
            expected[start+me*8+i]=canonical[i*64+me];
        }
        std::vector<Task> tasks;
        for (u32 me=0;me<groups*64;++me)
          tasks.push_back(aevumGf61PairedLds(base,actual.data()+me*8,f,groups,me%64));
        // Five resumptions: four barriers, then completion. Reverse lane
        // order on alternating phases to avoid relying on a sequential order.
        for (u32 phase=0;phase<5;++phase) {
          for (u32 j=0;j<tasks.size();++j) {
            lane=(phase&1)?u32(tasks.size()-1-j):j;
            assert(!tasks[lane].h.done()); tasks[lane].h.resume();
            assert(tasks[lane].h.done()==(phase==4));
          }
        }
        for (auto task:tasks) task.h.destroy();
        assert(actual==expected);
        for (size_t i=0;i<guard;++i) {
          assert(lds[i]==sentinel);
          assert(lds[lds.size()-1-i]==sentinel);
        }
        // Each transform must leave its own allocation padding untouched.
        for (u32 g=0;g<groups;++g) for (u32 i=512;i<stride/8;++i)
          assert(lds[guard+g*stride/8+i]==sentinel);
        input=actual; ++cases;
      }
    }
  }
  std::cout<<cases<<" actual-kernel permutation/partition/barrier cases: PASS\n";
}
'''
with tempfile.TemporaryDirectory(prefix="gf61-paired-lds-") as tmp:
    cpp = Path(tmp) / "test.cpp"
    cpp.write_text(prefix + source + checks)
    out = Path(tmp) / "test"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all", str(cpp), "-o", str(out)], check=True)
    subprocess.run([str(out)], check=True)
