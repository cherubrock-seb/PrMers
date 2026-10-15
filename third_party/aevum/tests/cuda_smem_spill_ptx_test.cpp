// Host-only PTX rewriting gate; no CUDA SDK/GPU or JIT emulation.
#include "../src/cuda/SmemSpillPtx.h"
#include <cassert>
#include <iostream>

int main() {
  const std::string ptx = R"(.version 9.0
.target sm_86
.address_size 64
.visible .entry tailSquareGF610()
.maxntid 64,1,1
{ ret; }
.visible .entry tailSquareGF61 (
 .param .u64 p0
)
.maxnreg 96
.maxntid 64,1,1
{
 .reg .b32 r;
 { mov.b32 r, 0; }
 ret;
}
.visible .entry carryFused()
.maxntid 256,1,1
.minnctapersm 2
{ ret; }
)";
  std::string candidate;
  assert(aevumSmemSpillPtx(ptx, "tailSquareGF61", candidate));
  const auto body = ptx.find("{\n .reg");
  const std::string pragma = "\n.pragma \"enable_smem_spilling\";";
  std::string expected = ptx;
  expected.insert(body + 1, pragma);
  assert(candidate == expected); // All other entries/limits byte-for-byte.
  assert(ptx.find("enable_smem_spilling") == std::string::npos);
  std::string again;
  assert(aevumSmemSpillPtx(candidate, "tailSquareGF61", again));
  assert(again == candidate);
  assert(!aevumSmemSpillPtx(ptx, "tailSquareGF6", candidate));
  assert(!aevumSmemSpillPtx(ptx, "missing", candidate));
  std::string old = ptx;
  old.replace(old.find("9.0"), 3, "8.8");
  assert(!aevumSmemSpillPtx(old, "tailSquareGF61", candidate));
  assert(aevumSmemSpillPtx(ptx, "carryFused", candidate));
  assert(candidate.find(".minnctapersm 2") != std::string::npos);
  assert(!aevumSmemSpillPtx(".version 9.0\n.entry x(){ret;}", "x", candidate));
  assert(!aevumSmemSpillPtx(".version 9.0\n.entry x()\n.maxntid 64;", "x", candidate));
  assert(!aevumSmemSpillPtx(".version 9.0\n.entry x()\n.maxntid 64\n{", "x", candidate));
  assert(!aevumSmemSpillPtx(".version 9.0\n.entry x(", "x", candidate));
  std::cout << "named-entry isolation, limits, version, idempotence and malformed input checked\n";
}
