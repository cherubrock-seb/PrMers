// Host control gate for the actual graph batching header, with a recording
// driver facade. This does not compile CUDA kernels, run a graph or emulate GPU
// arithmetic/order. Hardware gate remains mandatory.
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#define CUDA_VERSION 13000
using CUresult = int;
constexpr int CUDA_SUCCESS = 0;
using CUfunction = uintptr_t;
using CUstream = int;
struct Params;
struct Node { CUfunction f; unsigned blocks, threads; std::array<uint64_t,2> args; };
struct Graph { std::vector<Node*> nodes; };
struct Exec { std::vector<Node> nodes; std::vector<Node*> identity; };
using CUgraphNode = Node*;
using CUgraph = Graph*;
using CUgraphExec = Exec*;
struct CUDA_KERNEL_NODE_PARAMS {
  CUfunction func{};
  unsigned gridDimX{}, gridDimY{}, gridDimZ{};
  unsigned blockDimX{}, blockDimY{}, blockDimZ{}, sharedMemBytes{};
  void** kernelParams{};
  void** extra{};
};
struct _cl_kernel {
  static constexpr int MAX_ARGS=32, MAX_ARG_BYTES=512;
  CUfunction func{};
  int numArgs=2;
  char argData[MAX_ARG_BYTES]{};
  size_t argOffsets[MAX_ARGS]{},argSizes[MAX_ARGS]{};
};
using cl_kernel = _cl_kernel*;
static int creates=0, updates=0, launches=0;
static bool rejectUpdate=false;
static std::vector<std::vector<Node>> submitted;
Node copyParams(const CUDA_KERNEL_NODE_PARAMS* p) {
  assert(p->gridDimY==1 && p->gridDimZ==1 && p->blockDimY==1 && p->blockDimZ==1);
  Node n{p->func,p->gridDimX,p->blockDimX,{}};
  for(int i=0;i<2;++i) std::memcpy(&n.args[i],p->kernelParams[i],8);
  return n;
}
CUresult cuGraphCreate(CUgraph* out,unsigned) { *out=new Graph; ++creates; return 0; }
CUresult cuGraphAddKernelNode(CUgraphNode* out,CUgraph g,const CUgraphNode* deps,size_t count,const CUDA_KERNEL_NODE_PARAMS* p) {
  assert(count==(g->nodes.empty()?0u:1u));
  if(count) assert(*deps==g->nodes.back());
  *out=new Node(copyParams(p)); g->nodes.push_back(*out); return 0;
}
CUresult cuGraphInstantiateWithFlags(CUgraphExec* out,CUgraph g,unsigned long long) {
  *out=new Exec;
  for(auto* n:g->nodes) { (*out)->nodes.push_back(*n); (*out)->identity.push_back(n); }
  return 0;
}
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec e,CUgraphNode n,const CUDA_KERNEL_NODE_PARAMS* p) {
  if(rejectUpdate) return 1;
  for(size_t i=0;i<e->nodes.size();++i)
    if(e->identity[i]==n) { e->nodes[i]=copyParams(p); ++updates; return 0; }
  return 1;
}
CUresult cuGraphLaunch(CUgraphExec e,CUstream) { submitted.push_back(e->nodes); ++launches; return 0; }
CUresult cuGraphExecDestroy(CUgraphExec e) { delete e; return 0; }
CUresult cuGraphDestroy(CUgraph g) { for(auto* n:g->nodes)delete n; delete g; return 0; }
#include "../src/cuda/GraphBatch.h"
void set(_cl_kernel& k, uint64_t buffer, uint64_t slot) {
  k.argOffsets[0]=0; k.argOffsets[1]=8;
  k.argSizes[0]=k.argSizes[1]=8;
  std::memcpy(k.argData,&buffer,8); std::memcpy(k.argData+8,&slot,8);
}
int main() {
  AevumCudaGraphBatch b;
  _cl_kernel middle,carry;
  middle.func=1; carry.func=2;
  b.begin();
  set(middle,100,0); b.add(&middle,4096,64);
  set(carry,100,17); b.add(&carry,1024,256);
  set(middle,999,999); set(carry,999,999); // Snapshot must outlive reuse.
  b.launch(0);
  assert(creates==1 && launches==1 && updates==0);
  assert(submitted[0][0].args[0]==100 && submitted[0][1].args[1]==17);
  b.begin();
  set(middle,200,1); b.add(&middle,4096,64);
  set(carry,200,18); b.add(&carry,1024,256);
  b.launch(0);
  assert(creates==1 && updates==2 && launches==2);
  assert(submitted[1][0].args[0]==200 && submitted[1][1].args[1]==18);
  assert(submitted[0][1].args[1]==17); // Earlier launch remains unchanged.
  b.begin(); b.add(&middle,4096,64); b.abort();
  assert(!b.active() && launches==2);
  b.begin(); carry.func=3; b.add(&carry,1024,256); b.launch(0);
  assert(creates==2 && launches==3); // ROE/non-ROE function changes topology.
  b.begin(); b.add(&middle,4096,64); carry.func=2; b.add(&carry,1024,256);
  rejectUpdate=true;
  bool rejected=false;
  try { b.launch(0); } catch(const std::runtime_error&) { rejected=true; b.abort(); }
  assert(rejected && launches==3); // Never submit stale parameters on error.
  rejectUpdate=false;
  b.begin(); middle.argOffsets[0]=512;
  rejected=false;
  try { b.add(&middle,1,64); } catch(const std::runtime_error&) { rejected=true; b.abort(); }
  assert(rejected);
  // The same kernel handle can be configured twice before graph submission.
  middle.argOffsets[0]=0;
  for(int repeat=0;repeat<2;++repeat) {
    b.begin();
    set(middle,300+repeat,30+repeat); b.add(&middle,4096,64);
    set(middle,400+repeat,40+repeat); b.add(&middle,4096,64);
    b.launch(0);
    const auto& last=submitted.back();
    assert(last[0].args[0]==uint64_t(300+repeat));
    assert(last[1].args[0]==uint64_t(400+repeat));
    assert(last[0].args[1]==uint64_t(30+repeat));
    assert(last[1].args[1]==uint64_t(40+repeat));
  }
  assert(creates==3 && launches==5);
  std::cout<<"argument snapshots, fresh slots, repeated handles, cache, edges and rejection checked\n";
}
