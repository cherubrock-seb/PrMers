#pragma once
// Include after tinycuda.h. This batches native launch calls without CUDA
// stream capture: host slot allocation still runs on every square.
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class AevumCudaGraphBatch {
  struct Launch {
    CUfunction function{};
    unsigned blocks{}, threads{};
    int count{};
    alignas(16) std::array<char, _cl_kernel::MAX_ARG_BYTES> data{};
    std::array<size_t, _cl_kernel::MAX_ARGS> offsets{};
    bool sameShape(const Launch& other) const {
      return function == other.function && blocks == other.blocks &&
             threads == other.threads && count == other.count;
    }
    CUDA_KERNEL_NODE_PARAMS params(std::array<void*, _cl_kernel::MAX_ARGS>& pointers) {
      for (int i = 0; i < count; ++i) pointers[i] = data.data() + offsets[i];
      CUDA_KERNEL_NODE_PARAMS p{};
      p.func = function;
      p.gridDimX = blocks; p.gridDimY = p.gridDimZ = 1;
      p.blockDimX = threads; p.blockDimY = p.blockDimZ = 1;
      p.kernelParams = pointers.data();
      return p;
    }
  };
  struct Cached {
    CUgraph graph{};
    CUgraphExec exec{};
    std::vector<CUgraphNode> nodes;
    std::vector<Launch> shape;
    ~Cached() {
      if (exec) cuGraphExecDestroy(exec);
      if (graph) cuGraphDestroy(graph);
    }
  };
  bool active_{false};
  std::vector<Launch> pending_;
  std::vector<std::unique_ptr<Cached>> cache_;
  static void check(CUresult r) {
    if (r != CUDA_SUCCESS)
      throw std::runtime_error("CUDA square graph API failure: " + std::to_string(int(r)));
  }
public:
  AevumCudaGraphBatch() { pending_.reserve(16); cache_.reserve(2); }
  bool active() const { return active_; }
  void begin() {
    if (active_) throw std::runtime_error("nested CUDA square graph");
    pending_.clear();
    active_ = true;
  }
  void abort() noexcept { active_ = false; pending_.clear(); }
  void add(cl_kernel k, unsigned blocks, unsigned threads) {
    if (!active_ || pending_.size() >= 16 || !blocks || !threads ||
        k->numArgs < 0 || k->numArgs > _cl_kernel::MAX_ARGS)
      throw std::runtime_error("unsupported CUDA square graph launch");
    Launch launch;
    launch.function = k->func;
    launch.blocks = blocks;
    launch.threads = threads;
    launch.count = k->numArgs;
    std::memcpy(launch.data.data(), k->argData, launch.data.size());
    for (int i = 0; i < launch.count; ++i) {
      if (k->argOffsets[i] >= launch.data.size() ||
          k->argSizes[i] > launch.data.size() - k->argOffsets[i])
        throw std::runtime_error("invalid CUDA square graph argument span");
      launch.offsets[i] = k->argOffsets[i];
    }
    pending_.push_back(launch); // Snapshot now: later setArg calls may reuse k.
  }
  void launch(CUstream stream) {
    if (!active_ || pending_.empty()) throw std::runtime_error("empty CUDA square graph");
    active_ = false;
    Cached* selected = nullptr;
    for (auto& cached : cache_) {
      if (cached->shape.size() != pending_.size()) continue;
      bool same = true;
      for (size_t i = 0; i < pending_.size(); ++i)
        same = same && pending_[i].sameShape(cached->shape[i]);
      if (same) { selected = cached.get(); break; }
    }
    if (!selected) {
      if (cache_.size() >= 8) throw std::runtime_error("CUDA square graph topology limit");
      auto fresh = std::make_unique<Cached>();
      check(cuGraphCreate(&fresh->graph, 0));
      for (auto& item : pending_) {
        std::array<void*, _cl_kernel::MAX_ARGS> pointers{};
        auto p = item.params(pointers);
        CUgraphNode node{};
        const auto previous = fresh->nodes.empty() ? nullptr : &fresh->nodes.back();
        check(cuGraphAddKernelNode(&node, fresh->graph, previous, previous ? 1 : 0, &p));
        fresh->nodes.push_back(node); // Explicit serial edge, same as one stream.
      }
#if CUDA_VERSION >= 11040
      check(cuGraphInstantiateWithFlags(&fresh->exec, fresh->graph, 0));
#else
      check(cuGraphInstantiate(&fresh->exec, fresh->graph, nullptr, nullptr, 0));
#endif
      fresh->shape = pending_;
      selected = fresh.get();
      cache_.push_back(std::move(fresh));
    } else {
      // Every parameter of every node is refreshed, including all buffers,
      // ROE positions and carry sample slots. Never replay frozen arguments.
      for (size_t i = 0; i < pending_.size(); ++i) {
        std::array<void*, _cl_kernel::MAX_ARGS> pointers{};
        auto p = pending_[i].params(pointers);
        check(cuGraphExecKernelNodeSetParams(selected->exec, selected->nodes[i], &p));
      }
    }
    check(cuGraphLaunch(selected->exec, stream));
    pending_.clear();
  }
};
