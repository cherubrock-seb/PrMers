// The CUDA shim's CUmodule lifetime bookkeeping (plain pointers, no CUDA needed): a module is unloaded exactly once, when
// the last program or kernel using it goes, in its own context, and never after that context was destroyed.
#include "cuda/CudaModuleLife.h"

#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

static std::mutex g_unloadsMutex;
static std::vector<std::pair<void*, void*>> g_unloads;
static void fakeUnload(void* module, void* context) {
  std::lock_guard<std::mutex> lock(g_unloadsMutex);
  g_unloads.push_back({module, context});
}

static int failures = 0;
static void require(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

static void* handle(unsigned long v) { return reinterpret_cast<void*>(v); }

int main() {
  cuda_module::Registry reg;
  void* c1 = handle(0x1000);
  void* m1 = handle(0x2000);
  void* m2 = handle(0x3000);
  reg.contextCreated(c1);

  // loadAux(): the program loads the module, a kernel is created from it, the program is released, the kernel runs on.
  {
    std::shared_ptr<void> program = reg.adopt(m1, c1, fakeUnload);
    std::shared_ptr<void> kernel = program;
    program.reset();
    require(g_unloads.empty(), "releasing the program while a kernel uses the module must not unload it");
    std::shared_ptr<void> kernel2 = kernel;
    kernel.reset();
    require(g_unloads.empty(), "a second kernel keeps the module loaded");
  }
  require(g_unloads.size() == 1 && g_unloads[0] == std::make_pair(m1, c1), "the last reference unloads the module once, in its context");
  g_unloads.clear();

  // A program released without ever creating a kernel (a failed build) unloads at once.
  { std::shared_ptr<void> program = reg.adopt(m2, c1, fakeUnload); }
  require(g_unloads.size() == 1 && g_unloads[0].first == m2, "an unused program's module is unloaded");
  g_unloads.clear();

  // The PTX register-limit path swaps the program's module before any kernel exists: the old one goes, the new one stays.
  {
    std::shared_ptr<void> program = reg.adopt(m1, c1, fakeUnload);
    program = reg.adopt(m2, c1, fakeUnload);
    require(g_unloads.size() == 1 && g_unloads[0].first == m1, "swapping the module unloads the replaced one");
    std::shared_ptr<void> kernel = program;
    program.reset();
    require(g_unloads.size() == 1, "the replacement stays loaded for its kernel");
  }
  require(g_unloads.size() == 2 && g_unloads[1].first == m2, "the replacement is unloaded with its last kernel");
  g_unloads.clear();

  // The context is destroyed (which unloads its modules) before the last kernel goes: no second unload, and a newer
  // module that reuses the same handle in a newer context that reuses the same context handle is not touched by it.
  {
    std::shared_ptr<void> stale = reg.adopt(m1, c1, fakeUnload);
    reg.contextDestroying(c1);
    reg.contextCreated(c1);
    std::shared_ptr<void> fresh = reg.adopt(m1, c1, fakeUnload);
    stale.reset();
    require(g_unloads.empty(), "a module of a destroyed context must not be unloaded");
    fresh.reset();
    require(g_unloads.size() == 1 && g_unloads[0] == std::make_pair(m1, c1), "the newer module is unloaded normally");
  }
  g_unloads.clear();

  // A module of a context the shim did not create is left alone.
  { std::shared_ptr<void> foreign = reg.adopt(m1, handle(0x9000), fakeUnload); }
  require(g_unloads.empty(), "a module of an unknown context is never unloaded");

  // Worker threads building and dropping programs and kernels at the same time: every module is unloaded exactly once.
  {
    const int threads = 8, perThread = 2000;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&reg, c1, t] {
        for (int i = 0; i < perThread; ++i) {
          std::shared_ptr<void> program = reg.adopt(handle(0x100000ul * (t + 1) + i + 1), c1, fakeUnload);
          std::vector<std::shared_ptr<void>> kernels(3, program);
          program.reset();
        }
      });
    }
    for (auto& th : pool) th.join();
    require(g_unloads.size() == size_t(threads) * perThread, "concurrent use: one unload per module");
  }

  if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
  std::printf("cuda module lifetime test ok\n");
  return 0;
}
