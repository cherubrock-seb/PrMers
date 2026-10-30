// Lifetime of the CUmodules the CUDA shim loads.
//
// In OpenCL, clCreateKernel retains the program, so the code stays loaded until the program AND every kernel created
// from it are released.  KernelCompiler::loadAux() relies on that: it creates a kernel, then releases the program while
// the kernel keeps running.  So the shim cannot unload a module in clReleaseProgram, and it used to never unload one:
// every Gpu built (each work unit in the CLI, each autotune candidate) left its ~40 modules loaded in the context until
// the context was destroyed, and a long-lived context (the standalone CLI, -tune) grew until CUDA_ERROR_OUT_OF_MEMORY.
//
// Instead the program and every kernel created from it hold a shared reference to the module, and the last one to go
// unloads it -- with the module's own context made current, and only while that context is alive: cuCtxDestroy has
// already unloaded the modules of a destroyed context, and by then the same handle value may name a newer module of a
// newer context.
//
// Plain pointers and no CUDA includes, so the bookkeeping is tested on a host without CUDA
// (tests/cuda_module_life_test.cpp).
#pragma once

#include <map>
#include <memory>
#include <mutex>

namespace cuda_module {

class Registry {
public:
  using UnloadFn = void (*)(void* module, void* context);

  // A context the shim created, and the same context just before it is destroyed.
  void contextCreated(void* context) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_[context] = std::make_shared<bool>(true);
  }
  void contextDestroying(void* context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto it = live_.find(context); it != live_.end()) {
      *it->second = false;
      live_.erase(it);
    }
  }

  // A reference to a module just loaded in `context`.  Copy it into every program and kernel that uses the module; when
  // the last copy goes, `unload(module, context)` runs if the context is still alive.  A module of a context the shim did
  // not create is never unloaded (as before).
  std::shared_ptr<void> adopt(void* module, void* context, UnloadFn unload) {
    std::shared_ptr<bool> alive;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (auto it = live_.find(context); it != live_.end()) { alive = it->second; }
    }
    return std::shared_ptr<void>(module, [this, context, alive, unload](void* m) {
      // Under the lock, so that a context cannot be destroyed between the liveness check and the unload.
      std::lock_guard<std::mutex> lock(mutex_);
      if (m && alive && *alive) { unload(m, context); }
    });
  }

private:
  std::mutex mutex_;
  std::map<void*, std::shared_ptr<bool>> live_;
};

}  // namespace cuda_module
