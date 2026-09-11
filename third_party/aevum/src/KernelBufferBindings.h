// OpenCL-only argument cache. Kernel signatures fix the type of each position.
#pragma once
#include "clwrap.h"
#include <array>

class KernelBufferBindings {
  std::array<cl_mem,32> buffers{};
public:
  KernelBufferBindings() = default;
  KernelBufferBindings(const KernelBufferBindings&) = delete;
  KernelBufferBindings& operator=(const KernelBufferBindings&) = delete;
  ~KernelBufferBindings() {
    for (cl_mem mem : buffers) if (mem) clReleaseMemObject(mem);
  }
  void bind(cl_kernel kernel, int pos, cl_mem arg, const string& name) {
    if (pos < 0 || static_cast<size_t>(pos) >= buffers.size()) {
      ::setArg(kernel,pos,arg,name);
      return;
    }
    cl_mem& old=buffers[pos];
    if (old && old==arg) return;
    // Retention prevents a freed temporary from reusing the cached handle.
    if (arg) CHECK1(clRetainMemObject(arg));
    try { ::setArg(kernel,pos,arg,name); }
    catch (...) { if (arg) clReleaseMemObject(arg); throw; }
    if (old) clReleaseMemObject(old);
    old=arg;
  }
};
