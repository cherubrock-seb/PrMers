// The CUDA shim runs each program, kernel, queue and event operation in that object's own context, not in the newest
// context the shim made.
//
// With several contexts alive (an engine next to an autotune context, workers, engines on other devices), every
// launch, module load, copy and wait used to make the shim's newest context current first, so a module was loaded,
// or a kernel launched, in the wrong context; and since the current context is per thread, a worker thread got the
// newest context too.  This test builds the real src/cuda/clwrap_cuda.cpp against the recording stand-in driver in
// tests/cuda_stub, which fails a launch or stream operation made with another context current, as the driver does.
#include "cuda/tinycuda.h"
#include "stub_driver.h"

#include <cstdio>
#include <cstring>
#include <thread>

using namespace cuda_stub;

static int failures = 0;
static void require(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

// The last recorded use of `what`, if any.
static const Call* lastUse(const char* what) {
  for (auto it = uses.rbegin(); it != uses.rend(); ++it) {
    if (it->what == what) return &*it;
  }
  return nullptr;
}

static cl_context newContext(cl_device_id dev) {
  int err = -1;
  cl_context c = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  require(c && err == CL_SUCCESS, "clCreateContext");
  return c;
}

// A compiled, linked program for `ctx`, as KernelCompiler::compile() makes one.
static cl_program linkedProgram(cl_context ctx, cl_device_id dev) {
  const char* src = "kernel void k() {}";
  int err = -1;
  cl_program p = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
  require(p && err == CL_SUCCESS, "clCreateProgramWithSource");
  require(clCompileProgram(p, 1, &dev, "", 0, nullptr, nullptr, nullptr, nullptr) == CL_SUCCESS, "clCompileProgram");
  cl_program linked = clLinkProgram(ctx, 1, &dev, "", 1, &p, nullptr, nullptr, &err);
  require(linked && err == CL_SUCCESS, "clLinkProgram");
  clReleaseProgram(p);
  return linked;
}

int main() {
  cl_device_id dev{};
  unsigned n = 0;
  require(clGetDeviceIDs(nullptr, 0, 1, &dev, &n) == CL_SUCCESS && n == 1, "clGetDeviceIDs");

  // The engine's context and its queue, then an autotune context made after it: the shim's newest, and current.
  cl_context c1 = newContext(dev);
  int err = -1;
  cl_command_queue q1 = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);
  require(q1 && err == CL_SUCCESS, "a queue on the first context");
  cl_mem buf = clCreateBuffer(c1, CL_MEM_READ_WRITE, 4096, nullptr, &err);
  require(buf && err == CL_SUCCESS, "a buffer on the first context");
  cl_context c2 = newContext(dev);
  require(current() == c2->ctx, "the newest context is current");

  // A program linked for the first context loads its module there, although the second is the newest.
  uses.clear();
  cl_program prog = linkedProgram(c1, dev);
  const Call* load = lastUse("cuModuleLoadDataEx");
  require(load && load->current == c1->ctx, "clLinkProgram loads the module in the program's context, not the newest");

  cuCtxSetCurrent(c2->ctx);
  cl_kernel k = clCreateKernel(prog, "k", &err);
  require(k && err == CL_SUCCESS, "clCreateKernel");
  require(k && k->func && k->func->module->owner == c1->ctx, "the kernel's function belongs to the program's context");

  // A launch on the first context's queue runs there, whichever context is current.
  cuCtxSetCurrent(c2->ctx);
  uses.clear();
  const size_t global = 256, local = 64;
  require(clEnqueueNDRangeKernel(q1, k, 1, nullptr, &global, &local, 0, nullptr, nullptr) == CL_SUCCESS,
          "a launch on the first context's queue succeeds while the second is newest");
  const Call* launch = lastUse("cuLaunchKernel");
  require(launch && launch->current == c1->ctx, "the kernel is launched with its queue's context current");

  // Copies, fills and waits on the first context's queue run there too.
  std::vector<char> host(4096);
  cuCtxSetCurrent(c2->ctx);
  require(clEnqueueWriteBuffer(q1, buf, true, 0, host.size(), host.data(), 0, nullptr, nullptr) == CL_SUCCESS,
          "clEnqueueWriteBuffer in the queue's context");
  cuCtxSetCurrent(c2->ctx);
  require(clEnqueueReadBuffer(q1, buf, true, 0, host.size(), host.data(), 0, nullptr, nullptr) == CL_SUCCESS,
          "clEnqueueReadBuffer in the queue's context");
  cuCtxSetCurrent(c2->ctx);
  require(clEnqueueCopyBuffer(q1, buf, buf, 0, 2048, 1024, 0, nullptr, nullptr) == CL_SUCCESS,
          "clEnqueueCopyBuffer in the queue's context");
  cuCtxSetCurrent(c2->ctx);
  const unsigned zero = 0;
  require(clEnqueueFillBuffer(q1, buf, &zero, sizeof(zero), 0, 4096, 0, nullptr, nullptr) == CL_SUCCESS,
          "clEnqueueFillBuffer in the queue's context");
  cuCtxSetCurrent(c2->ctx);
  require(clFinish(q1) == CL_SUCCESS, "clFinish in the queue's context");
  for (const char* what : {"cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync", "cuMemcpyDtoDAsync", "cuMemsetD32Async",
                           "cuStreamSynchronize"}) {
    const Call* use = lastUse(what);
    require(use && use->current == c1->ctx, what);
  }

  // A marker event is made in its queue's context (and destroyed there later).
  cuCtxSetCurrent(c2->ctx);
  cl_event ev = nullptr;
  require(clEnqueueMarkerWithWaitList(q1, 0, nullptr, &ev) == CL_SUCCESS && ev, "a marker on the first queue");
  require(ev && ev->owner == c1->ctx && ev->end && ev->end->owner == c1->ctx,
          "the marker's event is created in its queue's context");
  cuCtxSetCurrent(c2->ctx);
  require(clWaitForEvents(1, &ev) == CL_SUCCESS, "clWaitForEvents");
  clReleaseEvent(ev);

  // A worker thread with no context current (the current context is per thread) launches in the queue's context.
  int threadStatus = -1;
  CUcontext threadLaunchContext = nullptr;
  std::thread worker([&] {
    threadStatus = clEnqueueNDRangeKernel(q1, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    threadLaunchContext = current();
  });
  worker.join();
  require(threadStatus == CL_SUCCESS && threadLaunchContext == c1->ctx,
          "a launch from another thread runs in the queue's context, not the shim's newest");

  // A program from a cached binary, for the first context, loads its module there.
  cuCtxSetCurrent(c2->ctx);
  uses.clear();
  const unsigned char* bin = reinterpret_cast<const unsigned char*>("ptx");
  const size_t binSize = 3;
  int binStatus = -1;
  cl_program cached = clCreateProgramWithBinary(c1, 1, &dev, &binSize, &bin, &binStatus, &err);
  require(cached && err == CL_SUCCESS && binStatus == CL_SUCCESS, "clCreateProgramWithBinary");
  load = lastUse("cuModuleLoadDataEx");
  require(load && load->current == c1->ctx, "a cached binary is loaded in the program's context, not the newest");
  require(clBuildProgram(cached, 1, &dev, nullptr, nullptr, nullptr) == CL_SUCCESS, "clBuildProgram of a binary");

  // clBuildProgram of a source program loads the module in the program's context.
  const char* src = "kernel void k() {}";
  cl_program built = clCreateProgramWithSource(c1, 1, &src, nullptr, &err);
  cuCtxSetCurrent(c2->ctx);
  uses.clear();
  require(clBuildProgram(built, 1, &dev, "", nullptr, nullptr) == CL_SUCCESS, "clBuildProgram of a source");
  load = lastUse("cuModuleLoadDataEx");
  require(load && load->current == c1->ctx, "clBuildProgram loads the module in the program's context");

  // The second context's own work still runs in the second context.
  cl_command_queue q2 = clCreateCommandQueueWithProperties(c2, dev, nullptr, &err);
  cl_program prog2 = linkedProgram(c2, dev);
  cl_kernel k2 = clCreateKernel(prog2, "k", &err);
  cuCtxSetCurrent(c1->ctx);
  require(clEnqueueNDRangeKernel(q2, k2, 1, nullptr, &global, &local, 0, nullptr, nullptr) == CL_SUCCESS,
          "a launch on the second context's queue while the first is current");
  launch = lastUse("cuLaunchKernel");
  require(launch && launch->current == c2->ctx, "the second context's kernel runs in the second context");

  // After the second (autotune) context is destroyed, the first context's work goes on.
  clReleaseKernel(k2);
  clReleaseProgram(prog2);
  clReleaseCommandQueue(q2);
  clReleaseContext(c2);
  require(clEnqueueNDRangeKernel(q1, k, 1, nullptr, &global, &local, 0, nullptr, nullptr) == CL_SUCCESS,
          "the first context's kernel still launches after the second context is destroyed");

  // Work on a queue whose context is gone fails, instead of running in whichever context is current.
  cl_context c3 = newContext(dev);
  cl_command_queue q3 = clCreateCommandQueueWithProperties(c3, dev, nullptr, &err);
  clReleaseContext(c3);
  cuCtxSetCurrent(c1->ctx);
  uses.clear();
  require(clEnqueueReadBuffer(q3, buf, true, 0, 16, host.data(), 0, nullptr, nullptr) != CL_SUCCESS,
          "a read on a queue of a destroyed context fails");
  require(lastUse("cuMemcpyDtoHAsync") == nullptr, "nothing is copied for a queue of a destroyed context");
  require(clFinish(q3) != CL_SUCCESS, "clFinish on a queue of a destroyed context fails");
  clReleaseCommandQueue(q3);

  clReleaseKernel(k);
  clReleaseProgram(prog);
  clReleaseProgram(cached);
  clReleaseProgram(built);
  clReleaseMemObject(buf);
  clReleaseCommandQueue(q1);
  clReleaseContext(c1);

  if (failures) { std::printf("cuda object context test: %d FAILED\n", failures); return 1; }
  std::printf("cuda object context test PASS\n");
  return 0;
}
