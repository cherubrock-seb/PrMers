// The CUDA shim releases each object in the context that owns it, not in whichever context it made current last.
//
// With several contexts alive (several devices or workers, autotune contexts, engines created and destroyed through
// the engine API), clReleaseMemObject used to make the newest context current and cuMemFree there, ignoring the
// result.  This test builds the real src/cuda/clwrap_cuda.cpp against the recording stand-in driver in
// tests/cuda_stub: no GPU and no CUDA install needed.
#include "cuda/tinycuda.h"
#include "stub_driver.h"

#include <cstdio>
#include <cstring>

using namespace cuda_stub;

const char* aevumCudaLastError();

// ---- The test ----

static int failures = 0;
static void require(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

// The single free/destroy recorded since the last clear, if it is `what`.
static const Call* only(const char* what) {
  return calls.size() == 1 && calls[0].what == what ? &calls[0] : nullptr;
}

static cl_context newContext(cl_device_id dev) {
  int err = -1;
  cl_context c = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  require(c && err == CL_SUCCESS, "clCreateContext");
  return c;
}

int main() {
  cl_device_id dev{};
  unsigned n = 0;
  require(clGetDeviceIDs(nullptr, 0, 1, &dev, &n) == CL_SUCCESS && n == 1, "clGetDeviceIDs");

  // An engine's context, then a second one (another engine, worker or an autotune context) made after it: the shim's
  // newest, and current.
  cl_context c1 = newContext(dev);
  int err = -1;
  cl_mem buf = clCreateBuffer(c1, CL_MEM_READ_WRITE, 4096, nullptr, &err);
  require(buf && err == CL_SUCCESS, "clCreateBuffer in the first context");
  cl_context c2 = newContext(dev);
  require(current() == c2->ctx, "the newest context is current");

  // A buffer of the first context, allocated while the second is the shim's newest, lives in the first.
  cl_mem buf2 = clCreateBuffer(c1, CL_MEM_READ_WRITE, 4096, nullptr, &err);
  require(buf2 && err == CL_SUCCESS, "clCreateBuffer in the first context while the second is newest");
  require(buf2 && allocations.count(buf2->ptr) && allocations[buf2->ptr] == c1->ctx,
          "a buffer is allocated in the context it is created for, not the shim's newest");

  // Releasing the first context's buffer frees it with the first context current, and leaves the caller's current.
  calls.clear();
  require(clReleaseMemObject(buf) == CL_SUCCESS, "clReleaseMemObject of the first context's buffer succeeds");
  const Call* free1 = only("cuMemFree");
  require(free1 && free1->current == c1->ctx, "the buffer is freed with its own context current, not the newest");
  require(current() == c2->ctx, "releasing a buffer restores the caller's current context");

  // A failing cuMemFree is reported, not swallowed; the caller's context is still restored.
  calls.clear();
  failNext["cuMemFree"] = CUDA_ERROR_UNKNOWN;
  require(clReleaseMemObject(buf2) != CL_SUCCESS, "a cuMemFree error is returned by clReleaseMemObject");
  require(aevumCudaLastError() && std::strstr(aevumCudaLastError(), "(999)"), "the CUDA error is kept for the message");
  require(only("cuMemFree") != nullptr, "the failing release tried exactly one free");
  require(current() == c2->ctx, "a failed release restores the caller's current context");

  require(clReleaseMemObject(nullptr) == CL_SUCCESS, "releasing a null buffer is a no-op");

  // A command queue's stream is destroyed in the queue's context.
  cl_command_queue q = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);
  require(q && err == CL_SUCCESS, "clCreateCommandQueueWithProperties on the first context");
  cuCtxSetCurrent(c2->ctx);   // as ensureContextCurrent() leaves it: the newest
  calls.clear();
  require(clReleaseCommandQueue(q) == CL_SUCCESS, "clReleaseCommandQueue of the first context's queue succeeds");
  const Call* stream = only("cuStreamDestroy");
  require(stream && stream->current == c1->ctx, "the stream is destroyed with its own context current");
  require(current() == c2->ctx, "releasing a queue restores the caller's current context");

  // An event is destroyed in the context it was made in.
  cl_command_queue q1 = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);   // leaves c1 current
  cl_event ev = nullptr;
  require(clEnqueueMarkerWithWaitList(q1, 0, nullptr, &ev) == CL_SUCCESS && ev, "a marker event in the first context");
  cuCtxSetCurrent(c2->ctx);
  calls.clear();
  require(clReleaseEvent(ev) == CL_SUCCESS, "clReleaseEvent");
  const Call* event = only("cuEventDestroy");
  require(event && event->current == c1->ctx, "the event is destroyed with its own context current");
  require(current() == c2->ctx, "releasing an event restores the caller's current context");

  // SVM memory, given its context, is allocated and freed there.
  void* svm = clSVMAlloc(c1, 0, 256, 0);
  require(svm && allocations.count((CUdeviceptr)(uintptr_t)svm) && allocations[(CUdeviceptr)(uintptr_t)svm] == c1->ctx,
          "clSVMAlloc allocates in the given context");
  calls.clear();
  clSVMFree(c1, svm);
  const Call* svmFree = only("cuMemFree");
  require(svmFree && svmFree->current == c1->ctx, "clSVMFree frees in the given context");
  require(current() == c2->ctx, "clSVMFree restores the caller's current context");

  // A buffer released after its context was destroyed: cuCtxDestroy freed the memory, so nothing is freed now (above
  // all not in whichever context is current), and the release reports the dead context.
  cl_mem orphan = clCreateBuffer(c1, CL_MEM_READ_WRITE, 64, nullptr, &err);
  cl_command_queue orphanQueue = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);
  cuCtxSetCurrent(c2->ctx);
  require(orphan && orphanQueue, "a buffer and a queue of the first context");
  require(clReleaseCommandQueue(q1) == CL_SUCCESS, "clReleaseCommandQueue");
  clReleaseContext(c1);
  require(current() == c2->ctx, "the surviving context is current after the first is released");
  calls.clear();
  require(clReleaseMemObject(orphan) != CL_SUCCESS, "releasing a buffer of a destroyed context reports it");
  require(clReleaseCommandQueue(orphanQueue) != CL_SUCCESS, "releasing a queue of a destroyed context reports it");
  require(calls.empty(), "nothing of a destroyed context is freed or destroyed again, in any context");
  require(current() == c2->ctx, "releases after a context is destroyed leave the caller's context current");

  // Creating a buffer for a destroyed context fails instead of allocating in whichever context is current.
  cl_mem buf3 = clCreateBuffer(c2, CL_MEM_READ_WRITE, 64, nullptr, &err);
  require(buf3 && err == CL_SUCCESS, "clCreateBuffer in the second context");
  require(clReleaseMemObject(buf3) == CL_SUCCESS, "clReleaseMemObject in the second context");
  cl_context c3 = newContext(dev);
  _cl_context staleCopy = *c2;   // a context handle that outlives the context, as a stale cl_context would
  clReleaseContext(c2);
  size_t before = allocations.size();
  cl_mem stale = clCreateBuffer(&staleCopy, CL_MEM_READ_WRITE, 64, nullptr, &err);
  require(!stale && err != CL_SUCCESS, "clCreateBuffer for a destroyed context fails");
  require(allocations.size() == before, "nothing is allocated for a destroyed context");
  require(current() == c3->ctx, "a failed clCreateBuffer leaves the caller's context current");
  clReleaseContext(c3);

  if (failures) { std::printf("cuda release context test: %d FAILED\n", failures); return 1; }
  std::printf("cuda release context test PASS\n");
  return 0;
}
