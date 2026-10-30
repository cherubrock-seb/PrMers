// clCreateBuffer with CL_MEM_COPY_HOST_PTR on the CUDA backend reports a failed upload instead of returning a buffer
// that silently lacks its initial contents (the trig and weight tables are uploaded this way).  Builds the real
// src/cuda/clwrap_cuda.cpp against the recording stand-in driver in tests/cuda_stub: no GPU and no CUDA install needed.
#include "cuda/tinycuda.h"
#include "stub_driver.h"

#include <cstdio>

using namespace cuda_stub;

static int failures = 0;
static void require(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

int main() {
  cl_device_id dev{};
  unsigned n = 0;
  require(clGetDeviceIDs(nullptr, 0, 1, &dev, &n) == CL_SUCCESS && n == 1, "clGetDeviceIDs");
  int err = -1;
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  require(ctx && err == CL_SUCCESS, "clCreateContext");

  unsigned table[16] = {1, 2, 3};
  cl_mem ok = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(table), table, &err);
  require(ok && err == CL_SUCCESS, "clCreateBuffer with a host copy");
  require(clReleaseMemObject(ok) == CL_SUCCESS, "clReleaseMemObject");
  require(allocations.empty(), "the released buffer is freed");

  failNext["cuMemcpyHtoD"] = CUDA_ERROR_ILLEGAL_ADDRESS;
  err = CL_SUCCESS;
  cl_mem bad = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(table), table, &err);
  require(!bad, "a failed host copy returns no buffer");
  require(err != CL_SUCCESS, "a failed host copy reports an error");
  require(allocations.empty(), "the device memory of a failed host copy is freed");
  if (bad) clReleaseMemObject(bad);

  clReleaseContext(ctx);
  if (failures) { std::printf("cuda create buffer copy test: %d FAILED\n", failures); return 1; }
  std::printf("cuda create buffer copy test PASS\n");
  return 0;
}
