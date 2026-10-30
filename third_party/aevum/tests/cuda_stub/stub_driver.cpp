// The recording stand-in for the CUDA driver (see stub_driver.h), and the few symbols clwrap_cuda.cpp takes from the
// rest of the program.
#include "stub_driver.h"
#include "cuda/cudawrap.h"

#include <algorithm>
#include <cstdio>
#include <iterator>

int prpll_verbose = 0;
std::string NvrtcProgram::preprocessOpenCL(const std::string& source) { return source; }
std::string NvrtcProgram::compile(const std::string&, const std::string&, const std::vector<std::string>&,
                                  const std::vector<std::pair<std::string, std::string>>&) { return {}; }
NvrtcProgram::Images NvrtcProgram::compileImages(const std::string&, const std::string&,
                                                       const std::vector<std::string>&,
                                                       const std::vector<std::pair<std::string, std::string>>&) { return {}; }

namespace cuda_stub {
thread_local std::vector<CUcontext> contextStack;
std::map<CUdeviceptr, CUcontext> allocations;
std::vector<Call> calls;
std::vector<Call> uses;
std::map<std::string, CUresult> failNext;
std::string pciBusId = "0000:00:00.0";
CUresult pciBusIdResult = CUDA_SUCCESS;

CUcontext current() { return contextStack.empty() ? nullptr : contextStack.back(); }
bool alive(CUcontext c) { return c && c->alive; }
}  // namespace cuda_stub

using namespace cuda_stub;

namespace {
CUdeviceptr nextPtr = 0x10000;

// The failure armed for `what`, once.
CUresult injected(const char* what) {
  auto it = failNext.find(what);
  if (it == failNext.end()) return CUDA_SUCCESS;
  CUresult r = it->second;
  failNext.erase(it);
  return r;
}

// Like the driver: a function or stream can only be used with its own (live) context current.
CUresult useIn(const char* what, CUcontext owner) {
  uses.push_back({what, current(), owner});
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  return current() == owner ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

// Like the driver: an object can only be freed or destroyed with its own (live) context current.
CUresult destroyIn(const char* what, CUcontext owner) {
  calls.push_back({what, current(), owner});
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  return current() == owner ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
}  // namespace

CUresult cuInit(unsigned) { return CUDA_SUCCESS; }
CUresult cuGetErrorName(CUresult r, const char** s) { *s = r == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_STUB"; return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int* v) { *v = 12000; return CUDA_SUCCESS; }
CUresult cuDeviceGetCount(int* n) { *n = 1; return CUDA_SUCCESS; }
CUresult cuDeviceGet(CUdevice* d, int i) { *d = i; return CUDA_SUCCESS; }
CUresult cuDeviceGetName(char* name, int len, CUdevice) { std::snprintf(name, len, "stub"); return CUDA_SUCCESS; }
CUresult cuDeviceGetPCIBusId(char* pci, int len, CUdevice) {
  if (pciBusIdResult != CUDA_SUCCESS) return pciBusIdResult;
  std::snprintf(pci, len, "%s", pciBusId.c_str());
  return CUDA_SUCCESS;
}
CUresult cuDeviceGetAttribute(int* v, CUdevice_attribute, CUdevice) { *v = 0; return CUDA_SUCCESS; }
CUresult cuDeviceTotalMem(size_t* b, CUdevice) { *b = 0; return CUDA_SUCCESS; }

CUresult cuCtxCreate(CUcontext* out, unsigned, CUdevice) {
  static int ids = 0;
  *out = new CUctx_st{++ids, true};
  contextStack.push_back(*out);   // cuCtxCreate pushes the new context
  return CUDA_SUCCESS;
}
CUresult cuCtxDestroy(CUcontext c) {
  if (!alive(c)) return CUDA_ERROR_INVALID_CONTEXT;
  c->alive = false;        // kept allocated so a stale handle stays distinguishable
  contextStack.erase(std::remove(contextStack.begin(), contextStack.end(), c), contextStack.end());
  for (auto it = allocations.begin(); it != allocations.end();) {
    it = it->second == c ? allocations.erase(it) : std::next(it);
  }
  return CUDA_SUCCESS;
}
CUresult cuCtxSetCurrent(CUcontext c) {
  if (!c) { if (!contextStack.empty()) contextStack.pop_back(); return CUDA_SUCCESS; }
  if (!alive(c)) return CUDA_ERROR_INVALID_CONTEXT;
  if (contextStack.empty()) contextStack.push_back(c); else contextStack.back() = c;
  return CUDA_SUCCESS;
}
CUresult cuCtxGetCurrent(CUcontext* c) { *c = current(); return CUDA_SUCCESS; }
CUresult cuCtxPushCurrent(CUcontext c) {
  if (!alive(c)) return CUDA_ERROR_INVALID_CONTEXT;
  contextStack.push_back(c);
  return CUDA_SUCCESS;
}
CUresult cuCtxPopCurrent(CUcontext* c) {
  if (contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
  if (c) *c = contextStack.back();
  contextStack.pop_back();
  return CUDA_SUCCESS;
}
CUresult cuCtxSetLimit(CUlimit, size_t) { return CUDA_SUCCESS; }

CUresult cuMemAlloc(CUdeviceptr* p, size_t) {
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  if (CUresult r = injected("cuMemAlloc")) return r;
  *p = nextPtr;
  nextPtr += 0x1000;
  allocations[*p] = current();
  return CUDA_SUCCESS;
}
CUresult cuMemFree(CUdeviceptr p) {
  auto it = allocations.find(p);
  CUresult r = destroyIn("cuMemFree", it == allocations.end() ? nullptr : it->second);
  if (r == CUDA_SUCCESS) r = injected("cuMemFree");
  if (r == CUDA_SUCCESS) allocations.erase(it);
  return r;
}
CUresult cuMemGetInfo(size_t* f, size_t* t) { *f = *t = 0; return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD(CUdeviceptr, const void*, size_t) { return injected("cuMemcpyHtoD"); }
CUresult cuMemcpyDtoH(void*, CUdeviceptr, size_t) { return CUDA_SUCCESS; }
CUresult cuMemcpyHtoDAsync(CUdeviceptr, const void*, size_t, CUstream s) { return useIn("cuMemcpyHtoDAsync", s->owner); }
CUresult cuMemcpyDtoHAsync(void*, CUdeviceptr, size_t, CUstream s) { return useIn("cuMemcpyDtoHAsync", s->owner); }
CUresult cuMemcpyDtoDAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream s) { return useIn("cuMemcpyDtoDAsync", s->owner); }
CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream s) { return useIn("cuMemsetD8Async", s->owner); }
CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, CUstream s) { return useIn("cuMemsetD16Async", s->owner); }
CUresult cuMemsetD32Async(CUdeviceptr, unsigned, size_t, CUstream s) { return useIn("cuMemsetD32Async", s->owner); }
CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned, size_t, size_t, CUstream s) {
  return useIn("cuMemsetD2D32Async", s->owner);
}

// A module is loaded in the current context, and its functions belong to it.
static CUresult loadModule(const char* what, CUmodule* m) {
  uses.push_back({what, current(), current()});
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  if (CUresult r = injected(what)) return r;
  *m = new CUmod_st{current()};
  return CUDA_SUCCESS;
}
CUresult cuModuleLoadData(CUmodule* m, const void*) { return loadModule("cuModuleLoadData", m); }
CUresult cuModuleLoadDataEx(CUmodule* m, const void*, unsigned, CUjit_option*, void**) {
  return loadModule("cuModuleLoadDataEx", m);
}
CUresult cuModuleUnload(CUmodule) { return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction* f, CUmodule m, const char*) {
  if (!m || !alive(m->owner)) return CUDA_ERROR_INVALID_HANDLE;
  *f = new CUfunc_st{m};
  return CUDA_SUCCESS;
}
CUresult cuFuncGetAttribute(int* v, CUfunction_attribute, CUfunction) { *v = 0; return CUDA_SUCCESS; }
CUresult cuLaunchKernel(CUfunction f, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream s,
                        void**, void**) {
  // The function, the stream and the current context must all be the same context.
  if (f->module->owner != s->owner) { uses.push_back({"cuLaunchKernel", current(), s->owner}); return CUDA_ERROR_INVALID_HANDLE; }
  return useIn("cuLaunchKernel", s->owner);
}

CUresult cuStreamCreate(CUstream* s, unsigned) {
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  *s = new CUstream_st{current()};
  return CUDA_SUCCESS;
}
CUresult cuStreamDestroy(CUstream s) {
  CUresult r = destroyIn("cuStreamDestroy", s->owner);
  if (r == CUDA_SUCCESS) delete s;
  return r;
}
CUresult cuStreamSynchronize(CUstream s) { return useIn("cuStreamSynchronize", s->owner); }
CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned) { return CUDA_SUCCESS; }
CUresult cuStreamSetAttribute(CUstream, CUstreamAttrID, const CUstreamAttrValue*) { return CUDA_SUCCESS; }

CUresult cuEventCreate(CUevent* e, unsigned) {
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  *e = new CUevent_st{current()};
  return CUDA_SUCCESS;
}
CUresult cuEventDestroy(CUevent e) {
  CUresult r = destroyIn("cuEventDestroy", e->owner);
  if (r == CUDA_SUCCESS) delete e;
  return r;
}
CUresult cuEventRecord(CUevent, CUstream) { return CUDA_SUCCESS; }
CUresult cuEventQuery(CUevent) { return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent) { return CUDA_SUCCESS; }
CUresult cuEventElapsedTime(float* ms, CUevent, CUevent) { *ms = 0; return CUDA_SUCCESS; }

CUresult cuGraphCreate(CUgraph*, unsigned) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuGraphDestroy(CUgraph) { return CUDA_SUCCESS; }
CUresult cuGraphAddKernelNode(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*) {
  return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuGraphInstantiateWithFlags(CUgraphExec*, CUgraph, unsigned long long) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) {
  return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuGraphExecDestroy(CUgraphExec) { return CUDA_SUCCESS; }
CUresult cuGraphLaunch(CUgraphExec, CUstream) { return CUDA_ERROR_INVALID_VALUE; }

nvrtcResult nvrtcVersion(int* major, int* minor) { *major = 12; *minor = 0; return NVRTC_SUCCESS; }
nvrtcResult nvrtcGetNumSupportedArchs(int* n) { *n = 0; return NVRTC_SUCCESS; }
nvrtcResult nvrtcGetSupportedArchs(int*) { return NVRTC_SUCCESS; }
