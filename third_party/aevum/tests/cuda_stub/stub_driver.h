// A recording stand-in for the CUDA driver, for host tests that build the real src/cuda/clwrap_cuda.cpp without a GPU
// or a CUDA install (with tests/cuda_stub/cuda.h and nvrtc.h).  It keeps a per-thread context stack like the driver,
// records every free and destroy with the context current at the time, and fails one made in the wrong context the
// way the driver can (CUDA_ERROR_INVALID_VALUE, or CUDA_ERROR_INVALID_CONTEXT with no live context current).  Modules
// are loaded in the current context, and a launch or stream operation made with a context current other than its
// function's and stream's fails with CUDA_ERROR_INVALID_HANDLE; each is recorded in `uses`.
#pragma once

#include <cuda.h>

#include <map>
#include <string>
#include <vector>

struct CUctx_st { int id; bool alive; };   // a destroyed context stays allocated, so a stale handle is recognisable
struct CUstream_st { CUcontext owner; };
struct CUevent_st { CUcontext owner; };
struct CUmod_st { CUcontext owner; };
struct CUfunc_st { CUmodule module; };

namespace cuda_stub {

struct Call { std::string what; CUcontext current; CUcontext owner; };

extern thread_local std::vector<CUcontext> contextStack;     // this thread's context stack; back() is current
extern std::map<CUdeviceptr, CUcontext> allocations;  // live device memory and the context it was allocated in
extern std::vector<Call> calls;                       // every free / destroy, with the context current at the time
extern std::vector<Call> uses;                        // every module load, launch and stream operation, with the
                                                      // context current at the time and the object's own context
extern std::map<std::string, CUresult> failNext;
extern std::string pciBusId;                         // value returned by cuDeviceGetPCIBusId
extern CUresult pciBusIdResult;                    // injected result for cuDeviceGetPCIBusId      // make the next call of "cuMemFree", "cuMemAlloc" or
                                                      // "cuMemcpyHtoD" fail with this

CUcontext current();
bool alive(CUcontext c);

}  // namespace cuda_stub
