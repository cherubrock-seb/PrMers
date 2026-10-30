// A stand-in for the CUDA driver API header, for host tests that compile src/cuda/clwrap_cuda.cpp without CUDA
// installed.  It declares just what the shim uses; a test defines the functions it needs (see
// tests/cuda_release_context_test.cpp).  Not the real ABI: never link it against libcuda.
#pragma once

#include <cstddef>

#define CUDA_VERSION 12000

typedef enum CUresult_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,
  CUDA_ERROR_NOT_READY = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
  CUDA_ERROR_INVALID_PC = 718,
  CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_UNKNOWN = 999,
} CUresult;

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUgraphNode_st* CUgraphNode;

typedef enum CUjit_option_enum {
  CU_JIT_INFO_LOG_BUFFER = 3,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
} CUjit_option;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
} CUdevice_attribute;

typedef enum CUfunction_attribute_enum {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
} CUfunction_attribute;

typedef enum CUlimit_enum { CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6 } CUlimit;

enum { CU_EVENT_DEFAULT = 0, CU_STREAM_NON_BLOCKING = 1 };

typedef enum CUaccessProperty_enum {
  CU_ACCESS_PROPERTY_NORMAL = 0,
  CU_ACCESS_PROPERTY_STREAMING = 1,
  CU_ACCESS_PROPERTY_PERSISTING = 2,
} CUaccessProperty;

typedef struct CUaccessPolicyWindow_st {
  void* base_ptr;
  size_t num_bytes;
  float hitRatio;
  CUaccessProperty hitProp;
  CUaccessProperty missProp;
} CUaccessPolicyWindow;

typedef union CUstreamAttrValue_union {
  CUaccessPolicyWindow accessPolicyWindow;
} CUstreamAttrValue;

typedef enum CUstreamAttrID_enum { CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1 } CUstreamAttrID;

typedef struct CUDA_KERNEL_NODE_PARAMS_st {
  CUfunction func;
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  void** kernelParams;
  void** extra;
} CUDA_KERNEL_NODE_PARAMS;

CUresult cuInit(unsigned int flags);
CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceGetPCIBusId(char* pci, int len, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* pctx);
CUresult cuCtxSetLimit(CUlimit limit, size_t value);

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemGetInfo(size_t* free, size_t* total);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytes);
CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t bytes);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t bytes, CUstream stream);
CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src, size_t bytes, CUstream stream);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream);
CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t n, CUstream stream);
CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short us, size_t n, CUstream stream);
CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t n, CUstream stream);
CUresult cuMemsetD2D32Async(CUdeviceptr dst, size_t pitch, unsigned int ui, size_t width, size_t height,
                            CUstream stream);

CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options,
                            void** optionValues);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags);
CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value);

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventQuery(CUevent hEvent);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags);
CUresult cuGraphDestroy(CUgraph hGraph);
CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies,
                              size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags);
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                        const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
