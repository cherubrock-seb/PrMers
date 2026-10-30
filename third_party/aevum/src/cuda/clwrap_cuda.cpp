// CUDA Driver API implementation of OpenCL API functions.
// This replaces clwrap.cpp when building with the native CUDA backend,
// mapping all cl* calls to cu* equivalents via the CUDA Driver API.

#include "tinycuda.h"
#include "GraphBatch.h"
#include "CudaModuleLife.h"
#include "PtxRegisterCache.h"
#include "cudawrap.h"  // For NvrtcProgram::preprocessOpenCL and compile
#include "CudaFill.h"
#include "CudaErrorMap.h"
#include "NvrtcArch.h"
#include "CudaBinaryBlob.h"

#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <unordered_set>
#include <mutex>
#ifdef __linux__
#include <unistd.h>
#endif
#include "CudaEventStatus.h"

using namespace std;

// Track allocated cl_mem objects so clSetKernelArg can distinguish buffer args from scalars.
// In OpenCL, buffer args are passed as &memobj where memobj is cl_mem (a pointer to _cl_mem).
// We need to convert these to CUdeviceptr for CUDA kernel launch.
// The set is shared by every thread that drives a Gpu (-workers N runs N of them at once), so all access goes
// through g_allocatedBuffersMutex.
static unordered_set<cl_mem> g_allocatedBuffers;
static std::mutex g_allocatedBuffersMutex;

// CUDA autotune temporarily creates a second context. Keep the live
// contexts in creation order so destroying a temporary tuning context restores
// the production context used by APIs that do not carry an owning context.
static CUcontext g_cudaContext = nullptr;
static vector<CUcontext> g_cudaContexts;

static void rememberCudaContext(CUcontext ctx) {
  g_cudaContexts.push_back(ctx);
  g_cudaContext = ctx;
}

static void forgetCudaContext(CUcontext ctx) {
  auto it = find(g_cudaContexts.begin(), g_cudaContexts.end(), ctx);
  if (it != g_cudaContexts.end()) g_cudaContexts.erase(it);
  g_cudaContext = g_cudaContexts.empty() ? nullptr : g_cudaContexts.back();
}

// The OpenCL-style status for a CUDA driver result.  Only a result that means a lost, reset or faulted device or
// context maps to the status that reads as a lost GPU; an out-of-memory maps to the allocation status and anything
// else to a generic one (see CudaErrorMap.h).  The CUDA error itself is kept for the message.
static thread_local char g_lastCuError[96];

static int clStatusFromCu(CUresult r) {
  if (r == CUDA_SUCCESS) { g_lastCuError[0] = 0; return CL_SUCCESS; }
  const char* name = nullptr;
  if (cuGetErrorName(r, &name) != CUDA_SUCCESS) name = nullptr;
  snprintf(g_lastCuError, sizeof(g_lastCuError), "%s (%d)", name ? name : "CUDA error", (int) r);
  return cuda_error_map::toClStatus((int) r);
}

const char* aevumCudaLastError() { return g_lastCuError[0] ? g_lastCuError : nullptr; }

static void ensureContextCurrent() {
  if (g_cudaContext) {
    cuCtxSetCurrent(g_cudaContext);
  }
}

// Makes an object's own context current on this thread: the context its program, kernel, queue or event was created
// in.  The shim's newest context (ensureContextCurrent) is the wrong one whenever several are alive (an engine next to
// an autotune context, workers or engines on other devices), and it is per process while the current context is per
// thread.  Like ensureContextCurrent it leaves the context current; a null owner falls back to the newest context.
static CUresult makeCurrent(CUcontext owner) {
  if (!owner) {
    ensureContextCurrent();
    return CUDA_SUCCESS;
  }
  CUcontext current{};
  if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == owner) return CUDA_SUCCESS;
  return cuCtxSetCurrent(owner);
}

// The context a program made for `ctx` belongs to: the given one, else the shim's newest.
static CUcontext programContext(cl_context ctx) {
  return ctx ? ctx->ctx : g_cudaContext;
}

// Every loaded CUmodule is unloaded when the last program or kernel using it is released, in its own context and only
// while that context is alive (see CudaModuleLife.h).  Never destroyed, so that a program or kernel released during
// static destruction at exit still finds it.
static cuda_module::Registry& g_modules = *new cuda_module::Registry;

static void unloadModule(void* module, void* context) {
  if (cuCtxPushCurrent(static_cast<CUcontext>(context)) != CUDA_SUCCESS) return;
  cuModuleUnload(static_cast<CUmodule>(module));
  CUcontext popped{};
  cuCtxPopCurrent(&popped);
}

// A reference to a module just loaded in the current context.
static std::shared_ptr<void> adoptModule(CUmodule module) {
  CUcontext current{};
  cuCtxGetCurrent(&current);
  return g_modules.adopt(module, current, unloadModule);
}

// Global state for CUDA initialization
static bool g_cudaInitialized = false;
static void ensureCudaInit() {
  if (!g_cudaInitialized) {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cuInit failed: %d\n", (int)err);
    }
    g_cudaInitialized = true;
  }
}

// Global device list (allocated once, never freed)
static vector<_cl_device_id> g_devices;
static bool g_devicesEnumerated = false;

static void enumerateDevices() {
  if (g_devicesEnumerated) return;
  ensureCudaInit();
  int count = 0;
  cuDeviceGetCount(&count);
  g_devices.resize(count);
  for (int i = 0; i < count; i++) {
    cuDeviceGet(&g_devices[i].dev, i);
  }
  g_devicesEnumerated = true;
}

// ---- OpenCL API implementations ----

extern "C" {

unsigned clGetPlatformIDs(unsigned num, cl_platform_id* platforms, unsigned* numRet) {
  // CUDA has no "platforms" concept — just return 1 dummy
  if (numRet) *numRet = 1;
  if (platforms && num >= 1) platforms[0] = nullptr;
  return CL_SUCCESS;
}

int clGetDeviceIDs(cl_platform_id, cl_device_type, unsigned num, cl_device_id* devices, unsigned* numRet) {
  enumerateDevices();
  unsigned n = g_devices.size();
  if (numRet) *numRet = n;
  if (devices) {
    for (unsigned i = 0; i < min(num, n); i++) {
      devices[i] = &g_devices[i];
    }
  }
  return n > 0 ? CL_SUCCESS : CL_DEVICE_NOT_FOUND;
}

cl_context clCreateContext(const intptr_t*, unsigned nDevices, const cl_device_id* devices,
                           void (*)(const char*, const void*, size_t, void*), void*, int* err) {
  if (!devices || nDevices == 0) { if (err) *err = CL_INVALID_DEVICE; return nullptr; }
  auto* ctx = new _cl_context;
  ctx->dev = devices[0]->dev;
#if CUDA_VERSION >= 13000
  CUctxCreateParams params{};
  CUresult r = cuCtxCreate_v4(&ctx->ctx, &params, 0, ctx->dev);
#else
  CUresult r = cuCtxCreate(&ctx->ctx, 0, ctx->dev);
#endif
  if (r != CUDA_SUCCESS) {
    delete ctx;
    if (err) *err = clStatusFromCu(r);
    return nullptr;
  }
  rememberCudaContext(ctx->ctx);
  g_modules.contextCreated(ctx->ctx);

  // L2 persistence: no benefit measured for this workload.
  // cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 16 * 1024 * 1024);

  if (err) *err = CL_SUCCESS;
  return ctx;
}

int clReleaseContext(cl_context ctx) {
  if (ctx) {
    const CUcontext dead = ctx->ctx;
    forgetCudaContext(dead);
    g_modules.contextDestroying(dead);   // cuCtxDestroy unloads its modules; programs and kernels released later must not
    cuCtxDestroy(dead);
    if (g_cudaContext) cuCtxSetCurrent(g_cudaContext);
    delete ctx;
  }
  return CL_SUCCESS;
}

int clReleaseProgram(cl_program p) {
  if (p) {
    // Drops the program's reference to its module.  As in OpenCL, where clCreateKernel retains the program, the
    // module stays loaded while any kernel created from it is alive (loadAux() releases the program and keeps the
    // kernel); the last reference unloads it (CudaModuleLife.h).
    delete p;
  }
  return CL_SUCCESS;
}

int clReleaseCommandQueue(cl_command_queue q) {
  int status = CL_SUCCESS;
  if (q) {
    {
      // The stream and its graphs belong to the queue's context, not necessarily the one the shim made current last.
      CudaContextScope owner(q->owner);
      CUresult r = owner.status();
      if (r == CUDA_SUCCESS) {
        if (q->squareGraph) {
          r = cuStreamSynchronize(q->stream);
          delete q->squareGraph;
        }
        CUresult const destroyed = cuStreamDestroy(q->stream);
        if (r == CUDA_SUCCESS) r = destroyed;
      }
      // Else the context is gone and cuCtxDestroy destroyed the stream and graphs with it; the graph cache's host
      // side is left alone rather than destroying handles that may since name another context's objects.
      status = clStatusFromCu(r);
    }
    delete q;
  }
  return status;
}

// ---- Program compilation (NVRTC) ----

cl_program clCreateProgramWithSource(cl_context ctx, unsigned count, const char** strings,
                                      const size_t* lengths, int* err) {
  auto* prog = new _cl_program;
  prog->owner = programContext(ctx);
  for (unsigned i = 0; i < count; i++) {
    if (lengths && lengths[i]) {
      prog->source.append(strings[i], lengths[i]);
    } else {
      prog->source.append(strings[i]);
    }
  }
  if (err) *err = CL_SUCCESS;
  return prog;
}

// Loads prog->cubin when there is one (SASS for this device: no JIT, and no PTX-version check to fail on a driver
// older than the toolkit), and otherwise, or when the driver rejects it (logged), the PTX through the driver's JIT.
// A rejected CUBIN is dropped, so the kernel cache stores what loaded.
static CUresult loadModule(_cl_program* prog, unsigned nOpts, CUjit_option* opts, void** optVals) {
  if (!prog->cubin.empty()) {
    CUresult const r = cuModuleLoadDataEx(&prog->module, prog->cubin.data(), nOpts, opts, optVals);
    if (r == CUDA_SUCCESS) return r;
    const char* errName = nullptr;
    cuGetErrorName(r, &errName);
    fprintf(stderr, "CUBIN rejected by the driver: %s (%d); loading the PTX through the JIT instead\n",
            errName ? errName : "?", (int)r);
    prog->cubin.clear();
  }
  return cuModuleLoadDataEx(&prog->module, prog->ptx.c_str(), nOpts, opts, optVals);
}

cl_program clCreateProgramWithBinary(cl_context ctx, unsigned nDevices, const cl_device_id*,
                                      const size_t* lengths, const unsigned char** binaries,
                                      int* binaryStatus, int* err) {
  // The "binary" is the PTX, followed by the CUBIN when there is one (see CudaBinaryBlob.h)
  auto* prog = new _cl_program;
  prog->owner = programContext(ctx);
  if (lengths && binaries && lengths[0] > 0) {
    const std::string blob((const char*)binaries[0], lengths[0]);
    if (cuda_blob::split(blob, prog->ptx, prog->cubin) == cuda_blob::Split::BareElf) {
      fprintf(stderr, "Cached kernel binary is a bare CUBIN (no PTX): recompiling\n");
      if (binaryStatus) binaryStatus[0] = CL_INVALID_BINARY;
      if (err) *err = CL_INVALID_BINARY;
      return prog;
    }
    prog->compiled = true;
 // Load in the program owning context: prefer the CUBIN, then fall back to the PTX JIT.
 CUresult r = makeCurrent(prog->owner);
 if (r == CUDA_SUCCESS) r = loadModule(prog, 0, nullptr, nullptr);
    if (r == CUDA_SUCCESS) {
      prog->moduleLoaded = true;
      prog->moduleRef = adoptModule(prog->module);
      if (binaryStatus) binaryStatus[0] = CL_SUCCESS;
    } else {
      fprintf(stderr, "cuModuleLoadData from cache failed: %d, blob size=%zu\n", (int)r, lengths[0]);
      prog->compiled = false;
      if (binaryStatus) binaryStatus[0] = CL_INVALID_BINARY;
      if (err) { *err = CL_INVALID_BINARY; return prog; }
    }
  }
  if (err) *err = CL_SUCCESS;
  return prog;
}

// Build log storage (per-program)
static string g_lastBuildLog;

// The architectures this NVRTC accepts (nvrtcGetSupportedArchs, NVRTC >= 11.2), as major * 10 + minor; empty when
// unknown.  See NvrtcArch.h for how the --gpu-architecture is chosen from them.
static const vector<int>& nvrtcSupportedArchs() {
  static const vector<int> archs = [] {
    vector<int> v;
#if CUDA_VERSION >= 11020
    int n = 0;
    if (nvrtcGetNumSupportedArchs(&n) == NVRTC_SUCCESS && n > 0) {
      v.resize(n);
      if (nvrtcGetSupportedArchs(v.data()) != NVRTC_SUCCESS) { v.clear(); }
    }
#endif
    return v;
  }();
  return archs;
}

int clCompileProgram(cl_program prog, unsigned nDevices, const cl_device_id* devices, const char* options,
                     unsigned numHeaders, const cl_program* headers, const char* const* headerNames,
                     void (*)(cl_program, void*), void*) {
  if (!prog) return CL_INVALID_PROGRAM;

  // Get device arch for NVRTC
  CUdevice dev = devices ? devices[0]->dev : g_devices[0].dev;
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

  // The GPU's own sm_XY when this NVRTC has it; otherwise PTX for the newest compute_XY below the GPU, which the driver
  // JIT-compiles (CUDA 13 dropped sm_50..sm_72, and an older toolkit does not know the newest GPUs).
  const int cc = major * 10 + minor;
  const vector<int>& archs = nvrtcSupportedArchs();
  const string arch = nvrtc_arch::choose(cc, archs);
  if (arch.empty() || arch.rfind("sm_", 0) != 0) {
    int nvMajor = 0, nvMinor = 0;
    nvrtcVersion(&nvMajor, &nvMinor);
    if (arch.empty()) {
      char mes[384];
      snprintf(mes, sizeof(mes), "This GPU is sm_%d, but this build's NVRTC %d.%d supports only sm_%d and newer. "
               "Use an Aevum build made with an older CUDA toolkit (CUDA 13 dropped Maxwell, Pascal and Volta).",
               cc, nvMajor, nvMinor, *min_element(archs.begin(), archs.end()));
      fprintf(stderr, "%s\n", mes);
      g_lastBuildLog = mes;
      prog->compiled = false;
      return CL_COMPILE_PROGRAM_FAILURE;
    }
    static std::once_flag logged;
    std::call_once(logged, [&] {
      fprintf(stderr, "NVRTC %d.%d does not support sm_%d; compiling for %s, which the driver JIT-compiles\n",
              nvMajor, nvMinor, cc, arch.c_str());
    });
  }
  const string archOpt = "--gpu-architecture=" + arch;

  // Parse OpenCL options string into NVRTC options
  // Convert OpenCL build options to NVRTC equivalents:
  //   -cl-std=CL2.0         → -std=c++17
  //   -cl-finite-math-only  → --fmad=true (enable FMA contraction, the safe subset)
  //   -Dfoo=bar             → -Dfoo=bar (pass through)
  vector<string> nvrtcOpts;
  nvrtcOpts.push_back(archOpt);
  nvrtcOpts.push_back("-default-device");
  nvrtcOpts.push_back("-std=c++17");
  nvrtcOpts.push_back("-w");  // Suppress NVRTC macro redefinition warnings

  // FMA contraction: OpenCL uses -cl-finite-math-only + #pragma OPENCL FP_CONTRACT ON
  // to allow the compiler to contract a*b+c into FMA instructions. NVRTC's --fmad=true
  // is the safe equivalent — it ONLY enables FMA contraction without the dangerous
  // parts of -use_fast_math (no flush-to-zero, no reduced-precision division/sqrt).
  // This is critical for FFT performance: every butterfly is multiply-add pairs.
  nvrtcOpts.push_back("--fmad=true");

  // NOTE: --restrict (all kernel pointers are __restrict__) was tested but causes GPU read
  // errors — some PRPLL kernels use in-place operations where in/out buffers alias.
  // Do NOT enable globally. The compiler still auto-uses __ldg() for const pointers on sm_35+.

  // Debug: dump full options string
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    static bool dumpedOpts = false;
    if (dumpPrefix && !dumpedOpts && options) {
      dumpedOpts = true;
      fprintf(stderr, "clCompileProgram options: [%s]\n", options);
      FILE* optLog = fopen("kernel_regs.log", "a");
      if (optLog) { fprintf(optLog, "clCompileProgram options: [%s]\n", options); fclose(optLog); }
    }
  }

  int maxregcount = 0;
  if (options) {
    istringstream iss(options);
    string tok;
    while (iss >> tok) {
      if (tok.substr(0, 2) == "-D") {
        // Fix AMD-only FFT variants for NVIDIA: variant_W=0 and variant_H=0 require
        // AMD builtins (__builtin_amdgcn_ds_bpermute etc). Replace with variant 2.
        // FFT_VARIANT is a 3-digit number WMH: e.g. 000, 101, 202
        if (tok.find("FFT_VARIANT=") != string::npos) {
          size_t eqPos = tok.find('=');
          string valStr = tok.substr(eqPos + 1);
          // Strip trailing 'u' suffix
          if (!valStr.empty() && valStr.back() == 'u') valStr.pop_back();
          int val = atoi(valStr.c_str());
          int vW = val / 100;
          int vM = (val % 100) / 10;
          int vH = val % 10;
          if (vW == 0) vW = 2;  // AMD BCAST → NVIDIA generic
          if (vH == 0) vH = 2;
          int newVal = vW * 100 + vM * 10 + vH;
          tok = "-DFFT_VARIANT=" + to_string(newVal) + "u";
        }
        nvrtcOpts.push_back(tok);
      } else if (tok == "-cl-finite-math-only" || tok == "-cl-fast-relaxed-math") {
        // FMA contraction already enabled above via --fmad=true.
        // Do NOT use -use_fast_math here — it enables flush-to-zero and
        // reduced-precision division/sqrt which breaks tailMul accuracy.
      } else if (tok.substr(0, 14) == "--maxrregcount") {
        nvrtcOpts.push_back(tok);
        maxregcount = atoi(tok.substr(15, 3).c_str());
      }
      // Skip other -cl-* options (not applicable to NVRTC)
    }
  }

  // Build NVRTC headers from the cl_program header array
  vector<pair<string, string>> nvrtcHeaders;

  // Add all OpenCL source headers
  for (unsigned i = 0; i < numHeaders; i++)  {
    // First header: opencl_compat.cuh (inject as virtual NVRTC header)
    if (i == 0) {
      nvrtcHeaders.push_back({"opencl_compat.cuh", headers[0]->source});
    }
    // Remaining headers need preprocessing
    else if (headers[i] && headerNames[i]) {
      // Preprocess OpenCL source for CUDA compatibility
      string processedSrc = NvrtcProgram::preprocessOpenCL(headers[i]->source);
      // Debug: verify KERNEL macro replacement
// I'm not sure what Sherpa was trying to print out here.  It prints out nothing useful.
//      {
//        static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
//        if (dumpPrefix && string(headerNames[i]) == "base.cl") {
//          auto pos = processedSrc.find("KERNEL");
//          if (pos != string::npos) {
//            string ctx = processedSrc.substr(pos > 20 ? pos-20 : 0, 120);
//            fprintf(stderr, "base.cl KERNEL context: [%s]\n", ctx.c_str());
//          }
//        }
//      }
      nvrtcHeaders.push_back({headerNames[i], processedSrc});
    }
  }

  // Preprocess the main source
  string processedSource = NvrtcProgram::preprocessOpenCL(prog->source);

  // Prepend opencl_compat.cuh include if not already there
  if (processedSource.find("opencl_compat.cuh") == string::npos) {
    processedSource = "#include \"opencl_compat.cuh\"\n" + processedSource;
  }

  // Debug: dump preprocessed source when PRPLL_DUMP_PTX is set
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    if (dumpPrefix) {
      static int srcCount = 0;
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_src_%d.cu", dumpPrefix, srcCount++);
      FILE* f = fopen(fname, "w");
      if (f) {
        fwrite(processedSource.c_str(), 1, processedSource.size(), f);
        fclose(f);
        fprintf(stderr, "Source dumped to %s (%zu bytes)\n", fname, processedSource.size());
      }
    }
  }

  // Store preprocessed source for __launch_bounds__ parsing in clCreateKernel
  prog->preprocessedSource = processedSource;
  for (auto& [name, src] : nvrtcHeaders) {
    prog->preprocessedSource += "\n";
    prog->preprocessedSource += src;
  }

  // Debug: dump NVRTC options when dumping PTX
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    static bool dumpedOnce = false;
    if (dumpPrefix && !dumpedOnce) {
      dumpedOnce = true;
      fprintf(stderr, "NVRTC options (%zu):\n", nvrtcOpts.size());
      for (auto& o : nvrtcOpts) fprintf(stderr, "  %s\n", o.c_str());
    }
  }

  try {
    // The PTX, and the CUBIN when NVRTC ran its ptxas for a real sm_XY (NvrtcArch.h falls back to compute_XY when
    // this NVRTC does not know the GPU; then there is no CUBIN and the driver JITs the PTX as before).
    auto images = NvrtcProgram::compileImages(processedSource, "prpll_kernel.cu", nvrtcOpts, nvrtcHeaders);
    prog->ptx = std::move(images.ptx);
    // PRPLL_PTX_ONLY=1 keeps the driver's JIT path, for comparing the two.
    static const bool ptxOnly = getenv("PRPLL_PTX_ONLY") != nullptr;
    prog->cubin = ptxOnly ? string{} : std::move(images.cubin);
    // -use PTX_VERSION=x.y: an older driver rejects this NVRTC's CUBIN too, so the relabeled PTX goes through the JIT.
    if (const string ptxVersion = cuda_blob::ptxVersionOverride(options ? options : ""); !ptxVersion.empty()) {
      const string old = cuda_blob::setPtxVersion(prog->ptx, ptxVersion);
      static std::once_flag logged;
      std::call_once(logged, [&] {
        fprintf(stderr, "PTX_VERSION: relabeling the PTX from version %s to %s\n", old.c_str(), ptxVersion.c_str());
      });
      prog->cubin.clear();
    }
    prog->compiled = true;
    g_lastBuildLog.clear();
  } catch (const exception& e) {
    g_lastBuildLog = e.what();
    prog->compiled = false;
    fprintf(stderr, "NVRTC COMPILE FAILED: %s\n", e.what());
    // Dump the full preprocessed source for debugging
    {
      char fname[64];
      static int failCount = 0;
      snprintf(fname, sizeof(fname), "prpll_fail_%d.cu", failCount++);
      FILE* f = fopen(fname, "w");
      if (f) {
        fprintf(f, "// === FAILED Main source ===\n%s\n", processedSource.c_str());
        for (auto& [name, src] : nvrtcHeaders) {
          fprintf(f, "\n// === Header: %s (%zu bytes) ===\n%s\n", name.c_str(), src.size(), src.c_str());
        }
        fclose(f);
        fprintf(stderr, "Dumped failed source to %s\n", fname);
      }
    }
    return CL_COMPILE_PROGRAM_FAILURE;
  }

  // NVRTC accepts --maxrregcount but its CUBIN's register use ignores it (seen with CUDA 13.0: 24 registers asked
  // for carryFused, 64 used), while the driver's JIT honors a .maxnreg directive spliced into the PTX.  So a kernel
  // that asks for a register cap drops the CUBIN and takes the PTX path; the rest keep the CUBIN.

  if (maxregcount) {
    string maxntidPattern = ".maxntid ";
    string maxnregPattern = ".maxnreg " + to_string(maxregcount) + "\n";
    for (size_t startpos = 0; ; ) {
      size_t pos = prog->ptx.find(maxntidPattern, startpos);
      if (pos == string::npos) break;
      prog->ptx.insert(pos, maxnregPattern);
      startpos = pos + 20;
    }
    prog->cubin.clear();
  }

  return CL_SUCCESS;
}

cl_program clLinkProgram(cl_context ctx, unsigned nDevices, const cl_device_id*,
                          const char* options, unsigned nProgs, const cl_program* progs,
                          void (*)(cl_program, void*), void*, int* err) {
  // In CUDA, compilation produces PTX directly — no separate link step needed.
  // Just load the PTX as a CUmodule, in the context the program is linked for.
  if (!progs || nProgs == 0 || !progs[0] || !progs[0]->compiled) {
    if (err) *err = CL_LINK_PROGRAM_FAILURE;
    return nullptr;
  }
  const CUcontext owner = ctx ? ctx->ctx : progs[0]->owner ? progs[0]->owner : g_cudaContext;
  if (CUresult r = makeCurrent(owner); r != CUDA_SUCCESS) {
    if (err) *err = clStatusFromCu(r);
    return nullptr;
  }

  auto* linked = new _cl_program;
  linked->owner = owner;
  linked->ptx = progs[0]->ptx;
  linked->cubin = progs[0]->cubin;
  linked->compiled = true;
  // Carry preprocessed source through for KERNEL(N) parsing in clCreateKernel
  for (unsigned i = 0; i < nProgs; ++i) {
    if (progs[i] && !progs[i]->preprocessedSource.empty()) {
      linked->preprocessedSource += progs[i]->preprocessedSource;
      linked->preprocessedSource += "\n";
    }
  }

  // Use cuModuleLoadDataEx with error log to see JIT errors
  char jitErrorLog[8192] = {};
  char jitInfoLog[4096] = {};
  CUjit_option jitOpts[] = {
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER
  };
  void* jitOptVals[] = {
    (void*)(size_t)sizeof(jitErrorLog), (void*)jitErrorLog,
    (void*)(size_t)sizeof(jitInfoLog), (void*)jitInfoLog
  };
  CUresult r = loadModule(linked, 4, jitOpts, jitOptVals);
  if (r != CUDA_SUCCESS) {
    const char* errName = nullptr;
    cuGetErrorName(r, &errName);
    fprintf(stderr, "cuModuleLoadData FAILED: %s (%d)\n", errName ? errName : "?", (int)r);
    if (jitErrorLog[0]) fprintf(stderr, "JIT error log: %s\n", jitErrorLog);
    if (jitInfoLog[0]) fprintf(stderr, "JIT info log: %s\n", jitInfoLog);
    // Dump first 2000 chars of PTX for debugging
    fprintf(stderr, "PTX size: %zu bytes\n", linked->ptx.size());
    // Dump full PTX to file
    {
      FILE* ptxFile = fopen("failed_ptx.ptx", "w");
      if (ptxFile) {
        fwrite(linked->ptx.c_str(), 1, linked->ptx.size(), ptxFile);
        fclose(ptxFile);
        fprintf(stderr, "Dumped failed PTX to failed_ptx.ptx\n");
      }
    }
    delete linked;
    if (err) *err = CL_LINK_PROGRAM_FAILURE;
    return nullptr;
  }
  linked->moduleLoaded = true;
  linked->moduleRef = adoptModule(linked->module);

  // Dump PTX to file when PRPLL_DUMP_PTX is set (e.g., PRPLL_DUMP_PTX=kernel)
  // Creates files like kernel_0.ptx, kernel_1.ptx, etc.
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    if (dumpPrefix) {
      static int ptxCount = 0;
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_%d.ptx", dumpPrefix, ptxCount++);
      FILE* f = fopen(fname, "w");
      if (f) {
        fwrite(linked->ptx.c_str(), 1, linked->ptx.size(), f);
        fclose(f);
        fprintf(stderr, "PTX dumped to %s (%zu bytes)\n", fname, linked->ptx.size());
      }
    }
  }

  if (err) *err = CL_SUCCESS;
  return linked;
}

int clBuildProgram(cl_program prog, unsigned nDevices, const cl_device_id* devices,
                   const char* options, void (*)(cl_program, void*), void*) {
  // If the program was loaded from binary (cached PTX) and already has a module,
  // skip recompilation — the module is already JIT'd and ready.
  if (prog && prog->moduleLoaded) {
    return CL_SUCCESS;
  }

  // clBuildProgram = compile + link in one step
  int err = clCompileProgram(prog, nDevices, devices, options, 0, nullptr, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) return err;

 CUresult r = makeCurrent(prog->owner);
 if (r != CUDA_SUCCESS) return clStatusFromCu(r);
 r = loadModule(prog, 0, nullptr, nullptr);
  if (r != CUDA_SUCCESS) return CL_BUILD_PROGRAM_FAILURE;
  prog->moduleLoaded = true;
  prog->moduleRef = adoptModule(prog->module);
  return CL_SUCCESS;
}

int clGetProgramBuildInfo(cl_program prog, cl_device_id, cl_program_build_info info,
                           size_t size, void* value, size_t* sizeRet) {
  if (info == CL_PROGRAM_BUILD_LOG) {
    size_t len = g_lastBuildLog.size() + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) {
      memcpy(value, g_lastBuildLog.c_str(), len);
    }
  }
  return CL_SUCCESS;
}

int clGetProgramInfo(cl_program prog, cl_program_info info, size_t size, void* value, size_t* sizeRet) {
  if (!prog) return CL_INVALID_PROGRAM;
  // The cache blob: the PTX, then the CUBIN behind its marker when one loaded (see CudaBinaryBlob.h).
  if (info == CL_PROGRAM_BINARY_SIZES) {
    size_t blobSize = cuda_blob::join(prog->ptx, prog->cubin).size();
    if (sizeRet) *sizeRet = sizeof(size_t);
    if (value && size >= sizeof(size_t)) memcpy(value, &blobSize, sizeof(size_t));
  } else if (info == CL_PROGRAM_BINARIES) {
    if (sizeRet) *sizeRet = sizeof(unsigned char*);
    if (value && size >= sizeof(unsigned char*)) {
      unsigned char** ptrs = (unsigned char**)value;
      if (ptrs[0]) {
        const string blob = cuda_blob::join(prog->ptx, prog->cubin);
        memcpy(ptrs[0], blob.data(), blob.size());
      }
    }
  }
  return CL_SUCCESS;
}

// ---- Kernel ----

cl_kernel clCreateKernel(cl_program prog, const char* name, int* err) {
  if (!prog || !prog->moduleLoaded) {
    if (err) *err = CL_INVALID_PROGRAM;
    return nullptr;
  }
  if (CUresult r = makeCurrent(prog->owner); r != CUDA_SUCCESS) {
    if (err) *err = clStatusFromCu(r);
    return nullptr;
  }
  auto* k = new _cl_kernel;
  k->name = name;
  k->parentModule = prog->module;
  k->owner = prog->owner;
  CUresult r = cuModuleGetFunction(&k->func, prog->module, name);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "cuModuleGetFunction('%s') failed: %d, moduleLoaded=%d, module=%p\n",
            name, (int)r, prog->moduleLoaded, (void*)prog->module);
    delete k;
    if (err) *err = CL_INVALID_KERNEL_NAME;
    return nullptr;
  }
  prog->kernelExported = true;
  k->moduleRef = prog->moduleRef;   // the module outlives the program while this kernel is alive

  // Shared memory carveout: default adaptive carveout is optimal for mixed kernel workloads.

  // Log register and shared memory usage per kernel when PRPLL_DUMP_PTX is set
  {
    static const char* dumpPrefix = getenv("PRPLL_DUMP_PTX");
    if (prpll_verbose || dumpPrefix) {
      int numRegs = 0, shmem = 0, localmem = 0, maxThreads = 0;
      cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, k->func);
      cuFuncGetAttribute(&shmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k->func);
      cuFuncGetAttribute(&localmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES , k->func);
      cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, k->func);
      fprintf(stderr, "  %-25s: %3d regs, %5d shmem, %d localmem, maxThreads=%d\n", name, numRegs, shmem, localmem, maxThreads);
      // Also write to file since WSL2+CUDA swallows stderr
      if (dumpPrefix) {
        FILE* regLog = fopen("kernel_regs.log", "a");
        if (regLog) { fprintf(regLog, "  %-25s: %3d regs, %5d shmem, %d localmem, maxThreads=%d\n", name, numRegs, shmem, localmem, maxThreads); fclose(regLog); }
      }
    }
  }

  // Parse .maxntid from PTX to get __launch_bounds__ value.
  // PTX pattern: .visible .entry <name>(...)\n.maxntid N, 1, 1
  k->reqWorkGroupSize = 256; // fallback
  {
    const string& ptx = prog->ptx;
    string entryPattern = ".entry " + string(name) + "(";
    size_t pos = ptx.find(entryPattern);
    if (pos != string::npos) {
      // Found the kernel entry. Now find .maxntid before the next .entry or opening brace
      size_t searchEnd = ptx.find(".entry ", pos + 1);
      if (searchEnd == string::npos) searchEnd = ptx.size();
      string maxntidPattern = ".maxntid ";
      size_t mpos = ptx.find(maxntidPattern, pos);
      if (mpos != string::npos && mpos < searchEnd) {
        int val = atoi(ptx.c_str() + mpos + maxntidPattern.size());
        if (val > 0) {
          k->reqWorkGroupSize = val;
        }
      }
    }
  }

  if (err) *err = CL_SUCCESS;
  return k;
}

int clReleaseKernel(cl_kernel k) {
  delete k;
  return CL_SUCCESS;
}

int clSetKernelArg(cl_kernel k, unsigned pos, size_t size, const void* value) {
  if (!k) return CL_INVALID_KERNEL;

  // Detect cl_mem buffer arguments and convert to CUdeviceptr.
  // In OpenCL, buffer args are set with clSetKernelArg(k, i, sizeof(cl_mem), &memobj).
  // sizeof(cl_mem) == sizeof(void*) == 8 on 64-bit. The value at 'value' is a cl_mem pointer.
  // We need to store the CUdeviceptr (GPU address) instead of the cl_mem (host pointer).
  if (size == sizeof(cl_mem) && value) {
    cl_mem mem = *(cl_mem*)value;
    if (mem) {
      // Look the buffer up and copy its device pointer under the lock: another worker may be creating or releasing
      // buffers at the same time (an insert can rehash the set under a concurrent lookup, and a release deletes the
      // _cl_mem).
      CUdeviceptr devPtr = 0;
      bool isBuffer = false;
      {
        std::lock_guard<std::mutex> lock(g_allocatedBuffersMutex);
        if (g_allocatedBuffers.count(mem)) {
          isBuffer = true;
          devPtr = mem->ptr;
        }
      }
      if (isBuffer) {
        k->setArg(pos, sizeof(CUdeviceptr), &devPtr);
        return CL_SUCCESS;
      }
    }
    // NULL cl_mem → pass a null device pointer
    if (!mem) {
      CUdeviceptr devPtr = 0;
      k->setArg(pos, sizeof(CUdeviceptr), &devPtr);
      return CL_SUCCESS;
    }
  }

  k->setArg(pos, size, value);
  return CL_SUCCESS;
}

// ---- Buffer ----

cl_mem clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size, void* hostPtr, int* err) {
  auto* buf = new _cl_mem;
  buf->size = size;
  if (!ctx) ensureContextCurrent();
  // Allocate in the buffer's own context (the shim's current one when none is given) and remember it: the buffer is
  // freed there, whichever context is current when it is released.
  CudaContextScope owner(ctx ? ctx->ctx : nullptr);
  CUresult r = owner.status();
  if (r == CUDA_SUCCESS) r = cuCtxGetCurrent(&buf->owner);
  if (r == CUDA_SUCCESS) r = cuMemAlloc(&buf->ptr, size);
  if (r != CUDA_SUCCESS) {
    delete buf;
    if (err) *err = clStatusFromCu(r);
    return nullptr;
  }
  // Handle CL_MEM_COPY_HOST_PTR
  if ((flags & CL_MEM_COPY_HOST_PTR) && hostPtr) {
    r = cuMemcpyHtoD(buf->ptr, hostPtr, size);
    if (r != CUDA_SUCCESS) {   // never hand out a buffer that silently lacks its initial contents
      cuMemFree(buf->ptr);
      delete buf;
      if (err) *err = clStatusFromCu(r);
      return nullptr;
    }
  }
  {
    std::lock_guard<std::mutex> lock(g_allocatedBuffersMutex);
    g_allocatedBuffers.insert(buf);
  }
  if (err) *err = CL_SUCCESS;
  return buf;
}

int clReleaseMemObject(cl_mem buf) {
  int status = CL_SUCCESS;
  if (buf) {
    // Free in the context the memory was allocated in, not whichever one the shim made current last.  If that
    // context is gone, cuCtxDestroy already freed the memory.
    CudaContextScope owner(buf->owner);
    {
      std::lock_guard<std::mutex> lock(g_allocatedBuffersMutex);
      g_allocatedBuffers.erase(buf);
    }
    CUresult const r = owner.status() == CUDA_SUCCESS ? cuMemFree(buf->ptr) : owner.status();
    status = clStatusFromCu(r);
    delete buf;
  }
  return status;
}

// ---- Command Queue ----

cl_command_queue clCreateCommandQueueWithProperties(cl_context ctx, cl_device_id dev,
                                                     const cl_queue_properties* props, int* err) {
  auto* q = new _cl_command_queue;
  q->context = ctx;
  q->profiling = false;

  // Check for profiling flag
  if (props) {
    for (int i = 0; props[i]; i += 2) {
      if (props[i] == CL_QUEUE_PROPERTIES && (props[i+1] & CL_QUEUE_PROFILING_ENABLE)) {
        q->profiling = true;
      }
    }
  }

  // Make sure context is current
  cuCtxSetCurrent(ctx->ctx);
  q->owner = ctx->ctx;
  CUresult r = cuStreamCreate(&q->stream, CU_STREAM_NON_BLOCKING);
  if (r != CUDA_SUCCESS) {
    delete q;
    if (err) *err = clStatusFromCu(r);
    return nullptr;
  }
  if (err) *err = CL_SUCCESS;
  return q;
}

// ---- Enqueue operations ----

int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, unsigned workDim,
                            const size_t* globalOffset, const size_t* globalSize,
                            const size_t* localSize, unsigned nWaits,
                            const cl_event* waits, cl_event* event) {
  if (!q || !k) return CL_INVALID_VALUE;
  // The launch, its events and a graph capture all happen in the queue's context.
  if (CUresult r = makeCurrent(q->owner); r != CUDA_SUCCESS) return clStatusFromCu(r);

  size_t gs = globalSize[0];
  size_t ls = localSize ? localSize[0] : 256;
  size_t numBlocks = (gs + ls - 1) / ls;

  if (q->squareGraph && q->squareGraph->active()) {
    if (q->profiling || nWaits || workDim != 1 ||
        (globalOffset && globalOffset[0]))
      throw std::runtime_error("unsupported event/dimension in CUDA square graph");
    q->squareGraph->add(k, (unsigned)numBlocks, (unsigned)ls);
    if (event) *event = nullptr;
    return CL_SUCCESS;
  }

  // Build args array
  void* argPtrs[_cl_kernel::MAX_ARGS];
  k->buildArgPointers(argPtrs);

  // Env-gated kernel profiling (PRPLL_PROFILE=1) — takes priority over event profiling
  static bool doProfile = (getenv("PRPLL_PROFILE") != nullptr);

  // Event handling (skipped when env profiler is active)
  if (!doProfile && event && q->profiling) {
    auto* ev = new _cl_event;
    cuEventCreate(&ev->start, CU_EVENT_DEFAULT);
    cuEventCreate(&ev->end, CU_EVENT_DEFAULT);
    ev->hasTimings = true;
    ev->commandType = CL_COMMAND_NDRANGE_KERNEL;
    cuEventRecord(ev->start, q->stream);
    CUresult r = cuLaunchKernel(k->func, numBlocks, 1, 1, ls, 1, 1, 0, q->stream, argPtrs, nullptr);
    cuEventRecord(ev->end, q->stream);
    *event = ev;
    return clStatusFromCu(r);
  }
  if (doProfile) {
    static std::map<std::string, double> kTime;
    static std::map<std::string, int> kCount;
    static std::map<std::string, int> kRegs;
    static std::map<std::string, int> kShmem;
    static int totalLaunches = 0;
    static CUevent pStart = nullptr, pEnd = nullptr;
    if (!pStart) { cuEventCreate(&pStart, CU_EVENT_DEFAULT); cuEventCreate(&pEnd, CU_EVENT_DEFAULT); }

    cuEventRecord(pStart, q->stream);
    CUresult r = cuLaunchKernel(k->func, numBlocks, 1, 1, ls, 1, 1, 0, q->stream, argPtrs, nullptr);
    cuEventRecord(pEnd, q->stream);
    cuEventSynchronize(pEnd);
    float ms = 0;
    cuEventElapsedTime(&ms, pStart, pEnd);
    kTime[k->name] += ms;
    kCount[k->name]++;
    if (!kRegs.count(k->name)) {
      int regs = 0, shmem = 0;
      cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, k->func);
      cuFuncGetAttribute(&shmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, k->func);
      kRegs[k->name] = regs;
      kShmem[k->name] = shmem;
    }
    totalLaunches++;

    if (totalLaunches % 10000 == 0) {
      fprintf(stderr, "\n=== NTT KERNEL PROFILE (%d launches) ===\n", totalLaunches);
      std::vector<std::pair<double, std::string>> sorted;
      double totalMs = 0;
      for (auto& [n, t] : kTime) { sorted.push_back({t, n}); totalMs += t; }
      std::sort(sorted.rbegin(), sorted.rend());
      for (auto& [t, n] : sorted) {
        fprintf(stderr, "  %6.1f ms (%5.1f%%) %5d calls  avg %.3f ms  regs=%d shmem=%d  %s\n",
                t, 100.0*t/totalMs, kCount[n], t/kCount[n], kRegs[n], kShmem[n], n.c_str());
      }
      fprintf(stderr, "  TOTAL: %.1f ms\n===\n\n", totalMs);
    }
    if (event) *event = nullptr;
    return clStatusFromCu(r);
  }

  CUresult r = cuLaunchKernel(k->func, numBlocks, 1, 1, ls, 1, 1, 0, q->stream, argPtrs, nullptr);
  if (r != CUDA_SUCCESS) {
    const char* errName = nullptr;
    cuGetErrorName(r, &errName);
    fprintf(stderr, "cuLaunchKernel FAILED for '%s': %s (%d)\n", k->name.c_str(), errName ? errName : "?", (int)r);
  }

  if (event) *event = nullptr;
  return clStatusFromCu(r);
}

int clEnqueueReadBuffer(cl_command_queue q, cl_mem buf, cl_bool blocking,
                         size_t offset, size_t size, void* ptr,
                         unsigned nWaits, const cl_event* waits, cl_event* event) {
  // Must use stream-ordered copy because the stream was created with CU_STREAM_NON_BLOCKING,
  // which means cuMemcpyDtoH (NULL stream) won't wait for pending kernels on this stream.
  CUresult r = makeCurrent(q->owner);
  if (r == CUDA_SUCCESS) r = cuMemcpyDtoHAsync(ptr, buf->ptr + offset, size, q->stream);
  if (r == CUDA_SUCCESS && blocking) {
    r = cuStreamSynchronize(q->stream);
  }
  if (event) *event = nullptr;
  return clStatusFromCu(r);
}

int clEnqueueWriteBuffer(cl_command_queue q, cl_mem buf, cl_bool blocking,
                          size_t offset, size_t size, const void* ptr,
                          unsigned nWaits, const cl_event* waits, cl_event* event) {
  // Must use stream-ordered copy (same reason as clEnqueueReadBuffer above)
  CUresult r = makeCurrent(q->owner);
  if (r == CUDA_SUCCESS) r = cuMemcpyHtoDAsync(buf->ptr + offset, ptr, size, q->stream);
  if (r == CUDA_SUCCESS && blocking) {
    r = cuStreamSynchronize(q->stream);
  }
  if (event) *event = nullptr;
  return clStatusFromCu(r);
}

int clEnqueueCopyBuffer(cl_command_queue q, cl_mem src, cl_mem dst,
                         size_t srcOffset, size_t dstOffset, size_t size,
                         unsigned nWaits, const cl_event* waits, cl_event* event) {
  CUresult r = makeCurrent(q->owner);
  if (r == CUDA_SUCCESS) r = cuMemcpyDtoDAsync(dst->ptr + dstOffset, src->ptr + srcOffset, size, q->stream);
  if (event) *event = nullptr;
  return clStatusFromCu(r);
}

int clEnqueueFillBuffer(cl_command_queue q, cl_mem buf, const void* pattern,
                         size_t patternSize, size_t offset, size_t size,
                         unsigned nWaits, const cl_event* waits, cl_event* event) {
  if (event) *event = nullptr;
  if (!q || !buf) return CL_INVALID_VALUE;
  // The CUDA driver memsets 8, 16 or 32-bit values only; CudaFill.h plans the call(s) for any pattern size OpenCL
  // allows, and rejects what OpenCL rejects instead of writing something the caller did not ask for.
  cuda_fill::Plan const p = cuda_fill::plan(pattern, patternSize, offset, size, buf->size);
  CUdeviceptr const dst = buf->ptr + p.offset;
  if (p.op == cuda_fill::Op::Invalid) return CL_INVALID_VALUE;
  CUresult r = makeCurrent(q->owner);
  if (r != CUDA_SUCCESS) return clStatusFromCu(r);
  switch (p.op) {
  case cuda_fill::Op::Invalid:
    return CL_INVALID_VALUE;
  case cuda_fill::Op::Memset8: {
    unsigned char val;
    memcpy(&val, pattern, 1);
    r = cuMemsetD8Async(dst, val, p.count, q->stream);
    break;
  }
  case cuda_fill::Op::Memset16: {
    unsigned short val;
    memcpy(&val, pattern, 2);
    r = cuMemsetD16Async(dst, val, p.count, q->stream);
    break;
  }
  case cuda_fill::Op::Memset32:
    r = cuMemsetD32Async(dst, cuda_fill::patternWord(pattern, 0), p.count, q->stream);
    break;
  case cuda_fill::Op::Strided32:
    // Word i of every pattern-sized slot: pitch = the pattern, width = one element, height = the slot count.
    for (unsigned i = 0; i < p.words && r == CUDA_SUCCESS; ++i) {
      r = cuMemsetD2D32Async(dst + 4 * i, p.pitch, cuda_fill::patternWord(pattern, i), 1, p.count, q->stream);
    }
    break;
  }
  return clStatusFromCu(r);
}

int clEnqueueMarkerWithWaitList(cl_command_queue q, unsigned nWaits, const cl_event* waits, cl_event* event) {
  // The marker's event is created in the current context, which must be the queue's.
  if (CUresult r = makeCurrent(q->owner); r != CUDA_SUCCESS) {
    if (event) *event = nullptr;
    return clStatusFromCu(r);
  }
  if (nWaits) {
    for (unsigned int i = 0; i < nWaits; ++i) {
      cuStreamWaitEvent(q->stream, waits[i]->end, 0);
    }
  }
  if (event) {
    auto* ev = new _cl_event;
    cuEventCreate(&ev->end, CU_EVENT_DEFAULT);
    cuEventRecord(ev->end, q->stream);
    ev->commandType = CL_COMMAND_MARKER;
    *event = ev;
  }
  return CL_SUCCESS;
}

int clFlush(cl_command_queue q) {
  // CUDA streams auto-flush; no-op
  return CL_SUCCESS;
}

int clFinish(cl_command_queue q) {
  // A kernel that faulted leaves a sticky error that cuStreamSynchronize returns: report it here, where the work
  // is waited for, rather than as the failure of whichever copy comes next.
  if (!q) return CL_SUCCESS;
  CUresult r = makeCurrent(q->owner);
  if (r == CUDA_SUCCESS) r = cuStreamSynchronize(q->stream);
  return clStatusFromCu(r);
}

// ---- Events ----

int clReleaseEvent(cl_event ev) {
  delete ev;
  return CL_SUCCESS;
}

int clWaitForEvents(unsigned n, const cl_event* events) {
  int status = CL_SUCCESS;
  for (unsigned i = 0; i < n; i++) {
    if (events[i] && events[i]->end) {
      CUresult r = makeCurrent(events[i]->owner);
      if (r == CUDA_SUCCESS) r = cuEventSynchronize(events[i]->end);
      int const st = clStatusFromCu(r);
      if (st != CL_SUCCESS && status == CL_SUCCESS) status = st;  // keep waiting for the rest, report the first error
    }
  }
  return status;
}

int clGetEventInfo(cl_event ev, cl_event_info info, size_t size, void* value, size_t* sizeRet) {
  if (!ev) return CL_INVALID_VALUE;
  if (info == CL_EVENT_COMMAND_EXECUTION_STATUS) {
    int status = CL_COMPLETE;
    if (ev->end) {
      // NOT_READY is running; any other failure is the sticky error of a faulted context and is reported as the
      // (negative) status of the command, not as complete.
      CUresult r = makeCurrent(ev->owner);
      if (r == CUDA_SUCCESS) r = cuEventQuery(ev->end);
      status = cuda_error_map::eventStatusFromCu((int) r);
    }
    if (sizeRet) *sizeRet = sizeof(int);
    if (value && size >= sizeof(int)) memcpy(value, &status, sizeof(int));
  } else if (info == CL_EVENT_COMMAND_TYPE) {
    u32 type = ev->commandType;
    if (sizeRet) *sizeRet = sizeof(u32);
    if (value && size >= sizeof(u32)) memcpy(value, &type, sizeof(u32));
  }
  return CL_SUCCESS;
}

int clGetEventProfilingInfo(cl_event ev, cl_profiling_info info, size_t size, void* value, size_t* sizeRet) {
  if (!ev || !ev->hasTimings) return CL_PROFILING_INFO_NOT_AVAILABLE;

  // CUDA events give elapsed time between two events, not absolute timestamps.
  // We fake absolute timestamps by using a base time.
  u64 timestamp = 0;
  if (info == CL_PROFILING_COMMAND_START || info == CL_PROFILING_COMMAND_SUBMIT ||
      info == CL_PROFILING_COMMAND_QUEUED) {
    timestamp = 0;  // Relative start
  } else if (info == CL_PROFILING_COMMAND_END || info == CL_PROFILING_COMMAND_COMPLETE) {
    float ms = 0;
    if (makeCurrent(ev->owner) == CUDA_SUCCESS) cuEventElapsedTime(&ms, ev->start, ev->end);
    timestamp = (u64)(ms * 1e6);  // Convert ms to ns
  }

  if (sizeRet) *sizeRet = sizeof(u64);
  if (value && size >= sizeof(u64)) memcpy(value, &timestamp, sizeof(u64));
  return CL_SUCCESS;
}

// ---- Device info ----

int clGetDeviceInfo(cl_device_id dev, cl_device_info info, size_t size, void* value, size_t* sizeRet) {
  if (!dev) return CL_INVALID_DEVICE;

  switch (info) {
  case CL_DEVICE_NAME: {
    char name[256];
    cuDeviceGetName(name, sizeof(name), dev->dev);
    size_t len = strlen(name) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, name, len);
    break;
  }
  case CL_DEVICE_VENDOR_ID: {
    // Return NVIDIA vendor ID
    unsigned int vid = 0x10DE;
    if (sizeRet) *sizeRet = sizeof(vid);
    if (value && size >= sizeof(vid)) memcpy(value, &vid, sizeof(vid));
    break;
  }
  case CL_DEVICE_MAX_COMPUTE_UNITS: {
    int units = 0;
    cuDeviceGetAttribute(&units, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev->dev);
    unsigned int val = units;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_MAX_CLOCK_FREQUENCY: {
    int mhz = 0;
    cuDeviceGetAttribute(&mhz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev->dev);
    unsigned int val = mhz / 1000;  // kHz to MHz
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_MAX_WORK_GROUP_SIZE: {
    int threads = 0;
    cuDeviceGetAttribute(&threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev->dev);
    size_t val = threads;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_LOCAL_MEM_SIZE: {
    // Kernels declare their shared memory statically, which is limited to this (48KB) without an opt-in
    int shared = 0;
    cuDeviceGetAttribute(&shared, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev->dev);
    u64 val = shared;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_GLOBAL_MEM_SIZE: {
    size_t mem = 0;
    cuDeviceTotalMem(&mem, dev->dev);
    u64 val = mem;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DRIVER_VERSION:
  case CL_DEVICE_VERSION: {
    int ver = 0;
    cuDriverGetVersion(&ver);
    char verStr[64];
    snprintf(verStr, sizeof(verStr), "CUDA %d.%d", ver / 1000, (ver % 1000) / 10);
    size_t len = strlen(verStr) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, verStr, len);
    break;
  }
  case CL_DEVICE_ERROR_CORRECTION_SUPPORT: {
    int ecc = 0;
    cuDeviceGetAttribute(&ecc, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev->dev);
    cl_bool val = ecc;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
    break;
  }
  case CL_DEVICE_BUILT_IN_KERNELS: {
    const char* empty = "";
    if (sizeRet) *sizeRet = 1;
    if (value && size >= 1) memcpy(value, empty, 1);
    break;
  }
  case CL_DEVICE_TOPOLOGY_AMD: {
    // The device's PCIe position in the AMD extension's shape, so `-pci <bus:device.function>` selects a CUDA device the way it
    // selects an AMD one (gpuid.cpp getPosFromBdf -> clwrap.cpp getBdfFromDevice).  The enumeration ordinal `-device N` names is
    // not stable (CUDA orders FASTEST_FIRST unless CUDA_DEVICE_ORDER says otherwise, and a second card or a driver update can
    // renumber), while the bus id is what nvidia-smi and every launcher already know the card by.
    char bdf[32] = {0};
    if (cuDeviceGetPCIBusId(bdf, sizeof(bdf), dev->dev) != CUDA_SUCCESS) return CL_INVALID_VALUE;
    // "0000:6a:00.0" is domain:bus:device.function; tolerate a missing domain.
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    if (sscanf(bdf, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4 &&
        sscanf(bdf, "%x:%x.%x", &bus, &device, &function) != 3) {
      return CL_INVALID_VALUE;
    }
    cl_device_topology_amd top{};
    top.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
    top.pcie.bus = (char) bus;
    top.pcie.device = (char) device;
    top.pcie.function = (char) function;
    if (sizeRet) *sizeRet = sizeof(top);
    if (value && size >= sizeof(top)) memcpy(value, &top, sizeof(top));
    break;
  }
  case CL_DEVICE_BOARD_NAME_AMD:
  case CL_DEVICE_PCIE_ID_AMD: {
    // AMD-specific queries with no CUDA counterpart — return failure
    return CL_INVALID_VALUE;
  }
  case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: {
    size_t freeMem = 0, totalMem = 0;
    cuMemGetInfo(&freeMem, &totalMem);
    // AMD returns in KB
    u64 freeKB = freeMem / 1024;
    if (sizeRet) *sizeRet = sizeof(freeKB);
    if (value && size >= sizeof(freeKB)) memcpy(value, &freeKB, sizeof(freeKB));
    break;
  }
  case CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV: {
    int major = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev->dev);
    if (sizeRet) *sizeRet = sizeof(major);
    if (value && size >= sizeof(major)) memcpy(value, &major, sizeof(major));
    break;
  }
  case CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV: {
    int minor = 0;
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev->dev);
    if (sizeRet) *sizeRet = sizeof(minor);
    if (value && size >= sizeof(minor)) memcpy(value, &minor, sizeof(minor));
    break;
  }
  default:
    return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

int clGetPlatformInfo(cl_platform_id, cl_device_info info, size_t size, void* value, size_t* sizeRet) {
  if (info == CL_PLATFORM_VERSION) {
    const char* ver = "CUDA (via PRPLL CUDA backend)";
    size_t len = strlen(ver) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, ver, len);
    return CL_SUCCESS;
  }
  return CL_INVALID_VALUE;
}

int clGetCommandQueueInfo(cl_command_queue q, cl_command_queue_info info,
                           size_t size, void* value, size_t* sizeRet) {
  if (info == CL_QUEUE_CONTEXT) {
    if (sizeRet) *sizeRet = sizeof(cl_context);
    if (value && size >= sizeof(cl_context)) memcpy(value, &q->context, sizeof(cl_context));
    return CL_SUCCESS;
  }
  return CL_INVALID_VALUE;
}

// ---- Kernel info ----

int clGetKernelInfo(cl_kernel k, cl_kernel_info info, size_t size, void* value, size_t* sizeRet) {
  if (!k) return CL_INVALID_KERNEL;
  if (info == CL_KERNEL_NUM_ARGS) {
    int n = k->numArgs;
    if (sizeRet) *sizeRet = sizeof(n);
    if (value && size >= sizeof(n)) memcpy(value, &n, sizeof(n));
  } else if (info == CL_KERNEL_ATTRIBUTES) {
    const char* empty = "";
    if (sizeRet) *sizeRet = 1;
    if (value && size >= 1) memcpy(value, empty, 1);
  }
  return CL_SUCCESS;
}

int clGetKernelArgInfo(cl_kernel k, unsigned pos, cl_kernel_arg_info info,
                        size_t size, void* value, size_t* sizeRet) {
  if (info == CL_KERNEL_ARG_NAME) {
    char name[32];
    snprintf(name, sizeof(name), "arg%u", pos);
    size_t len = strlen(name) + 1;
    if (sizeRet) *sizeRet = len;
    if (value && size >= len) memcpy(value, name, len);
  }
  return CL_SUCCESS;
}

int clGetKernelWorkGroupInfo(cl_kernel k, cl_device_id dev, cl_kernel_work_group_info info,
                              size_t size, void* value, size_t* sizeRet) {
  if (!k) return CL_INVALID_KERNEL;
  if (CUresult r = makeCurrent(k->owner); r != CUDA_SUCCESS) return clStatusFromCu(r);
  if (info == CL_KERNEL_COMPILE_WORK_GROUP_SIZE) {
    // Return the __launch_bounds__ value parsed from source during clCreateKernel.
    // This matches OpenCL's CL_KERNEL_COMPILE_WORK_GROUP_SIZE which returns reqd_work_group_size.
    // Previously we used CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK which returns the hardware max
    // based on register/shared memory usage — NOT the declared group size. This caused wrong
    // block sizes for every kernel (e.g., tailMul expected 64 threads but got 1024).
    int wgSize = k->reqWorkGroupSize > 0 ? k->reqWorkGroupSize : 256;
    size_t wgs[3] = { (size_t)wgSize, 1, 1 };
    if (sizeRet) *sizeRet = sizeof(wgs);
    if (value && size >= sizeof(wgs)) memcpy(value, wgs, sizeof(wgs));
  } else if (info == CL_KERNEL_WORK_GROUP_SIZE) {
    // The largest block the compiled kernel can be launched with.  Register use can make this smaller than
    // the __launch_bounds__ value (e.g. 768 for a 1024-thread carryFused).
    int maxThreads = 0;
    cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, k->func);
    size_t const val = maxThreads;
    if (sizeRet) *sizeRet = sizeof(val);
    if (value && size >= sizeof(val)) memcpy(value, &val, sizeof(val));
  }
  return CL_SUCCESS;
}

// ---- SVM (not used but must exist) ----

void* clSVMAlloc(cl_context ctx, cl_svm_mem_flags, size_t size, unsigned) {
  CUdeviceptr ptr = 0;
  if (!ctx) ensureContextCurrent();
  CudaContextScope owner(ctx ? ctx->ctx : nullptr);
  if (owner.status() != CUDA_SUCCESS || cuMemAlloc(&ptr, size) != CUDA_SUCCESS) return nullptr;
  return (void*)(uintptr_t)ptr;
}

void clSVMFree(cl_context ctx, void* ptr) {
  if (!ptr) return;
  if (!ctx) ensureContextCurrent();
  CudaContextScope owner(ctx ? ctx->ctx : nullptr);
  if (owner.status() == CUDA_SUCCESS) cuMemFree((CUdeviceptr)(uintptr_t)ptr);
}

int clSetKernelArgSVMPointer(cl_kernel k, unsigned pos, const void* ptr) {
  CUdeviceptr dp = (CUdeviceptr)(uintptr_t)ptr;
  k->setArg(pos, sizeof(dp), &dp);
  return CL_SUCCESS;
}

} // extern "C"

// C++ linkage — must be outside the extern "C" block above.

bool cudaLoadRegLimitedKernel(cl_program prog, const char* kernelName, int maxRegs, int minBlocks) {
  if (!prog || !prog->moduleLoaded || prog->kernelExported)
    throw std::runtime_error("PTX register limit must precede kernel creation");
  // The replacement module belongs to the program's context, like the one it replaces.
  if (CUresult r = makeCurrent(prog->owner); r != CUDA_SUCCESS) {
    fprintf(stderr, "%s: cannot make the program's CUDA context current (%d)\n", kernelName, int(r));
    return false;
  }
  const string limited = aevumPtxWithRegisterLimit(prog->ptx, kernelName, maxRegs, minBlocks);
  CUmodule replacement{};
  char errors[8192] = {};
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
  void* values[] = {(void*)(size_t)sizeof(errors), errors};
  const CUresult result = cuModuleLoadDataEx(&replacement, limited.c_str(), 2, options, values);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "%s: PTX register-limit JIT failed (%d): %s\n", kernelName, int(result), errors);
    return false; // Keep the original program intact; caller fails instead of running it.
  }
  // No kernel has escaped from this freshly loaded program (kernelExported), so the program holds the only reference
  // to the neutral module: dropping it unloads it, only after the replacement loaded successfully.
  prog->moduleRef = adoptModule(replacement);
  prog->module = replacement;
  // Do not replace prog->ptx: saveBinary must only persist the neutral image.
  fprintf(stderr, "%s: PTX register limit active (maxnreg=%d, minnctapersm=%d)\n", kernelName, maxRegs, minBlocks);
  return true;
}

void cudaGraphBatchBegin(cl_command_queue q) {
  if (!q || q->profiling) throw std::runtime_error("CUDA square graph requires an unprofiled queue");
  if (makeCurrent(q->owner) != CUDA_SUCCESS) throw std::runtime_error("CUDA square graph: queue context is gone");
  if (!q->squareGraph) q->squareGraph = new AevumCudaGraphBatch;
  q->squareGraph->begin();
}
void cudaGraphBatchEnd(cl_command_queue q) {
  if (!q || !q->squareGraph) throw std::runtime_error("CUDA square graph not begun");
  if (makeCurrent(q->owner) != CUDA_SUCCESS) throw std::runtime_error("CUDA square graph: queue context is gone");
  q->squareGraph->launch(q->stream);
}
void cudaGraphBatchAbort(cl_command_queue q) noexcept {
  if (q && q->squareGraph) q->squareGraph->abort();
}


// Set L2 cache persistence for multiple read-only buffers on the given stream.
// Computes the minimum address span covering all buffers, then sets one access policy
// window with hitRatio sized so that only the actual buffer bytes get persisting treatment,
// not the gaps between non-contiguous allocations.
void cudaSetL2Persistent(cl_command_queue q, const std::vector<cl_mem>& buffers) {
  if (!q) return;
  if (makeCurrent(q->owner) != CUDA_SUCCESS) return;

  // Find address span and total data size
  CUdeviceptr minAddr = ~(CUdeviceptr)0;
  CUdeviceptr maxAddr = 0;
  size_t totalDataBytes = 0;

  for (auto buf : buffers) {
    if (!buf || buf->size == 0) continue;
    CUdeviceptr lo = buf->ptr;
    CUdeviceptr hi = buf->ptr + buf->size;
    if (lo < minAddr) minAddr = lo;
    if (hi > maxAddr) maxAddr = hi;
    totalDataBytes += buf->size;
  }

  if (totalDataBytes == 0 || maxAddr <= minAddr) return;

  size_t spanBytes = (size_t)(maxAddr - minAddr);

  // Query the device's max access policy window size
  int maxWindowSize = 0;
  cuDeviceGetAttribute(&maxWindowSize, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, 0);
  if (maxWindowSize > 0 && spanBytes > (size_t)maxWindowSize) {
    fprintf(stderr, "L2 persist: span %zuMB exceeds max window %dMB, clamping\n",
            spanBytes / (1024*1024), maxWindowSize / (1024*1024));
    spanBytes = maxWindowSize;
  }

  // hitRatio = actual data / window span. This way only the real buffer data gets
  // persisting treatment, and any gaps between allocations get streaming treatment.
  float hitRatio = (float)totalDataBytes / (float)spanBytes;
  if (hitRatio > 1.0f) hitRatio = 1.0f;

  CUstreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = (void*)(uintptr_t)minAddr;
  attr.accessPolicyWindow.num_bytes = spanBytes;
  attr.accessPolicyWindow.hitRatio = hitRatio;
  attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
  attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;

  CUresult r = cuStreamSetAttribute(q->stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "L2 persist: cuStreamSetAttribute failed (%d)\n", (int)r);
  } else {
    fprintf(stderr, "L2 persist: window %zuMB (%.1f%% hit ratio), %zuMB actual data, %zu buffers\n",
            spanBytes / (1024*1024), hitRatio * 100.0f, totalDataBytes / (1024*1024),
            buffers.size());
  }
}

