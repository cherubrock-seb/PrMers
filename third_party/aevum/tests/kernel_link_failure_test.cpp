// Device test (any OpenCL device; run it on PoCL): KernelCompiler's handling of clLinkProgram.
//
// clLinkProgram may hand back a program object even when it fails (so that its build log can be read).  The test
// interposes clLinkProgram and clGetProgramBuildInfo: the real link runs, and the wrapper can then report
// CL_LINK_PROGRAM_FAILURE while still returning that program.  It checks that
//   1. after the link, the build log is read from the linked program (where the linker's diagnostics are), and
//   2. a failed link is reported as "Can't compile <file>" instead of the half-linked program being used.
//
// usage: kernel_link_failure_test
//
// Linux only: it relies on symbol interposition with dlsym(RTLD_NEXT), and Apple builds one monolithic program
// without a link step.

#include <cstdio>

#if defined(__APPLE__) || defined(_WIN32)
int main() { std::printf("SKIP: kernel link failure test needs ELF symbol interposition and a separate link step\n"); return 0; }
#else

#include "Args.h"
#include "Context.h"
#include "KernelCompiler.h"

#include <cstdio>
#include <dlfcn.h>
#include <set>
#include <string>

static bool g_failLink = false;
static cl_program g_linked = nullptr;
static std::set<cl_program> g_logQueried;

extern "C" {
cl_program clLinkProgram(cl_context ctx, unsigned nDevices, const cl_device_id* devices, const char* options, unsigned nProgs,
                         const cl_program* progs, void (*notify)(cl_program, void*), void* userData, int* err) {
  using Fn = cl_program (*)(cl_context, unsigned, const cl_device_id*, const char*, unsigned, const cl_program*,
                            void (*)(cl_program, void*), void*, int*);
  static Fn real = (Fn) dlsym(RTLD_NEXT, "clLinkProgram");
  cl_program p = real(ctx, nDevices, devices, options, nProgs, progs, notify, userData, err);
  g_linked = p;
  if (g_failLink && err) { *err = CL_LINK_PROGRAM_FAILURE; }
  return p;
}

int clGetProgramBuildInfo(cl_program prog, cl_device_id dev, cl_program_build_info info, size_t size, void* value, size_t* sizeRet) {
  using Fn = int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
  static Fn real = (Fn) dlsym(RTLD_NEXT, "clGetProgramBuildInfo");
  if (info == CL_PROGRAM_BUILD_LOG) { g_logQueried.insert(prog); }
  return real(prog, dev, info, size, value, sizeRet);
}
}

static const std::string DEFINES =
  "-DEXP=20000033u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DCARRY_LEN=8u -DNW=4u -DNH=4u -DPFA_RADIX=0u -DFFT_VARIANT=101u "
  "-DMAXBPW=3000u -DFFT_TYPE=1 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=1 -DWordSize=8u -DCARRY64=1 -DNO_ASM=1 "
  "-DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0 -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul "
  "-DDISTHTRIGGF61=0ul -DFRAC_BPW_HI=0u -DFRAC_BPW_LO=0u -DTAILTGF31=U2(1u,1u) -DTAILTGF61=U2(1ul,1ul)";

int main() {
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  Args args(true);
  Context context{devices[0]};
  KernelCompiler compiler{args, &context, DEFINES};
  int failures = 0;

  // A good link: the kernel loads, and the link's own log was read from the linked program.
  try {
    KernelHolder k = compiler.load("etc.cl", "isEqual", "-DISEQUAL=1").get();
    if (!k) { std::printf("FAIL: isEqual did not load\n"); ++failures; }
  } catch (...) {
    std::printf("FAIL: a good link threw\n");
    ++failures;
  }
  if (!g_linked || !g_logQueried.count(g_linked)) {
    std::printf("FAIL: the build log was not read from the linked program\n");
    ++failures;
  }

  // A failed link that still returns a program object must take the "Can't compile" path.
  g_failLink = true;
  bool refused = false;
  try {
    KernelHolder k = compiler.load("etc.cl", "isEqual", "-DISEQUAL=1 -DLINK_FAILURE_CASE=1").get();
  } catch (const std::string& e) {
    refused = e.find("Can't compile etc.cl") != std::string::npos;
    if (!refused) std::printf("FAIL: unexpected error '%s'\n", e.c_str());
  } catch (...) {
    std::printf("FAIL: unexpected exception type\n");
  }
  if (!refused) { std::printf("FAIL: a failed link was used as a compiled program\n"); ++failures; }

  std::printf("%s\n", failures ? "FAIL" : "kernel link failure test ok");
  return failures ? 1 : 0;
}
#endif
