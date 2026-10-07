// Host test: the GF61 complex multiply kernels (stock, and the AEVUM_GF61_LIMB32 limb version) must build and match exact arithmetic.
//
// math.cl selects the cmul body by AEVUM_GF61_LIMB32; the limb version once lacked its closing brace, so every GF61 kernel failed to build
// with AEVUM_GF61_LIMB32=1.  The harness compiles the real math.cl for both settings on any OpenCL device and compares cmul on random
// operands with a reference computed in 128-bit integers modulo 2^61 - 1.
//
// usage: gf61_limb32_cmul_kernel_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* HARNESS = R"CL(
#include "base.cl"
#include "math.cl"
KERNEL(64) tcmul(global ulong2* a, global ulong2* b, global ulong2* o) {
  u32 i = get_global_id(0);
  GF61 r = cmul(U2(a[i].x, a[i].y), U2(b[i].x, b[i].y));
  o[i] = (ulong2)(r.x, r.y);
}
)CL";

typedef unsigned long long u64_t;
typedef unsigned __int128 u128_t;
static const u64_t M61 = (1ull << 61) - 1;

static u64_t mulmod(u64_t a, u64_t b) {
  u128_t p = (u128_t) a * b;
  u64_t r = (u64_t) (p & M61) + (u64_t) (p >> 61);
  r = (r & M61) + (r >> 61);
  return r >= M61 ? r - M61 : r;
}
static u64_t addmod(u64_t a, u64_t b) { u64_t r = a + b; return r >= M61 ? r - M61 : r; }
static u64_t submod(u64_t a, u64_t b) { return a >= b ? a - b : a + M61 - b; }

static bool runCase(cl_device_id dev, cl_context ctx, cl_queue q, const std::string& dir, int limb32) {
  Program prog = loadSource(ctx, HARNESS);
  std::string opts = "-cl-std=CL1.2 -I " + dir + " -DNO_ASM=1 -DAMDGPU=0 -DFFT_TYPE=3 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=1 -DWordSize=8u "
    "-DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u -DEXP=1257787u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=4u -DNH=4u -DFFT_VARIANT=101u "
    "-DFRAC_BPW_HI=1u -DFRAC_BPW_LO=4294967295u -DTAILTGF61=U2(1ul,1ul) -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul "
    "-DAEVUM_GF61_LIMB32=" + std::to_string(limb32);
  if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::string log = getBuildLog(prog.get(), dev);
    size_t e = log.find("error:");
    std::printf("AEVUM_GF61_LIMB32=%d: FAIL: build: %s\n", limb32, (e == std::string::npos ? log.substr(0, 200) : log.substr(e, 200)).c_str());
    return false;
  }
  int err = 0;
  cl_kernel k = clCreateKernel(prog.get(), "tcmul", &err);
  CHECK1(err);
  const size_t n = 4096;
  cl_mem ba = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 16, nullptr, &err);
  CHECK1(err);
  cl_mem bb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 16, nullptr, &err);
  CHECK1(err);
  cl_mem bo = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 16, nullptr, &err);
  CHECK1(err);
  std::vector<u64_t> a(2 * n), b(2 * n), o(2 * n);
  srand(7);
  auto rnd = [] { return ((u64_t) rand() << 40 ^ (u64_t) rand() << 20 ^ (u64_t) rand()) % M61; };
  for (size_t i = 0; i < 2 * n; ++i) { a[i] = rnd(); b[i] = rnd(); }
  a[0] = a[1] = b[0] = b[1] = M61 - 1;                     // largest canonical operands
  a[2] = 0; a[3] = 1; b[2] = M61 - 1; b[3] = 0;
  CHECK1(clEnqueueWriteBuffer(q, ba, 1, 0, n * 16, a.data(), 0, nullptr, nullptr));
  CHECK1(clEnqueueWriteBuffer(q, bb, 1, 0, n * 16, b.data(), 0, nullptr, nullptr));
  CHECK1(clSetKernelArg(k, 0, sizeof ba, &ba));
  CHECK1(clSetKernelArg(k, 1, sizeof bb, &bb));
  CHECK1(clSetKernelArg(k, 2, sizeof bo, &bo));
  size_t g = n, l = 64;
  CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
  CHECK1(clEnqueueReadBuffer(q, bo, 1, 0, n * 16, o.data(), 0, nullptr, nullptr));
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    u64_t ar = a[2 * i], ai = a[2 * i + 1], br = b[2 * i], bi = b[2 * i + 1];
    u64_t re = submod(mulmod(ar, br), mulmod(ai, bi)), im = addmod(mulmod(ar, bi), mulmod(ai, br));
    if (o[2 * i] % M61 != re || o[2 * i + 1] % M61 != im) ++bad;
  }
  std::printf("AEVUM_GF61_LIMB32=%d: %zu of %zu products wrong  %s\n", limb32, bad, n, bad ? "FAIL" : "ok");
  return bad == 0;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);
  bool ok = true;
  ok &= runCase(dev, ctx, q, dir, 0);
  ok &= runCase(dev, ctx, q, dir, 1);
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
