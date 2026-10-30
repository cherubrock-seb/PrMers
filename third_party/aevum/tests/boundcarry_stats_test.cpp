// Device test (any OpenCL device; run it on PoCL): boundCarry(i64), which maps a 64-bit carry to the STATS carry statistic.
//
// The statistic is scaled so that a carry at the overflow limit reads 0.5, like a round-off of 0.5.  FFT64 and FFT3231
// can also run with 32-bit carries, so their 64-bit carries are measured against CARRY32's 2^31 (for -carryTune).  The
// other types only have 64-bit carries, whose limit is 2^63; measuring those against 2^31 too kept only bits 8..39 of the
// carry, so a large carry read as anything from 0 to 0.5.
//
// The harness compiles the real carryutil.cl for several FFT types and evaluates boundCarry on chosen carries.
//
// usage: boundcarry_stats_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static const char* HARNESS =
  "#include \"base.cl\"\n"
  "#include \"math.cl\"\n"
  "#include \"carryutil.cl\"\n"
  "kernel void probe(global const long* in, global float* out) { u32 i = get_global_id(0); out[i] = boundCarry((i64) in[i]); }\n";

struct Case { long long carry; float type64; float type1; };   // expected boundCarry for FFT64/FFT3231 and for the other types

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  const long long T31 = 1ll << 31, T62 = 1ll << 62;
  const std::vector<Case> cases = {
    {0, 0.0f, 0.0f},
    {T31, 0.5f, 0.0f},                     // CARRY32's limit
    {-T31, 0.5f, 0.0f},
    {1ll << 36, 16.0f, 0.0f},              // 32-bit-capable types: 2^36 / 2^32
    {T62, -1.0f, 0.25f},                   // 64-bit-only types: 2^62 / 2^64 (-1: not checked, outside the 32-bit-capable range)
    {-T62, -1.0f, 0.25f},
    {(long long) (~0ull >> 1), -1.0f, 0.5f},   // 2^63 - 1
    {(long long) (1ull << 63), -1.0f, 0.5f},   // -2^63
    {3ll << 60, -1.0f, 0.1875f},
  };
  std::vector<long long> in;
  for (const Case& c : cases) in.push_back(c.carry);
  const size_t n = in.size();

  struct Type { int fftType; const char* defs; bool carry32Capable; };
  const Type types[] = {
    {0,  "-DFFT_TYPE=0 -DFFT_FP64=1 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=0 -DWordSize=4u", true},
    {1,  "-DFFT_TYPE=1 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=1 -DWordSize=8u", false},
    {4,  "-DFFT_TYPE=4 -DFFT_FP64=0 -DFFT_FP32=1 -DNTT_GF31=1 -DNTT_GF61=1 -DWordSize=8u", false},
    {50, "-DFFT_TYPE=50 -DFFT_FP64=0 -DFFT_FP32=1 -DNTT_GF31=1 -DNTT_GF61=0 -DWordSize=4u", true},
  };

  int failures = 0;
  for (const Type& t : types) {
    Program prog = loadSource(ctx, HARNESS);
    // This focused harness uses only OpenCL 1.2 features; forcing CL3.0 rejects otherwise valid AMD runtimes.
    std::string opts = std::string("-cl-std=CL1.2 -I ") + dir + " -DNO_ASM=1 -DCARRY64=1 -DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3000u "
      "-DEXP=20000033u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=4u -DNH=4u -DFFT_VARIANT=101u "
      "-DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0 -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul "
      "-DFRAC_BPW_HI=0u -DFRAC_BPW_LO=0u -DTAILT=U2(1.0f,0.0f) -DTAILTGF31=U2(1u,1u) -DTAILTGF61=U2(1ul,1ul) " + std::string(t.defs);
    if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
      std::printf("FAIL: build for FFT_TYPE=%d\n%s\n", t.fftType, getBuildLog(prog.get(), dev).c_str());
      return 1;
    }
    int err = 0;
    cl_kernel k = clCreateKernel(prog.get(), "probe", &err);
    CHECK1(err);
    KernelHolder kHold{k};
    cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * 8, in.data(), &err);
    CHECK1(err);
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 4, nullptr, &err);
    CHECK1(err);
    std::unique_ptr<cl_mem> binHold{bin}, boutHold{bout};
    CHECK1(clSetKernelArg(k, 0, sizeof bin, &bin));
    CHECK1(clSetKernelArg(k, 1, sizeof bout, &bout));
    size_t global = n;
    CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
    std::vector<float> out(n);
    CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, n * 4, out.data(), 0, nullptr, nullptr));
    for (size_t i = 0; i < n; ++i) {
      float want = t.carry32Capable ? cases[i].type64 : cases[i].type1;
      if (want < 0) continue;
      bool ok = std::fabs(out[i] - want) <= 1e-6f * std::max(1.0f, want);
      if (!ok) {
        ++failures;
        std::printf("FAIL: FFT_TYPE=%d boundCarry(%lld) = %g, expected %g\n", t.fftType, cases[i].carry, out[i], want);
      }
    }
  }
  std::printf("%s\n", failures ? "FAIL" : "boundCarry stats test ok");
  return failures ? 1 : 0;
}
