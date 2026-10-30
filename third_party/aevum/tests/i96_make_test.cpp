// Device test (any OpenCL device; run it on PoCL): every i96 implementation in math.cl builds the same values.
//
// math.cl has three i96 implementations; "#if 1" selects the all-32-bit one, the others (__int128, and {u64 lo64, u32 hi32})
// are alternatives.  The two-argument constructors use two conventions, told apart only by the type of "lo":
// make_i96(hi, u32 lo) = hi * 2^32 + lo, make_i96(hi, u64 lo) = hi * 2^64 + lo.  The alternatives lacked the u32 form,
// so FFT3161's make_i96(i64, u32) call (carryutil.cl) silently bound to the u64 form and was off by a factor of 2^32.
// The harness compiles each implementation (by rewriting the selector lines of the real math.cl) and checks the
// constructors and accessors against values computed on the host.
//
// usage: i96_make_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char* KERNEL = R"CL(
kernel void probe(global const long* hi, global const uint* lo32in, global const ulong* lo64in, global uint* out) {
  uint i = get_global_id(0);
  i96 a = make_i96((i64) hi[i], (u32) lo32in[i]);
  i96 b = make_i96((i64) hi[i], (u64) lo64in[i]);
  i96 c = make_i96((i64) hi[i]);
  i96 d = add(a, (i64) hi[i]);
  out[i * 12 + 0] = i96_lo32(a); out[i * 12 + 1] = i96_mid32(a); out[i * 12 + 2] = i96_hi32(a);
  out[i * 12 + 3] = i96_lo32(b); out[i * 12 + 4] = i96_mid32(b); out[i * 12 + 5] = i96_hi32(b);
  out[i * 12 + 6] = i96_lo32(c); out[i * 12 + 7] = i96_mid32(c); out[i * 12 + 8] = i96_hi32(c);
  out[i * 12 + 9] = i96_lo32(d); out[i * 12 + 10] = i96_mid32(d); out[i * 12 + 11] = i96_hi32(d);
}
)CL";

// Selects implementation `which` (0: all 32-bit, 1: __int128, 2: lo64/hi32) in the real math.cl text.
static bool select(std::string& math, int which) {
  const char* lines[] = {"#if 1\n// An all 32-bit implementation.", "#elif defined(__SIZEOF_INT128__)\n// An i128 implementation.",
                         "#elif 1\n// A u64 lo32, u32 hi32 implementation."};
  const char* chosen[] = {"#if I96_IMPL == 0\n// An all 32-bit implementation.",
                          "#elif I96_IMPL == 1\n// An i128 implementation.",
                          "#elif I96_IMPL == 2\n// A u64 lo32, u32 hi32 implementation."};
  for (int k = 0; k < 3; ++k) {
    size_t p = math.find(lines[k]);
    if (p == std::string::npos) { std::printf("FAIL: math.cl i96 selector %d not found\n", k); return false; }
    math.replace(p, std::string(lines[k]).size(), chosen[k]);
  }
  math = "#define I96_IMPL " + std::to_string(which) + "\n" + math;
  return true;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  std::ifstream f(dir + "/math.cl");
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string mathSrc = ss.str();

  const std::vector<long long> hi = {0, 1, -1, 0x123456789ll, -0x123456789ll, 0x7fffffffll, -0x80000000ll, 5};
  const std::vector<unsigned> lo32 = {0, 2, 0xffffffffu, 0xabcdef01u, 7, 0x80000000u, 1, 0};
  const std::vector<unsigned long long> lo64 = {0, 3, ~0ull, 0x0123456789abcdefull, 9, 1ull << 63, 1, 0x100000000ull};
  const size_t n = hi.size();

  int failures = 0;
  const char* names[] = {"all 32-bit", "__int128", "lo64/hi32"};
  for (int which = 0; which < 3; ++which) {
    std::string math = mathSrc;
    if (!select(math, which)) return 1;
    Program prog = loadSource(ctx, "#include \"base.cl\"\n" + math + KERNEL);
    std::string opts = "-I " + dir + " -DNO_ASM=1 -DFFT_TYPE=1 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=1 -DWordSize=8u "
                       "-DEXP=20000033u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=4u -DNH=4u -DCARRY_LEN=8u -DPFA_RADIX=0u "
                       "-DMAXBPW=3000u -DFFT_VARIANT=101u";
    if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
      std::string log = getBuildLog(prog.get(), dev);
      if (which == 1 && log.find("__int128") != std::string::npos) { std::printf("SKIP: %s (no __int128 on this device)\n", names[which]); continue; }
      std::printf("FAIL: %s does not build:\n%.600s\n", names[which], log.c_str());
      ++failures;
      continue;
    }
    int err = 0;
    cl_kernel k = clCreateKernel(prog.get(), "probe", &err);
    CHECK1(err);
    KernelHolder kHold{k};
    cl_mem bhi = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * 8, (void*) hi.data(), &err); CHECK1(err);
    cl_mem blo32 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * 4, (void*) lo32.data(), &err); CHECK1(err);
    cl_mem blo64 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * 8, (void*) lo64.data(), &err); CHECK1(err);
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 12 * 4, nullptr, &err); CHECK1(err);
    std::unique_ptr<cl_mem> h1{bhi}, h2{blo32}, h3{blo64}, h4{bout};
    CHECK1(clSetKernelArg(k, 0, sizeof bhi, &bhi));
    CHECK1(clSetKernelArg(k, 1, sizeof blo32, &blo32));
    CHECK1(clSetKernelArg(k, 2, sizeof blo64, &blo64));
    CHECK1(clSetKernelArg(k, 3, sizeof bout, &bout));
    size_t global = n;
    CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
    std::vector<unsigned> out(n * 12);
    CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    for (size_t i = 0; i < n; ++i) {
      using u128 = unsigned __int128;
      const u128 h = (u128) (__int128) hi[i];
      const u128 want[4] = {(h << 32) + lo32[i], (h << 64) + lo64[i], h, (h << 32) + lo32[i] + h};
      const char* what[4] = {"make_i96(i64, u32)", "make_i96(i64, u64)", "make_i96(i64)", "add(i96, i64)"};
      for (int c = 0; c < 4; ++c) {
        const unsigned w0 = unsigned(want[c]), w1 = unsigned(want[c] >> 32), w2 = unsigned(want[c] >> 64);
        const unsigned* g = &out[i * 12 + c * 3];
        if (g[0] != w0 || g[1] != w1 || g[2] != w2) {
          std::printf("FAIL: %s %s with hi=%lld: got %08x:%08x:%08x, want %08x:%08x:%08x\n", names[which], what[c], hi[i],
                      g[2], g[1], g[0], w2, w1, w0);
          ++failures;
        }
      }
    }
  }
  std::printf("%s\n", failures ? "FAIL" : "i96 implementations test ok");
  return failures ? 1 : 0;
}
