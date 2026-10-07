// Host test: the FP64 width FFT must be a W-point DFT for every width variant, in particular variant 0 (broadcast) at a 1K width.
//
// A 1K width runs fft_common with WG == 128 and RADIX == 8, and 1024 = 8 * 8 * 16 is not a power of RADIX.  The variant 1 and 2
// code has a dedicated 8 * 8 * 16 path for it; the variant 0 generic loop `for (s = 1; s < WG; s *= RADIX)` used to run four
// radix-8 passes (the shape of a 4096-point FFT) and so computed garbage.
//
// Variant 0 needs AMD's broadcast builtins, which no CPU OpenCL device has.  For variant 0 the harness defines the three builtins
// as small emulations over a program-scope buffer (one work-group of at most 128 work-items, wave64 semantics) and compiles the
// real fftwidth.cl as an AMD GPU.  Each case checks that the W outputs are, in some order, exactly the W DFT bins of the input.
// The 1K variant 1 case and the 512 variant 0 case are controls that were always correct.  Variant 0 cases are skipped on a
// device whose OpenCL C compiler has no program-scope variables.
//
// usage: fft_variant0_1k_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"
#include "Trig.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* HARNESS = R"CL(
#if AMDGPU
// Emulate the amdgcn builtins used by fftbase.cl's bcast4/8/16/64 (wave64, one work-group).
global int emu_exchange[128];
int emu_bcast(int x, uint srcMask) {
  uint lid = get_local_id(0);
  barrier(CLK_GLOBAL_MEM_FENCE);
  emu_exchange[lid] = x;
  barrier(CLK_GLOBAL_MEM_FENCE);
  int r = emu_exchange[lid & srcMask];
  barrier(CLK_GLOBAL_MEM_FENCE);
  return r;
}
#define __builtin_amdgcn_mov_dpp(x, ctl, rm, bm, bc) emu_bcast((x), ~3u)       // quad_perm lane 0 of each quad
#define __builtin_amdgcn_ds_swizzle(x, pattern) emu_bcast((x), (pattern) == 0x0018 ? ~7u : ~15u)
#define __builtin_amdgcn_readfirstlane(x) emu_bcast((x), ~63u)                   // first lane of the wave
// base.cl falls back to variant 1 unless the compiler reports the amdgcn builtins; the emulations above stand in for them.
#define __has_builtin(x) 1
#endif

#include "base.cl"
#include "fftwidth.cl"

// The small trig table variant 1 reads (TABMUL_CHAIN): the same broadcast value variant 0 computes for itself.
KERNEL(G_W) mktrig(global double2* trig) {
  u32 me = get_local_id(0);
  T2 w = fancyTrig_N(ND / (G_W * 8) * me);
  trig[me] = (double2)(w.x, w.y);
}

KERNEL(G_W) fftw(global double2* in, global double2* out, Trig trig) {
  local T2 lds[LDS_BYTES / sizeof(T2)];
  u32 me = get_local_id(0);
  T2 u[8];
  for (u32 i = 0; i < 8; ++i) u[i] = U2(in[i * G_W + me].x, in[i * G_W + me].y);
  fft_WIDTH(lds, u, trig, 1, me);
  for (u32 i = 0; i < 8; ++i) out[i * G_W + me] = (double2)(u[i].x, u[i].y);
}
)CL";

static bool hasProgramScopeVariables(cl_device_id dev, cl_context ctx) {
  Program probe = loadSource(ctx, "global int probe[4];\nkernel void k(global int* o) { probe[get_local_id(0) & 3] = 1; o[0] = probe[0]; }\n");
  return clBuildProgram(probe.get(), 1, &dev, "-cl-std=CL3.0", nullptr, nullptr) == 0;
}

static bool runCase(cl_device_id dev, cl_context ctx, cl_queue q, const std::string& dir, size_t W, int variantW) {
  Program prog = loadSource(ctx, HARNESS);
  // The reducedCosSin coefficients the host passes for every FP64 FFT: trigCoefs(FFTShape::size() / 4), where size() is
  // width * height * middle * 2 and height * middle is 512 here.
  TrigCoefs coefs = trigCoefs(W * 256);
  auto list = [](const std::array<double, 8>& v) {
    std::string r = "{";
    char buf[64];
    for (double d : v) { std::snprintf(buf, sizeof buf, "%.17g,", d); r += buf; }
    return r + "}";
  };
  std::string opts = std::string(variantW == 0 ? "-cl-std=CL3.0 " : "") + "-I " + dir + " -DNO_ASM=1 -DAMDGPU=" + (variantW == 0 ? "1" : "0") +
    " -DWAVEFRONT=1 -DFFT_FP64=1 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=0 -DFFT_TYPE=0 "
    "-DEXP=17000023u -DWIDTH=" + std::to_string(W) + "u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=8u -DNH=4u "
    "-DWordSize=4u -DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u -DFFT_VARIANT=" + std::to_string(variantW * 100) + "u "
    "-DFUSE_WEIGHT_BUTTERFLY=0 -DTABMUL_CHAIN=1 -DTRIG_SCALE=" + std::to_string(coefs.scale) +
    " -DTRIG_SIN=" + list(coefs.sinCoefs) + " -DTRIG_COS=" + list(coefs.cosCoefs);
  if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::printf("FAIL: build\n%s\n", getBuildLog(prog.get(), dev).c_str());
    return false;
  }
  int err = 0;
  cl_kernel k = clCreateKernel(prog.get(), "fftw", &err);
  CHECK1(err);
  cl_kernel mk = clCreateKernel(prog.get(), "mktrig", &err);
  CHECK1(err);
  const size_t N = W;
  cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 16, nullptr, &err);
  CHECK1(err);
  cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 16, nullptr, &err);
  CHECK1(err);
  cl_mem btrig = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 16, nullptr, &err);
  CHECK1(err);

  std::vector<double> in(2 * N), out(2 * N);
  srand(4242);
  for (double& x : in) { x = double(rand()) / RAND_MAX * 2 - 1; }
  CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, N * 16, in.data(), 0, nullptr, nullptr));
  size_t wg = W / 8;
  CHECK1(clSetKernelArg(mk, 0, sizeof btrig, &btrig));
  CHECK1(clEnqueueNDRangeKernel(q, mk, 1, nullptr, &wg, &wg, 0, nullptr, nullptr));
  CHECK1(clSetKernelArg(k, 0, sizeof bin, &bin));
  CHECK1(clSetKernelArg(k, 1, sizeof bout, &bout));
  CHECK1(clSetKernelArg(k, 2, sizeof btrig, &btrig));
  CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &wg, &wg, 0, nullptr, nullptr));
  CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, N * 16, out.data(), 0, nullptr, nullptr));

  // Reference DFT of the input (element i * G_W + me of the input is u[i] of work-item me).  The transform leaves the bins in
  // a fixed permuted order, so match each output to the reference bin it equals.
  using cplx = std::complex<double>;
  std::vector<cplx> x(N), ref(N);
  for (size_t j = 0; j < N; ++j) x[j] = cplx(in[2 * j], in[2 * j + 1]);
  for (size_t f = 0; f < N; ++f) {
    cplx s = 0;
    for (size_t j = 0; j < N; ++j) { double a = 2 * M_PI * double((f * j) % N) / double(N); s += x[j] * cplx(std::cos(a), std::sin(a)); }
    ref[f] = s;
  }
  std::vector<int> hits(N, 0);
  size_t unmatched = 0;
  double worst = 0;
  for (size_t j = 0; j < N; ++j) {
    cplx o(out[2 * j], out[2 * j + 1]);
    size_t best = 0;
    double bestErr = 1e300;
    for (size_t f = 0; f < N; ++f) {
      double e = std::abs(o - ref[f]);
      if (e < bestErr) bestErr = e, best = f;
    }
    if (bestErr > 1e-9) ++unmatched; else ++hits[best];
    worst = std::fmax(worst, bestErr);
  }
  size_t missing = 0;
  for (size_t f = 0; f < N; ++f) if (hits[f] != 1) ++missing;
  bool ok = unmatched == 0 && missing == 0;
  std::printf("width %4zu variant %d: %zu of %zu outputs match no DFT bin, %zu bins not hit exactly once, worst nearest-bin error %.3e  %s\n",
              W, variantW, unmatched, N, missing, worst, ok ? "ok" : "FAIL");
  return ok;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];

  char ext[8192] = {};
  clGetDeviceInfo(dev, 0x1030 /* CL_DEVICE_EXTENSIONS */, sizeof(ext) - 1, ext, nullptr);
  if (std::string(ext).find("cl_khr_fp64") == std::string::npos) { std::printf("SKIP: device has no double support\n"); return 0; }
  size_t maxWg = 0;
  clGetDeviceInfo(dev, 0x1004 /* CL_DEVICE_MAX_WORK_GROUP_SIZE */, sizeof maxWg, &maxWg, nullptr);
  if (maxWg < 128) { std::printf("SKIP: device work-group size limit %zu < 128\n", maxWg); return 0; }

  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);
  bool ok = runCase(dev, ctx, q, dir, 1024, 1);
  if (hasProgramScopeVariables(dev, ctx)) {
    ok &= runCase(dev, ctx, q, dir, 512, 0);
    ok &= runCase(dev, ctx, q, dir, 1024, 0);
  } else {
    std::printf("SKIP: variant 0 cases need program-scope variables (OpenCL C 2.0/3.0)\n");
  }
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
