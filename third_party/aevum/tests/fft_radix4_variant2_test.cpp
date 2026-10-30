// Host test: the FP64 radix-4 variant 2 width FFT (fft_common's WG == 64 and WG == 256 RADIX == 4 VARIANT == 2 paths) must
// be a W-point DFT.
//
// Those paths pass preloaded cosines between threads through LDS in partial_tabMul4, at f = 4, 16 and (WG 256) 64.  The
// LDS slot of each cosine is permuted to avoid bank conflicts, so a store/load index mismatch gives some outputs the
// wrong twiddle.  The test compiles the real fftwidth.cl at widths 256 (WG 64) and 1024 (WG 256) with NW = 4, feeds it
// the host's trig table (genSmallTrigFP64, as the engine uploads it), and checks that the W outputs are, in some
// order, exactly the W DFT bins of the input.  Variant 1 at the same widths, which does not use partial_tabMul4, is
// the control.
//
// usage: fft_radix4_variant2_test [cl-source-dir]   (default src/cl)

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

vector<double2> genSmallTrigFP64(u32 size, u32 radix);

static const char* HARNESS = R"CL(
#include "base.cl"
#include "fftwidth.cl"

KERNEL(G_W) fftw(global double2* in, global double2* out, Trig trig) {
  local T2 lds[LDS_BYTES / sizeof(T2)];
  u32 me = get_local_id(0);
  T2 u[NW];
  for (u32 i = 0; i < NW; ++i) u[i] = U2(in[i * G_W + me].x, in[i * G_W + me].y);
  fft_WIDTH(lds, u, trig, 1, me);
  for (u32 i = 0; i < NW; ++i) out[i * G_W + me] = (double2)(u[i].x, u[i].y);
}
)CL";

enum class CaseResult { Pass, Fail, ResourceLimited };

static CaseResult runCase(cl_device_id dev, cl_context ctx, cl_queue q, const std::string& dir, size_t W, int variantW) {
  const size_t NW = 4, wg = W / NW;
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
  std::string opts = "-I " + dir + " -DNO_ASM=1 -DAMDGPU=0 -DWAVEFRONT=1 -DFFT_FP64=1 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=0 -DFFT_TYPE=0 "
    "-DEXP=17000023u -DWIDTH=" + std::to_string(W) + "u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=4u -DNH=4u "
    "-DWordSize=4u -DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u -DFFT_VARIANT=" + std::to_string(variantW * 100) + "u "
    "-DFUSE_WEIGHT_BUTTERFLY=0 -DTABMUL_CHAIN=0 -DTRIG_SCALE=" + std::to_string(coefs.scale) +
    " -DTRIG_SIN=" + list(coefs.sinCoefs) + " -DTRIG_COS=" + list(coefs.cosCoefs);
  if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::printf("FAIL: build\n%s\n", getBuildLog(prog.get(), dev).c_str());
    return CaseResult::Fail;
  }
  int err = 0;
  cl_kernel k = clCreateKernel(prog.get(), "fftw", &err);
  CHECK1(err);
  vector<double2> trig = genSmallTrigFP64(W, NW);
  const size_t N = W;
  cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 16, nullptr, &err);
  CHECK1(err);
  cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 16, nullptr, &err);
  CHECK1(err);
  cl_mem btrig = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, trig.size() * 16, trig.data(), &err);
  CHECK1(err);
  KernelHolder kHold{k};
  std::unique_ptr<cl_mem> binHold{bin}, boutHold{bout}, btrigHold{btrig};

  // A built kernel can expose a smaller legal work-group limit than the device maximum.
  constexpr cl_kernel_work_group_info kKernelWorkGroupSize = 0x11B0;
  size_t maxWg = 0;
  CHECK1(clGetKernelWorkGroupInfo(k, dev, kKernelWorkGroupSize, sizeof(maxWg), &maxWg, nullptr));
  if (maxWg != 0 && wg > maxWg) {
    std::printf("width %4zu variant %d: resource-limited requested WG %zu, kernel limit %zu\n", W, variantW, wg, maxWg);
    return CaseResult::ResourceLimited;
  }

  std::vector<double> in(2 * N), out(2 * N);
  srand(4243);
  for (double& x : in) { x = double(rand()) / RAND_MAX * 2 - 1; }
  CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, N * 16, in.data(), 0, nullptr, nullptr));
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
  std::printf("width %4zu radix 4 variant %d: %zu of %zu outputs match no DFT bin, %zu bins not hit exactly once, worst nearest-bin error %.3e  %s\n",
              W, variantW, unmatched, N, missing, worst, ok ? "ok" : "FAIL");
  return ok ? CaseResult::Pass : CaseResult::Fail;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];

  char ext[8192] = {};
  clGetDeviceInfo(dev, 0x1030 /* CL_DEVICE_EXTENSIONS */, sizeof(ext) - 1, ext, nullptr);
  if (std::string(ext).find("cl_khr_fp64") == std::string::npos) { std::printf("SKIP: device has no double support\n"); return 0; }

  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);
  bool ok = true;
  unsigned executed = 0, resourceLimited = 0;
  for (size_t width : {256, 1024}) {
    for (int variant : {1, 2}) {
      const CaseResult r = runCase(dev, ctx, q, dir, width, variant);
      if (r == CaseResult::Pass) ++executed;
      else if (r == CaseResult::ResourceLimited) ++resourceLimited;
      else ok = false;
    }
  }
  std::printf("radix-4 width cases: %u executed, %u resource-limited\n", executed, resourceLimited);
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
