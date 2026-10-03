// Host test: the FP64 fft8_skip1 (used with FUSE_WEIGHT_BUTTERFLY) must equal fft8 once the first butterfly has been done.
//
// fft8_skip1 is fft8 for a caller that has already done the first butterfly's adds/subs, so the post-rotations of
// u[5], u[6] and u[7] are done inside it. u[7] has to be rotated by mul_3t8_delayed, as fft8Core does, because
// fft4CoreSpecial does not absorb the factor of i. Compiling and running the real src/cl/fft8.cl on any OpenCL
// device with double support compares fft8 and fft8_skip1 against a reference DFT.
//
// usage: fft8_skip1_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static const char* HARNESS = R"CL(
#include "base.cl"
#include "math.cl"
#include "fft4.cl"
#include "fft8.cl"

KERNEL(1) t8(global double2* in, global double2* out, int mode) {
  T2 u[8];
  for (int k = 0; k < 8; ++k) u[k] = U2(in[k].x, in[k].y);
  if (mode == 0) {
    fft8(u);
  } else {
    // What carryFused does before calling fft8_skip1: the first butterfly's adds/subs.
    T2 v[8];
    for (int k = 0; k < 4; ++k) { v[k] = u[k] + u[k + 4]; v[k + 4] = u[k] - u[k + 4]; }
    for (int k = 0; k < 8; ++k) u[k] = v[k];
    fft8_skip1(u);
  }
  for (int k = 0; k < 8; ++k) out[k] = (double2)(u[k].x, u[k].y);
}
)CL";

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
  Program prog = loadSource(ctx, HARNESS);
  std::string opts = "-I " + dir + " -DNO_ASM=1 -DFFT_FP64=1 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=0 -DFFT_TYPE=0 "
    "-DEXP=11999989u -DWIDTH=512u -DSMALL_HEIGHT=256u -DMIDDLE=4u -DNW=8u -DNH=4u -DWordSize=4u -DCARRY_LEN=8u "
    "-DPFA_RADIX=0u -DMAXBPW=3998u -DFFT_VARIANT=202u";
  if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::printf("FAIL: build\n%s\n", getBuildLog(prog.get(), dev).c_str());
    return 1;
  }
  int err = 0;
  cl_kernel k = clCreateKernel(prog.get(), "t8", &err);
  CHECK1(err);
  cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 8 * 16, nullptr, &err);
  CHECK1(err);
  cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 8 * 16, nullptr, &err);
  CHECK1(err);

  double maxFft8 = 0, maxSkip1 = 0;
  srand(12345);
  for (int trial = 0; trial < 50; ++trial) {
    double in[16], out[2][16];
    for (double& x : in) { x = double(rand()) / RAND_MAX * 2 - 1; }
    CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, sizeof in, in, 0, nullptr, nullptr));
    for (int mode = 0; mode < 2; ++mode) {
      CHECK1(clSetKernelArg(k, 0, sizeof bin, &bin));
      CHECK1(clSetKernelArg(k, 1, sizeof bout, &bout));
      CHECK1(clSetKernelArg(k, 2, sizeof mode, &mode));
      size_t one = 1;
      CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &one, &one, 0, nullptr, nullptr));
      CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, sizeof out[mode], out[mode], 0, nullptr, nullptr));
    }
    // Reference DFT (forward transform, exp(+i...) as fft8 defines it).
    for (int j = 0; j < 8; ++j) {
      double rr = 0, ri = 0;
      for (int m = 0; m < 8; ++m) {
        double a = 2 * M_PI * j * m / 8, c = std::cos(a), s = std::sin(a);
        rr += in[2*m] * c - in[2*m+1] * s;
        ri += in[2*m] * s + in[2*m+1] * c;
      }
      for (int mode = 0; mode < 2; ++mode) {
        double d = std::fmax(std::fabs(rr - out[mode][2*j]), std::fabs(ri - out[mode][2*j+1]));
        (mode ? maxSkip1 : maxFft8) = std::fmax(mode ? maxSkip1 : maxFft8, d);
      }
    }
  }
  std::printf("fft8       max error vs DFT: %.3e\n", maxFft8);
  std::printf("fft8_skip1 max error vs DFT: %.3e\n", maxSkip1);
  bool ok = maxFft8 < 1e-12 && maxSkip1 < 1e-12;
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
