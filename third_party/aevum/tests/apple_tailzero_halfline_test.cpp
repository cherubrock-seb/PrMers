// Host test: the Apple staged tailSquareZero path must write line H/2 where the stock tail kernels do.
//
// The host passes the element offset of line H/2 in the tail output plane to tailSquareZeroGF61WriteDirectApple.  The stock kernels store line k
// at memory line transPos(k, MIDDLE, WIDTH) (fftheight.cl).  The harness compiles the real fftheight.cl on any OpenCL device, asks it for
// transPos(H/2, MIDDLE, WIDTH) for several plan shapes, and compares with appleTailZeroHalfLineOffset().  MIDDLE == 1 used to put the line at
// slot 0 (on top of line 0) instead of memory line WIDTH / 2.
//
// usage: apple_tailzero_halfline_test [cl-source-dir]   (default src/cl)

#include "AppleTailZero.h"
#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <string>

static const char* HARNESS = R"CL(
#include "base.cl"
#include "fftheight.cl"
KERNEL(1) halfLine(global uint* out) { out[0] = transPos(WIDTH * MIDDLE / 2, MIDDLE, WIDTH); }
)CL";

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  struct Shape { u32 width, middle, height; };
  const Shape shapes[] = {{256, 1, 256}, {512, 1, 512}, {1024, 1, 256}, {256, 2, 256}, {512, 4, 512}, {256, 8, 256}, {1024, 16, 1024}};
  int bad = 0;
  for (const Shape& s : shapes) {
    Program prog = loadSource(ctx, HARNESS);
    std::string opts = "-I " + dir + " -DNO_ASM=1 -DAMDGPU=0 -DFFT_TYPE=3 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=1 -DWordSize=8u -DCARRY_LEN=8u "
      "-DPFA_RADIX=0u -DMAXBPW=3998u -DEXP=1257787u -DFFT_VARIANT=101u -DINPLACE=0 -DNW=4u -DNH=4u "
      "-DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul -DFRAC_BPW_HI=1u -DFRAC_BPW_LO=4294967295u -DTAILTGF61=U2(1ul,1ul) "
      "-DWIDTH=" + std::to_string(s.width) + "u -DSMALL_HEIGHT=" + std::to_string(s.height) + "u -DMIDDLE=" + std::to_string(s.middle) + "u";
    if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
      std::printf("FAIL: build\n%s\n", getBuildLog(prog.get(), dev).c_str());
      return 1;
    }
    int err = 0;
    cl_kernel k = clCreateKernel(prog.get(), "halfLine", &err);
    CHECK1(err);
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, nullptr, &err);
    CHECK1(err);
    // Release the kernel and buffer when the case ends (a kernel keeps its program alive).
    KernelHolder kHold{k};
    std::unique_ptr<cl_mem> bHold{b};
    CHECK1(clSetKernelArg(k, 0, sizeof b, &b));
    size_t one = 1;
    CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &one, &one, 0, nullptr, nullptr));
    unsigned memLine = 0;
    CHECK1(clEnqueueReadBuffer(q, b, 1, 0, 4, &memLine, 0, nullptr, nullptr));

    const u32 expected = memLine * s.height;
    const u32 got = appleTailZeroHalfLineOffset(s.width, s.middle, s.height);
    const u32 previous = (s.middle / 2u) * s.height;
    const bool ok = got == expected;
    std::printf("WIDTH %4u MIDDLE %2u SMALL_HEIGHT %4u: line H/2 is at element offset %u, helper gives %u (previous host formula gave %u)  %s\n",
                s.width, s.middle, s.height, expected, got, previous, ok ? "ok" : "FAIL");
    if (!ok) ++bad;
  }
  std::printf("%s\n", bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
