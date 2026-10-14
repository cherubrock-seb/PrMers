// Host test: fftMiddleOutGF61 and fftMiddleInGF61 with AEVUM_GF61_MIDDLE_BASE_TABLE = 1, 2 or 3 (the direct base twiddle table; 1 = both kernels,
// 2 = middle-out only, 3 = middle-in only) must equal the stock kernels (mode 0).
//
// TrigBufCache's genMiddleTrigGF61 stores the table row-major as [row < SMALL_HEIGHT][column < WIDTH] holding w^(row * column).  fftMiddleIn
// indexes it as y * WIDTH + x with x < WIDTH, y < SMALL_HEIGHT.  fftMiddleOut's coordinates are the other way round (x < SMALL_HEIGHT,
// y < WIDTH), and it used to index y * WIDTH + x as well: right only when WIDTH == SMALL_HEIGHT, silently wrong twiddles when WIDTH <
// SMALL_HEIGHT and an out-of-bounds table read when WIDTH > SMALL_HEIGHT.
//
// The harness compiles the real fftmiddleout.cl and fftmiddlein.cl (both the separate-buffer and the in-place kernels), runs them on random data with the table
// built by the real genMiddleTrigGF61, and compares each table mode with mode 0 word for word.  Shapes with WIDTH != SMALL_HEIGHT are the point;
// WIDTH == SMALL_HEIGHT is the control.  The table sits at the very end of the trig buffer, so a read past it is caught by a canary entry
// that is not a valid twiddle.
//
// The kernels are compiled with reqd_work_group_size (IN_WG/OUT_WG = 128, 256 for the in-place layout), so the workgroup size cannot be shrunk to
// fit a device.  A device that cannot run that size (CL_DEVICE_MAX_WORK_GROUP_SIZE or the kernel's CL_KERNEL_WORK_GROUP_SIZE below it, or
// clEnqueueNDRangeKernel failing with CL_INVALID_WORK_GROUP_SIZE, as Apple's CPU device does) skips those cases with a "skip:" line; a wrong
// result still fails.  No OpenCL platform or device skips the whole test.
//
// usage: gf61_middle_base_table_kernel_test [cl-source-dir]   (default src/cl); AEVUM_TEST_DEVICE is not used, device 0 runs.

#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

vector<ulong2> genMiddleTrigGF61(u32 smallH, u32 middle, u32 width, bool direct_base_table);

static const char* HARNESS_OUT = "#include \"fftmiddleout.cl\"\n";
static const char* HARNESS_IN = "#include \"fftmiddlein.cl\"\n";

typedef unsigned long long u64_t;
static const u64_t M61 = (1ull << 61) - 1;
static_assert(sizeof(ulong2) == 16, "trig entries are uploaded as raw 16-byte pairs");

struct Shape { u32 width, smallH, middle; };

// Elements to allocate for the transform data.  The in-place layout pads its rows and slabs (SIZEBLK/SIZEW/SIZEM in middle.cl): the nVidia
// layout (INPLACE=1) adds 32 elements per 16-row slab and 16 per middle row, i.e. middle * (2 * width + 16) on top of the width * middle * smallH
// of the separate-buffer layout.  Allocate twice that, so a wider pad in the kernel cannot overflow the buffer.  The same random data fills the
// whole buffer in every run, so the layout itself does not matter to the comparison.
static size_t dataElems(const Shape& s) { return (size_t) s.width * s.middle * s.smallH + (size_t) s.middle * (4 * (size_t) s.width + 64) + 1024; }

#ifndef CL_DEVICE_MAX_WORK_GROUP_SIZE
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#endif
#ifndef CL_KERNEL_WORK_GROUP_SIZE
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0
#endif

enum RunResult { RUN_OK, RUN_FAIL, RUN_SKIP };

// Runs fftMiddleOutGF61 on `input` with the given table mode and INPLACE setting; RUN_FAIL if the kernel failed to build, RUN_SKIP if the device
// cannot launch the kernel's required workgroup size.
static RunResult run(cl_device_id dev, cl_context ctx, cl_queue q, const std::string& dir, const Shape& s, bool middleIn, int mode, int inplace,
                const std::vector<u64_t>& input, const std::vector<ulong2>& trig, std::vector<u64_t>& out) {
  Program prog = loadSource(ctx, middleIn ? HARNESS_IN : HARNESS_OUT);
  std::string opts = "-cl-std=CL1.2 -I " + dir + " -DNO_ASM=1 -DAMDGPU=0 -DFFT_TYPE=3 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=1 -DWordSize=8u "
    "-DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u -DEXP=1257787u -DNW=4u -DNH=4u -DFFT_VARIANT=101u "
    "-DFRAC_BPW_HI=1u -DFRAC_BPW_LO=4294967295u -DTAILTGF61=U2(1ul,1ul) -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul "
    "-DWIDTH=" + std::to_string(s.width) + "u -DSMALL_HEIGHT=" + std::to_string(s.smallH) + "u -DMIDDLE=" + std::to_string(s.middle) + "u "
    "-DINPLACE=" + std::to_string(inplace) + " -DAEVUM_GF61_MIDDLE_BASE_TABLE=" + std::to_string(mode);
  if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::string log = getBuildLog(prog.get(), dev);
    size_t e = log.find("error:");
    std::printf("  build FAIL: %s\n", (e == std::string::npos ? log.substr(0, 300) : log.substr(e, 300)).c_str());
    return RUN_FAIL;
  }
  int err = 0;
  cl_kernel k = clCreateKernel(prog.get(), middleIn ? "fftMiddleInGF61" : "fftMiddleOutGF61", &err);
  CHECK1(err);
  KernelHolder kHold{k};
  const size_t n = dataElems(s);
  cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 16, nullptr, &err);
  CHECK1(err);
  cl_mem bout = inplace ? bin : clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 16, nullptr, &err);
  CHECK1(err);
  cl_mem btrig = clCreateBuffer(ctx, CL_MEM_READ_WRITE, trig.size() * 16, nullptr, &err);
  CHECK1(err);
  std::unique_ptr<cl_mem> inHold{bin}, outHold{inplace ? nullptr : bout}, trigHold{btrig};
  CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, n * 16, input.data(), 0, nullptr, nullptr));
  if (!inplace) {  // The separate output buffer has padding the kernel never writes: give it defined contents so the runs compare.
    std::vector<u64_t> zeros(2 * n, 0);
    CHECK1(clEnqueueWriteBuffer(q, bout, 1, 0, n * 16, zeros.data(), 0, nullptr, nullptr));
  }
  CHECK1(clEnqueueWriteBuffer(q, btrig, 1, 0, trig.size() * 16, trig.data(), 0, nullptr, nullptr));
  CHECK1(clSetKernelArg(k, 0, sizeof bout, &bout));
  CHECK1(clSetKernelArg(k, 1, sizeof bin, &bin));
  CHECK1(clSetKernelArg(k, 2, sizeof btrig, &btrig));
  size_t local, groups;
  if (inplace) { local = 256; groups = (size_t) (s.smallH / 16) * (s.width / 16); }
  else if (middleIn) { local = 128; groups = (size_t) (s.width / 16) * (s.smallH / (128 / 16)); }
  else               { local = 128; groups = (size_t) (s.smallH / 16) * (s.width / (128 / 16)); }
  size_t g = groups * local;
  const char* kind = middleIn ? "middle-in " : "middle-out";
  size_t devMax = 0, kernMax = 0;
  if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof devMax, &devMax, nullptr) != CL_SUCCESS) devMax = 0;
  if (clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof kernMax, &kernMax, nullptr) != CL_SUCCESS) kernMax = 0;
  // A failed query (0) is not a reason to skip: the enqueue below decides.
  if ((devMax && devMax < local) || (kernMax && kernMax < local)) {
    std::printf("skip: W=%u H=%u M=%u %s INPLACE=%d BASE_TABLE=%d: needs workgroup size %zu, device max %zu, kernel max %zu\n",
                s.width, s.smallH, s.middle, kind, inplace, mode, local, devMax, kernMax);
    return RUN_SKIP;
  }
  err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &local, 0, nullptr, nullptr);
  if (err == CL_INVALID_WORK_GROUP_SIZE) {
    std::printf("skip: W=%u H=%u M=%u %s INPLACE=%d BASE_TABLE=%d: device rejected workgroup size %zu (INVALID_WORK_GROUP_SIZE)\n",
                s.width, s.smallH, s.middle, kind, inplace, mode, local);
    return RUN_SKIP;
  }
  CHECK1(err);
  out.assign(2 * n, 0);
  CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, n * 16, out.data(), 0, nullptr, nullptr));
  for (auto& v : out) v %= M61;     // M61 and 0 are the same residue
  return RUN_OK;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices;
  try { devices = getAllDeviceIDs(); }
  catch (const std::exception& e) { std::printf("skip: no usable OpenCL platform (%s)\n", e.what()); return 0; }
  if (devices.empty()) { std::printf("skip: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  const Shape shapes[] = {{256, 256, 2}, {256, 512, 2}, {512, 256, 2}, {256, 1024, 2}, {1024, 256, 2}, {512, 512, 2}, {512, 256, 4}};
  bool ok = true;
  size_t compared = 0, skipped = 0;
  srand(5);
  for (const Shape& s : shapes) {
    const size_t n = dataElems(s);
    std::vector<u64_t> input(2 * n);
    for (auto& v : input) v = ((u64_t) rand() << 40 ^ (u64_t) rand() << 20 ^ (u64_t) rand()) % M61;
    std::vector<ulong2> stock = genMiddleTrigGF61(s.smallH, s.middle, s.width, false);
    std::vector<ulong2> table = genMiddleTrigGF61(s.smallH, s.middle, s.width, true);
    if (table.size() != stock.size() + (size_t) s.width * s.smallH) {
      std::printf("W=%u H=%u M=%u: direct table has %zu entries, expected %zu: FAIL\n", s.width, s.smallH, s.middle, table.size(),
                  stock.size() + (size_t) s.width * s.smallH);
      ok = false;
      continue;
    }
    // Canary past the table: a table read beyond its end returns a value that makes the output differ.
    ulong2 canary{0x1234567890abcdefull % M61, 0x0fedcba098765432ull % M61};
    std::vector<ulong2> tableCanary = table;
    for (int i = 0; i < 4096; ++i) tableCanary.push_back(canary);
    for (int middleIn = 0; middleIn < 2; ++middleIn) {
      for (int inplace = 0; inplace < 3; ++inplace) {
        std::vector<u64_t> ref;
        RunResult rr = run(dev, ctx, q, dir, s, middleIn, 0, inplace, input, stock, ref);
        if (rr == RUN_SKIP) { ++skipped; continue; }   // the same workgroup size serves every mode of this case
        if (rr != RUN_OK) { ok = false; continue; }
        for (int mode = 1; mode <= 3; ++mode) {
          std::vector<u64_t> got;
          rr = run(dev, ctx, q, dir, s, middleIn, mode, inplace, input, tableCanary, got);
          if (rr == RUN_SKIP) { ++skipped; continue; }
          if (rr != RUN_OK) { ok = false; continue; }
          ++compared;
          size_t bad = 0;
          for (size_t i = 0; i < got.size(); ++i) bad += got[i] != ref[i];
          std::printf("W=%u H=%u M=%u %s INPLACE=%d BASE_TABLE=%d: %zu of %zu words differ from mode 0  %s\n",
                      s.width, s.smallH, s.middle, middleIn ? "middle-in " : "middle-out", inplace, mode, bad, got.size(), bad ? "FAIL" : "ok");
          ok &= bad == 0;
        }
      }
    }
  }
  if (ok && compared == 0 && skipped > 0) {
    std::printf("skip: %zu cases skipped, none could run on this device (workgroup size)\n", skipped);
    return 0;
  }
  if (skipped) std::printf("note: %zu cases skipped (workgroup size), %zu compared\n", skipped, compared);
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
