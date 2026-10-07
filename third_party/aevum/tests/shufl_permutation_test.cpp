// Host test: every special case of fftbase.cl's shufl (LDSPAD / LDSSWIZ / SHUFL_BYTES variants) must produce the same permutation as the
// generic code, stay inside its LDS allocation, and the fused shufl_and_fft2 used by the 1K radix-8 transform must be shuffle + fft2.
//
// The generic shuffle writes element i of work-item me to slot  i * f + (me & ~(f-1)) * RADIX + (me & (f-1))  and reads slot  i * WG + me.
// The harness compiles the real fftwidth.cl for each (FFT type, width, SHUFL_BYTES_W, LDSPAD_W / LDSSWIZ_W), runs one work-group, fills
// the memory just past the LDS_BYTES the kernels allocate with a sentinel, and checks both the permutation and the sentinel.
//
// usage: shufl_permutation_test [cl-source-dir] [-v]   (default src/cl)   -v lists every case, not just failures

#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* HARNESS = R"CL(
#include "base.cl"
#include "fftwidth.cl"

#if FFT_FP64
typedef T2 V;
#define MKV(c) U2((double)(c), (double)(-(c)))
#define VSIZE 2
#elif FFT_FP32
typedef F2 V;
#define MKV(c) U2((float)(c), (float)(-(c)))
#define VSIZE 2
#elif NTT_GF31
typedef GF31 V;
#define MKV(c) U2((uint)(c), (uint)((c) + 7u))
#define VSIZE 2
#else
typedef GF61 V;
#define MKV(c) U2((ulong)(c), (ulong)((c) + 7u))
#define VSIZE 2
#endif

#define GUARD 1024

// mode 0: shufl(f = 1), mode 1: shufl(f = RADIX), mode 2: shufl(f = RADIX * RADIX), mode 3: shufl_and_fft2 (1K radix 8 only)
KERNEL(G_W) tshufl(global V* out, global V* in, global int* bad, uint mode) {
  local V lds[LDS_BYTES / sizeof(V) + GUARD];
  u32 me = get_local_id(0);
  const u32 n = LDS_BYTES / sizeof(V);
  for (u32 k = me; k < GUARD; k += G_W) lds[n + k] = MKV(777u + k);
  V u[RADIX];
  for (u32 i = 0; i < RADIX; ++i) u[i] = in[i * G_W + me];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (mode == 0)      shufl(lds, u, 1, 1, me);
  else if (mode == 1) shufl(lds, u, RADIX, 1, me);
  else if (mode == 2) shufl(lds, u, RADIX * RADIX, 1, me);
#if RADIX == 8 && WG == 128
  else                shufl_and_fft2(lds, u, 8, 1, me);
#endif
  bar(WG);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (u32 k = me; k < GUARD; k += G_W) {
    V g = lds[n + k], e = MKV(777u + k);
    if (g.x != e.x || g.y != e.y) atomic_or(bad, 1);
  }
  for (u32 i = 0; i < RADIX; ++i) out[i * G_W + me] = u[i];
}
)CL";

enum Type { FP64, FP32, GF31, GF61 };
static const char* typeName[] = {"FP64", "FP32", "GF31", "GF61"};

struct Case { Type type; unsigned width, nw, bytes; int pad, swiz; };

// One compiled program per case; returns number of failing sub-checks (or -1 for a build failure).
static int runCase(cl_device_id dev, cl_context ctx, cl_queue q, const std::string& dir, const Case& c, bool verbose, std::string& detail) {
  Program prog = loadSource(ctx, HARNESS);
  const unsigned WG = c.width / c.nw, R = c.nw;
  std::string opts = "-I " + dir + " -DNO_ASM=1 -DAMDGPU=0 -DWAVEFRONT=1 -DFFT_TYPE=0 ";
  opts += "-DFFT_FP64=" + std::to_string(c.type == FP64) + " -DFFT_FP32=" + std::to_string(c.type == FP32) + " -DNTT_GF31=" + std::to_string(c.type == GF31) +
          " -DNTT_GF61=" + std::to_string(c.type == GF61);
  opts += " -DEXP=17000023u -DWIDTH=" + std::to_string(c.width) + "u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=" + std::to_string(c.nw) + "u -DNH=4u "
          "-DWordSize=4u -DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u -DFFT_VARIANT=111u -DFUSE_WEIGHT_BUTTERFLY=0 "
          "-DTRIG_SCALE=1 -DTRIG_SIN={0,0,0,0,0,0,0,0} -DTRIG_COS={0,0,0,0,0,0,0,0} ";
  opts += "-DSHUFL_BYTES_W=" + std::to_string(c.bytes) + " -DLDSPAD_W=" + std::to_string(c.pad) + " -DLDSSWIZ_W=" + std::to_string(c.swiz);
  if (c.type == GF31) opts += " -DTAILTGF31=U2(1u,1u) -DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0";
  if (c.type == GF61) opts += " -DTAILTGF61=U2(1ul,1ul) -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul";
  if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::string log = getBuildLog(prog.get(), dev);
    size_t e = log.find("error:");
    detail = "build failed: " + (e == std::string::npos ? log.substr(0, 200) : log.substr(e, 160));
    for (char& ch : detail) if (ch == '\n') ch = ' ';
    return -1;
  }
  int err = 0;
  cl_kernel k = clCreateKernel(prog.get(), "tshufl", &err);
  CHECK1(err);
  const size_t N = c.width;
  const size_t esz = (c.type == FP64 || c.type == GF61) ? 16 : 8;
  cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * esz, nullptr, &err);
  CHECK1(err);
  cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * esz, nullptr, &err);
  CHECK1(err);
  cl_mem bbad = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, nullptr, &err);
  CHECK1(err);

  // Element (i, me) carries the code i * 1000 + me + 1 (component 0) and its negation / +7 (component 1).
  auto code = [&](unsigned i, unsigned me) { return i * 1000u + me + 1u; };
  auto put = [&](std::vector<unsigned char>& buf, size_t idx, unsigned long long x, long long y) {
    switch (c.type) {
      case FP64: { double d[2] = {double(x), double(y)}; memcpy(&buf[idx * 16], d, 16); break; }
      case FP32: { float d[2] = {float(x), float(y)}; memcpy(&buf[idx * 8], d, 8); break; }
      case GF31: { unsigned d[2] = {unsigned(x), unsigned(y)}; memcpy(&buf[idx * 8], d, 8); break; }
      case GF61: { unsigned long long d[2] = {x, (unsigned long long) y}; memcpy(&buf[idx * 16], d, 16); break; }
    }
  };
  auto get = [&](const std::vector<unsigned char>& buf, size_t idx, double& x, double& y) {
    switch (c.type) {
      case FP64: { double d[2]; memcpy(d, &buf[idx * 16], 16); x = d[0]; y = d[1]; break; }
      case FP32: { float d[2]; memcpy(d, &buf[idx * 8], 8); x = d[0]; y = d[1]; break; }
      case GF31: { unsigned d[2]; memcpy(d, &buf[idx * 8], 8); x = d[0]; y = d[1]; break; }
      case GF61: { unsigned long long d[2]; memcpy(d, &buf[idx * 16], 16); x = double(d[0]); y = double(d[1]); break; }
    }
  };
  const bool signedY = c.type == FP64 || c.type == FP32;
  auto yOf = [&](unsigned v) -> long long { return signedY ? -(long long) v : (long long) v + 7; };

  std::vector<unsigned char> in(N * esz), out(N * esz);
  for (unsigned i = 0; i < R; ++i) for (unsigned me = 0; me < WG; ++me) put(in, i * WG + me, code(i, me), yOf(code(i, me)));
  CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, N * esz, in.data(), 0, nullptr, nullptr));

  int fails = 0;
  std::string msg;
  const bool fused = R == 8 && WG == 128;
  for (unsigned mode = 0; mode < (fused ? 4u : 3u); ++mode) {
    if (mode == 2 && R * R > WG) continue;
    unsigned f = mode == 0 ? 1 : (mode == 1 || mode == 3) ? R : R * R;
    int zero = 0;
    CHECK1(clEnqueueWriteBuffer(q, bbad, 1, 0, 4, &zero, 0, nullptr, nullptr));
    CHECK1(clSetKernelArg(k, 0, sizeof bout, &bout));
    CHECK1(clSetKernelArg(k, 1, sizeof bin, &bin));
    CHECK1(clSetKernelArg(k, 2, sizeof bbad, &bbad));
    CHECK1(clSetKernelArg(k, 3, sizeof mode, &mode));
    size_t wg = WG;
    CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &wg, &wg, 0, nullptr, nullptr));
    CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, N * esz, out.data(), 0, nullptr, nullptr));
    int bad = 0;
    CHECK1(clEnqueueReadBuffer(q, bbad, 1, 0, 4, &bad, 0, nullptr, nullptr));

    // Model: slot -> element, per the generic code.
    std::vector<int> slotI(N, -1), slotM(N, -1);
    for (unsigned i = 0; i < R; ++i)
      for (unsigned me = 0; me < WG; ++me) {
        unsigned mask = f - 1;
        size_t s = size_t(i) * f + size_t(me & ~mask) * R + (me & mask);
        if (s < N) slotI[s] = int(i), slotM[s] = int(me);
      }
    size_t wrong = 0;
    for (unsigned i = 0; i < R; ++i)
      for (unsigned me = 0; me < WG; ++me) {
        double x, y;
        get(out, i * WG + me, x, y);
        double ex, ey;
        if (mode < 3) {
          size_t s = size_t(i) * WG + me;
          unsigned c0 = code(unsigned(slotI[s]), unsigned(slotM[s]));
          ex = c0; ey = double(yOf(c0));
        } else {
          size_t sa = size_t(i) * (WG / 2) + me % (WG / 2), sb = 4 * size_t(WG) + size_t(i) * (WG / 2) + me % (WG / 2);
          unsigned ca = code(unsigned(slotI[sa]), unsigned(slotM[sa])), cb = code(unsigned(slotI[sb]), unsigned(slotM[sb]));
          double ax = ca, bx = cb, ay = double(yOf(ca)), by = double(yOf(cb));
          // GF subtraction wraps modulo the field prime, so compare modulo it.
          if (me < WG / 2) { ex = ax + bx; ey = ay + by; } else { ex = ax - bx; ey = ay - by; }
          if (c.type == GF31 || c.type == GF61) {
            double p = c.type == GF31 ? 2147483647.0 : 2305843009213693951.0;
            auto red = [&](double v) { while (v < 0) v += p; while (v >= p) v -= p; return v; };
            ex = red(ex); ey = red(ey); x = red(x); y = red(y);
          }
        }
        if (x != ex || y != ey) ++wrong;
      }
    if (wrong || bad) {
      ++fails;
      char buf[160];
      std::snprintf(buf, sizeof buf, "[f=%u%s: %zu of %u elements wrong%s] ", f, mode == 3 ? " fused" : "", wrong, R * WG, bad ? ", wrote outside LDS_BYTES" : "");
      msg += buf;
    }
  }
  detail = fails ? msg : "ok";
  (void) verbose;
  return fails;
}

int main(int argc, char** argv) {
  std::string dir = "src/cl";
  bool verbose = false;
  for (int a = 1; a < argc; ++a) { if (!strcmp(argv[a], "-v")) verbose = true; else dir = argv[a]; }
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  char ext[8192] = {};
  clGetDeviceInfo(dev, 0x1030 /* CL_DEVICE_EXTENSIONS */, sizeof(ext) - 1, ext, nullptr);
  const bool hasFp64 = std::string(ext).find("cl_khr_fp64") != std::string::npos;
  size_t maxWg = 0;
  clGetDeviceInfo(dev, 0x1004 /* CL_DEVICE_MAX_WORK_GROUP_SIZE */, sizeof maxWg, &maxWg, nullptr);

  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);
  struct Shape { unsigned width, nw; };
  const Shape shapes[] = {{256, 4}, {512, 8}, {1024, 8}, {1024, 4}, {4096, 8}};
  struct Mode { int pad, swiz; };
  const Mode modes[] = {{0, 0}, {1, 0}, {0, 1}};
  int total = 0, failed = 0, buildFailed = 0;
  for (Type t : {FP64, FP32, GF31, GF61}) {
    if ((t == FP64 || t == GF61) && !hasFp64 && t == FP64) continue;
    for (const Shape& s : shapes) {
      if (s.width / s.nw > maxWg) continue;
      for (unsigned bytes : {4u, 8u, 16u}) {
        for (const Mode& m : modes) {
          // The fused 1K radix-8 shuffle + fft2 has no 4-byte implementation; that combination is a compile-time #error.
          if (bytes == 4 && s.width == 1024 && s.nw == 8) continue;
          Case c{t, s.width, s.nw, bytes, m.pad, m.swiz};
          std::string detail;
          int r = runCase(dev, ctx, q, dir, c, verbose, detail);
          ++total;
          if (r != 0) { ++failed; if (r < 0) ++buildFailed; }
          if (r != 0 || verbose)
            std::printf("%s width %4u radix %u SHUFL_BYTES=%2u LDSPAD=%d LDSSWIZ=%d: %s\n", typeName[t], s.width, s.nw, bytes, m.pad, m.swiz, detail.c_str());
        }
      }
    }
  }
  std::printf("%d cases, %d failed (%d did not build)\n", total, failed, buildFailed);
  std::printf("%s\n", failed ? "FAIL" : "ok");
  return failed ? 1 : 0;
}
