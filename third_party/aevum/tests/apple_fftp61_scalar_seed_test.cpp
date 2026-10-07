// Host test: fftP61WeightScalarApple must compute the exact GF61 weight shift for every word of a 2^27-word transform.
//
// The kernel seeds the shift with  make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF)  in u32 arithmetic.  word_index spans the
// whole transform here, so at NWORDS = 2^27 the product wraps for shifts >= 33 and, since 2^32 mod 61 = 57, the shift changes.  The harness
// compiles the real fftp.cl with the Apple split-fftP define (this kernel is plain OpenCL C 1.2, so any OpenCL device runs it), feeds the value
// 1 in both components, and compares the output 2^shift against a reference computed with the product taken exactly.  Only a sample of the
// 2^26 work-items is run (blocks of 256 spread over the whole range, plus the last block).
//
// usage: apple_fftp61_scalar_seed_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* HARNESS = "#include \"fftp.cl\"\n";

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  unsigned long long maxAlloc = 0;
  clGetDeviceInfo(dev, 0x1010 /* CL_DEVICE_MAX_MEM_ALLOC_SIZE */, sizeof maxAlloc, &maxAlloc, nullptr);

  const u32 WIDTH = 4096, SMALL_HEIGHT = 1024, MIDDLE = 16;
  const u64 N = u64(WIDTH) * SMALL_HEIGHT * MIDDLE * 2;       // NWORDS = 2^27
  const size_t items = N / 2;                                  // one Word2 / GF61 per work-item
  if (maxAlloc < items * 16) { std::printf("SKIP: device max allocation %llu bytes is below the 1 GiB the 2^27-word shape needs\n", (unsigned long long) maxAlloc); return 0; }

  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  // Several exponents, so that bigword_weight_shift_minus1 (the multiplier) takes values both below and above 33.
  const u64 exponents[] = {2000000011ull, 1000000021ull, 3000000019ull};
  u64 totalBad = 0, totalChecked = 0;
  for (u64 E : exponents) {
    // FRAC_BPW as the host computes it (Gpu.cpp): frac(E / N) * 2^64, minus one.
    u32 bpw_hi = (u64(E % N) << 32) / N;
    u32 bpw_lo = (((u64(E % N) << 32) % N) << 32) / N;
    u64 bpw = (u64(bpw_hi) << 32) + bpw_lo;
    bpw--;
    const u32 fracHi = u32(bpw >> 32), fracLo = u32(bpw);
    // M61_LOG2_ROOT_TWO (base.cl, power-of-two NWORDS): ((1 << 60) / NWORDS) % 61.
    const u32 log2RootTwo = u32(((1ull << 60) / N) % 61);
    const u32 bigShift = u32(((N - E % N) * log2RootTwo) % 61);   // computed in 32 bits in the kernel; N * 61 < 2^32 is not guaranteed, so see below
    (void) bigShift;

    Program prog = loadSource(ctx, HARNESS);
    std::string opts = "-cl-std=CL1.2 -I " + dir + " -DAEVUM_APPLE_SPLIT_FFTP=1 -DNO_ASM=1 -DAMDGPU=0 "
      "-DFFT_TYPE=1 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=1 -DWordSize=8u -DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u "
      "-DEXP=" + std::to_string(E) + "u -DWIDTH=" + std::to_string(WIDTH) + "u -DSMALL_HEIGHT=" + std::to_string(SMALL_HEIGHT) +
      "u -DMIDDLE=" + std::to_string(MIDDLE) + "u -DNW=8u -DNH=8u -DFFT_VARIANT=101u "
      "-DDISTGF31=0 -DDISTWTRIGGF31=0 -DDISTMTRIGGF31=0 -DDISTHTRIGGF31=0 -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul "
      "-DFRAC_BPW_HI=" + std::to_string(fracHi) + "u -DFRAC_BPW_LO=" + std::to_string(fracLo) + "u "
      "-DTAILTGF31=U2(1u,1u) -DTAILTGF61=U2(1ul,1ul)";
    if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
      std::printf("FAIL: build\n%s\n", getBuildLog(prog.get(), dev).c_str());
      return 1;
    }
    int err = 0;
    cl_kernel k = clCreateKernel(prog.get(), "fftP61WeightScalarApple", &err);
    CHECK1(err);
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, items * 16, nullptr, &err);
    CHECK1(err);
    cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, items * 16, nullptr, &err);
    CHECK1(err);
    // Word2 = long2 (WordSize 8).  Every component is 1, so the output component is 2^shift.
    std::vector<long long> ones(items * 2, 1);
    CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, items * 16, ones.data(), 0, nullptr, nullptr));
    CHECK1(clSetKernelArg(k, 0, sizeof bout, &bout));
    CHECK1(clSetKernelArg(k, 1, sizeof bin, &bin));

    // The kernel's own bigword_weight_shift: (NWORDS - EXP % NWORDS) * log2_root_two % 61, all in u32.
    const u32 s = u32(u32(N - E % N) * log2RootTwo) % 61;
    const u32 s1 = (s + 60) % 61;

    u64 bad = 0, checked = 0;
    std::vector<size_t> starts;
    for (size_t b = 0; b < 48; ++b) starts.push_back((items / 48 * b + 12345 * b) / 256 * 256);
    starts.push_back(items - 256);
    for (size_t start : starts) {
      size_t global = 256, local = 256;
      CHECK1(clEnqueueNDRangeKernel(q, k, 1, &start, &global, &local, 0, nullptr, nullptr));
      std::vector<unsigned long long> out(256 * 2);
      CHECK1(clEnqueueReadBuffer(q, bout, 1, start * 16, 256 * 16, out.data(), 0, nullptr, nullptr));
      for (size_t i = 0; i < 256; ++i) {
        const u64 p = start + i;
        const u64 line = p / WIDTH, x = p % WIDTH;
        const u64 wi = (line + u64(SMALL_HEIGHT) * MIDDLE * x) * 2;       // BIG_HEIGHT = SMALL_HEIGHT * MIDDLE
        // comboFracBits(word_index) = wi * (FRAC_BPW_HI + 1) - 1 for a power-of-two NWORDS (u64).
        const u64 cfb = wi * u64(fracHi + 1u) - 1;
        // Shift = high word of cfb + (wi * s1 : 0xFFFFFFFF), taken mod 61, with the product exact.
        const u64 lo = (cfb & 0xFFFFFFFFull) + 0xFFFFFFFFull;
        const u64 hi = (((cfb >> 32) + (lo >> 32)) & 0xFFFFFFFFull) + (wi % 61) * s1;   // word 0: cfb = -1 wraps in the kernel's u64 add
        const u32 shift0 = u32(hi % 61);
        // The second component's shift: combo += (s1 : FRAC_BPW_HI), then adjust_m61_weight_shift (a single conditional subtraction of 61).
        u64 lo1 = (lo & 0xFFFFFFFFull) + fracHi;
        u64 hi1 = (hi % 61) + s1 + (lo1 >> 32);
        u32 shift1 = u32(hi1);
        if (shift1 >= 61) shift1 -= 61;
        if (shift1 >= 61) shift1 -= 61;
        ++checked;
        if (out[2 * i] != (1ull << shift0) || out[2 * i + 1] != (1ull << shift1)) ++bad;
      }
    }
    std::printf("E=%llu (bigword_weight_shift_minus1 = %u): %llu of %llu sampled words have the wrong GF61 weight\n",
                (unsigned long long) E, s1, (unsigned long long) bad, (unsigned long long) checked);
    totalBad += bad;
    totalChecked += checked;
  }
  std::printf("%s\n", totalBad ? "FAIL" : "ok");
  return totalBad ? 1 : 0;
}
