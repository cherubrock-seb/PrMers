// Host test, no CUDA and no device: the CUDA backend's clEnqueueFillBuffer plan (src/cuda/CudaFill.h).  Each plan is
// executed here against a byte array with the exact semantics of the cuMemset calls it names, and the result must
// equal a byte-by-byte OpenCL fill.  The CUDA backend used to write zero for every pattern that was not 1 or 4 bytes,
// so Buffer<Word>::set(v) (8 bytes) stored 0.
#include "cuda/CudaFill.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cuda_fill;

static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } } while (0)

// The OpenCL definition: the pattern repeated over [offset, offset + size).
static void reference(std::vector<unsigned char>& mem, const unsigned char* pat, size_t patSize, size_t offset, size_t size) {
  for (size_t i = 0; i < size; ++i) mem[offset + i] = pat[i % patSize];
}

// cuMemsetD8/D16/D32 and cuMemsetD2D32 (width 1) on a host array.
static void execute(std::vector<unsigned char>& mem, const Plan& p, const unsigned char* pat) {
  switch (p.op) {
  case Op::Invalid: break;
  case Op::Memset8:  for (size_t i = 0; i < p.count; ++i) mem[p.offset + i] = pat[0]; break;
  case Op::Memset16: for (size_t i = 0; i < p.count; ++i) std::memcpy(&mem[p.offset + 2 * i], pat, 2); break;
  case Op::Memset32: {
    uint32_t w = patternWord(pat, 0);
    for (size_t i = 0; i < p.count; ++i) std::memcpy(&mem[p.offset + 4 * i], &w, 4);
    break;
  }
  case Op::Strided32:
    for (unsigned i = 0; i < p.words; ++i) {
      uint32_t w = patternWord(pat, i);
      for (size_t row = 0; row < p.count; ++row) std::memcpy(&mem[p.offset + 4 * i + row * p.pitch], &w, 4);
    }
    break;
  }
}

static void checkFill(const unsigned char* pat, size_t patSize, size_t offset, size_t size, size_t bufSize, const char* what) {
  std::vector<unsigned char> expect(bufSize, 0xAB), got(bufSize, 0xAB);
  reference(expect, pat, patSize, offset, size);
  Plan p = plan(pat, patSize, offset, size, bufSize);
  char msg[160];
  std::snprintf(msg, sizeof msg, "%s: patSize %zu offset %zu size %zu is accepted", what, patSize, offset, size);
  EXPECT(p.op != Op::Invalid, msg);
  execute(got, p, pat);
  std::snprintf(msg, sizeof msg, "%s: patSize %zu offset %zu size %zu matches the OpenCL fill", what, patSize, offset, size);
  EXPECT(got == expect, msg);
}

int main() {
  // The reported bug: an 8-byte pattern (Buffer<i64>::set(3)) must store 3, not 0.
  {
    int64_t v = 3;
    std::vector<unsigned char> mem(64, 0xFF);
    Plan p = plan(&v, 8, 0, 8, mem.size());
    execute(mem, p, reinterpret_cast<unsigned char*>(&v));
    int64_t out[2];
    std::memcpy(out, mem.data(), 16);
    EXPECT(out[0] == 3 && uint8_t(mem[8]) == 0xFF, "an 8-byte fill of 3 over one element stores 3 and touches nothing else");
  }
  {
    int64_t v = 0x1122334455667788LL;
    std::vector<unsigned char> mem(64, 0);
    Plan p = plan(&v, 8, 8, 32, mem.size());
    EXPECT(p.op == Op::Strided32 && p.words == 2 && p.count == 4 && p.pitch == 8, "a mixed-word 8-byte pattern is one strided memset per word");
  }

  // Every OpenCL pattern size, with several offsets and lengths, constant and varied patterns, against the reference.
  const size_t sizes[] = {1, 2, 4, 8, 16, 32, 64, 128};
  for (size_t ps : sizes) {
    unsigned char varied[128], zero[128] = {0}, ones[128], sameWords[128];
    for (size_t i = 0; i < 128; ++i) { varied[i] = (unsigned char)(0x11 * (i % 15) + 7); ones[i] = 0xFF; sameWords[i] = (unsigned char)(0x10 + i % 4); }
    const unsigned char* pats[] = {varied, zero, ones, sameWords};
    const char* names[] = {"varied", "zero", "all-ones", "equal-words"};
    const size_t bufSize = 2048;
    for (int k = 0; k < 4; ++k) {
      for (size_t offset : {size_t(0), ps, 5 * ps, 256 - (256 % ps), size_t(1024)}) {
        for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(33), size_t(100)}) {
          if (offset + n * ps > bufSize) continue;
          checkFill(pats[k], ps, offset, n * ps, bufSize, names[k]);
        }
      }
    }
    // A fill of the whole buffer, exactly at the end.
    checkFill(varied, ps, 0, bufSize, bufSize, "whole buffer");
    checkFill(varied, ps, bufSize - ps, ps, bufSize, "last element");
  }

  // Uniform wide patterns collapse to a single 32-bit memset (the zero fill of Buffer<Word>/<double>).
  {
    unsigned char zero[16] = {0};
    Plan p = plan(zero, 16, 0, 160, 160);
    EXPECT(p.op == Op::Memset32 && p.count == 40, "a zero double2 fill is one 32-bit memset");
    uint32_t w2[2] = {0xDEADBEEF, 0xDEADBEEF};
    p = plan(w2, 8, 0, 80, 80);
    EXPECT(p.op == Op::Memset32 && p.count == 20, "an 8-byte pattern of equal words is one 32-bit memset");
    uint32_t w3[2] = {0xDEADBEEF, 0xDEADBEEE};
    p = plan(w3, 8, 0, 80, 80);
    EXPECT(p.op == Op::Strided32, "an 8-byte pattern of different words is not collapsed");
    uint32_t w4[4] = {1, 1, 1, 2};
    p = plan(w4, 16, 0, 64, 64);
    EXPECT(p.op == Op::Strided32 && p.words == 4, "only the last word differing is not collapsed");
  }

  // What OpenCL rejects with CL_INVALID_VALUE is rejected, never zero-filled.
  {
    unsigned char pat[256] = {1};
    const size_t bufSize = 1024;
    EXPECT(plan(nullptr, 4, 0, 4, bufSize).op == Op::Invalid, "a null pattern is invalid");
    for (size_t bad : {size_t(0), size_t(3), size_t(5), size_t(6), size_t(7), size_t(12), size_t(24), size_t(100), size_t(129), size_t(256)})
      EXPECT(plan(pat, bad, 0, 768, bufSize).op == Op::Invalid, "a pattern size that is not 1, 2, 4, ... 128 is invalid");
    EXPECT(plan(pat, 8, 0, 0, bufSize).op == Op::Invalid, "a zero size is invalid");
    EXPECT(plan(pat, 8, 0, 12, bufSize).op == Op::Invalid, "a size that is not a multiple of the pattern is invalid");
    EXPECT(plan(pat, 4, 0, 6, bufSize).op == Op::Invalid, "a 4-byte pattern over 6 bytes is invalid (no truncation)");
    EXPECT(plan(pat, 8, 4, 16, bufSize).op == Op::Invalid, "an offset that is not a multiple of the pattern is invalid");
    EXPECT(plan(pat, 2, 1, 2, bufSize).op == Op::Invalid, "an odd offset for a 2-byte pattern is invalid");
    EXPECT(plan(pat, 8, 1016, 16, bufSize).op == Op::Invalid, "a range past the end of the buffer is invalid");
    EXPECT(plan(pat, 8, 1032, 8, bufSize).op == Op::Invalid, "an offset past the end of the buffer is invalid");
    EXPECT(plan(pat, 8, 16, SIZE_MAX - 7, bufSize).op == Op::Invalid, "an offset + size that overflows is invalid");
    EXPECT(plan(pat, 8, SIZE_MAX - 7, 8, bufSize).op == Op::Invalid, "a huge offset is invalid");
    EXPECT(plan(pat, 8, 1016, 8, bufSize).op == Op::Strided32 || plan(pat, 8, 1016, 8, bufSize).op == Op::Memset32, "the last element is valid");
  }

  std::printf("%s\n", failures ? "FAIL" : "ok");
  return failures ? 1 : 0;
}
