// Plans a clEnqueueFillBuffer on the CUDA backend: which cuMemset call(s) write a pattern of 1, 2, 4, 8, 16, 32,
// 64 or 128 bytes over a range of a buffer, and whether the request is invalid (CL_INVALID_VALUE) the way the
// OpenCL runtime rejects it.
//
// The CUDA driver only memsets 8, 16 or 32-bit values.  A wider pattern (Buffer<i64>/<u64>/<double> = 8 bytes,
// Buffer<double2> = 16) is written one 32-bit word at a time with a strided 2D memset: word i of the pattern goes to
// every pattern-sized slot, so a pattern of N words takes N driver calls.  A pattern whose words are all equal
// (the zero fill that is by far the common case) is a single 32-bit memset instead.
//
// Plain integers and no CUDA includes, so the planning is tested on a host without CUDA (tests/cuda_fill_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cuda_fill {

enum class Op {
  Invalid,    // reject with CL_INVALID_VALUE
  Memset8,    // cuMemsetD8Async(dst, byte, count)
  Memset16,   // cuMemsetD16Async(dst, halfword, count)
  Memset32,   // cuMemsetD32Async(dst, word, count)
  Strided32,  // for i in [0, words): cuMemsetD2D32Async(dst + 4*i, pitch, word i, 1, count, ...)
};

struct Plan {
  Op op = Op::Invalid;
  size_t offset = 0;   // first byte written (the caller's offset)
  size_t count = 0;    // Memset*: elements of the memset width; Strided32: rows (pattern-sized slots), width 1
  size_t pitch = 0;    // Strided32: bytes between rows (the pattern size)
  unsigned words = 0;  // Strided32: 32-bit words of the pattern, one driver call each
};

// patternSize must be one OpenCL allows (a power of two from 1 to 128), offset and size multiples of it, size > 0,
// and offset + size inside a buffer of bufSize bytes.
inline bool validRequest(const void* pattern, size_t patternSize, size_t offset, size_t size, size_t bufSize) {
  if (!pattern || patternSize == 0 || patternSize > 128 || (patternSize & (patternSize - 1))) return false;
  if (size == 0 || size % patternSize || offset % patternSize) return false;
  return offset <= bufSize && size <= bufSize - offset;
}

inline Plan plan(const void* pattern, size_t patternSize, size_t offset, size_t size, size_t bufSize) {
  Plan p;
  if (!validRequest(pattern, patternSize, offset, size, bufSize)) return p;
  p.offset = offset;
  if (patternSize == 1)      { p.op = Op::Memset8;  p.count = size; return p; }
  if (patternSize == 2)      { p.op = Op::Memset16; p.count = size / 2; return p; }
  if (patternSize == 4)      { p.op = Op::Memset32; p.count = size / 4; return p; }
  p.words = unsigned(patternSize / 4);
  const unsigned char* b = static_cast<const unsigned char*>(pattern);
  bool uniform = true;
  for (unsigned i = 1; i < p.words && uniform; ++i) uniform = !std::memcmp(b, b + 4 * i, 4);
  if (uniform) { p.op = Op::Memset32; p.count = size / 4; return p; }
  p.op = Op::Strided32;
  p.count = size / patternSize;
  p.pitch = patternSize;
  return p;
}

// Word i (host byte order, as the memset takes it) of a pattern.
inline uint32_t patternWord(const void* pattern, unsigned i) {
  uint32_t w;
  std::memcpy(&w, static_cast<const unsigned char*>(pattern) + 4 * size_t(i), 4);
  return w;
}

}  // namespace cuda_fill
