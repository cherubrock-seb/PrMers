// Limits on carryFused's width multiplier (WMUL).  Kept free of OpenCL so it can be unit tested on the host.
#pragma once

#include <algorithm>
#include <cstdint>

namespace wmul_limit {

// The largest WMUL carryFused can be built with.  carryFused runs G_W * WMUL threads per workgroup (G_W = WIDTH / NW) and needs
// WMUL * WIDTH * SHUFL_BYTES_W bytes of local memory.  So WMUL is limited by the local memory (at most 32KB is planned for), by
// the device's maximum workgroup size, and, because the CUDA compiler has been seen to refuse a kernel with 1024 threads, to 2
// for a 1K width and to 1 for a 4K width.  The result is at least 1: a width that cannot fit even one line is rejected elsewhere.
inline uint32_t maxWmul(uint32_t width, uint32_t nW, uint32_t shuflBytesW, uint64_t localMemSize, uint32_t maxWorkGroupSize) {
  uint64_t const ldsLimit = std::min<uint64_t>(32768, localMemSize);
  uint64_t maxW = ldsLimit / (uint64_t(width) * shuflBytesW);
  if (maxW > 2 && width >= 1024) maxW = 2;
  if (maxW > 1 && width >= 4096) maxW = 1;
  maxW = std::min<uint64_t>(maxW, maxWorkGroupSize / (width / nW));
  return uint32_t(std::max<uint64_t>(maxW, 1));
}

// Clamp a requested WMUL to [1, max] and then down to a divisor of BIG_HEIGHT, as carryFused is launched with
// BIG_HEIGHT / WMUL + 1 workgroups.
inline uint32_t clampWmul(uint32_t requested, uint32_t maxWmul, uint32_t bigHeight) {
  uint32_t wmul = std::min(requested, maxWmul);
  if (wmul < 1) wmul = 1;
  while (bigHeight % wmul) --wmul;
  return wmul;
}

}  // namespace wmul_limit
