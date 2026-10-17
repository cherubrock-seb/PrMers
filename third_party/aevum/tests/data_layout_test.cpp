// Host test: the FFT data buffers must hold every element the layouts in cl/middle.cl address.
//
// The test carries literal copies of the address arithmetic of the four out-of-place layouts and the two
// in-place layouts, enumerates their highest element index for a range of shapes, MIDDLE values, PAD, IN_WG/IN_SIZEX
// and OUT_WG/OUT_SIZEX settings, and compares it with the size middleDataElements() allocates.  The address
// functions are linear in every loop variable, so each one is enumerated over its full range in one variable with
// the others at their extremes, which finds the same maximum as the full product.
//
// It also counts how many of the same configurations the fixed-ratio table used before middleDataElements() would
// have undersized (informational, but it must be non-zero or this test could not catch the defect).

#include "DataLayout.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using u64 = uint64_t;
using u32 = uint32_t;

// The sizing used before middleDataElements(), in units of FFT elements (it was in units of sizeof(double)).
u64 legacyElements(u32 W, u32 M, u32 H, u32 inplace, u32 pad) {
  u64 const N = u64(W) * M * H * 2;
  u64 r = inplace ? 3 * N / 2 : pad == 0 ? N : pad <= 128 ? 9 * N / 8 : pad <= 256 ? 5 * N / 4 : 3 * N / 2;
  if (!inplace && pad != 0 && M == 4) r = r * 5 / 4;
  return r / 2;
}

struct Geometry { u32 W, M, H, pad, in_wg, in_sizex, out_wg, out_sizex; };

// Highest element index + 1 touched by the out-of-place layouts of middle.cl.
u64 simulateOutOfPlace(Geometry const& g) {
  u64 const W = g.W, M = g.M, H = g.H;
  u64 const PAD_SIZE = g.pad / 16;
  u64 const BIG_PAD_SIZE = (PAD_SIZE / 2 + 1) * PAD_SIZE;
  u64 hi = 0;
  auto touch = [&](u64 idx) { if (idx > hi) hi = idx; };

  // 1. writeCarryFusedLine / readMiddleInLine
  for (u64 line = 0; line < M * H; ++line) {
    touch(line * W + line * PAD_SIZE + line / H * BIG_PAD_SIZE + (W - 1));
  }

  // 3. writeTailFusedLine / readMiddleOutLine
  for (u64 line = 0; line < M * W; ++line) {
    u64 const big = (PAD_SIZE && (M == 4 || M == 8 || M == 16)) ? line / M * BIG_PAD_SIZE : 0;
    touch(line * (H + PAD_SIZE) + big + (H - 1));
  }

  // 2. writeMiddleInLine / readTailFusedLine and 4. writeMiddleOutLine / readCarryFusedLine.
  auto chunked = [&](u64 xExtent, u64 yExtent, u64 sizex, u64 wg) {
    u64 const sizey = wg / sizex;
    u64 const nx = xExtent / sizex, ny = yExtent / sizey;
    auto at = [&](u64 chunk_y, u64 chunk_x) {
      u64 idx;
      if (PAD_SIZE) {
        idx = chunk_y * (M * wg + PAD_SIZE) + chunk_x * (yExtent * M * sizex + yExtent / sizey * PAD_SIZE + BIG_PAD_SIZE);
      } else {
        idx = chunk_y * M * wg + chunk_x * M * yExtent * sizex;
      }
      touch(idx + (M - 1) * wg + (wg - 1));       // last i, last lane
    };
    for (u64 cx = 0; cx < nx; ++cx) { at(0, cx); at(ny - 1, cx); }
    for (u64 cy = 0; cy < ny; ++cy) { at(cy, 0); at(cy, nx - 1); }
  };
  chunked(W, H, g.in_sizex, g.in_wg);       // fftMiddleIn:  x < WIDTH, y < SMALL_HEIGHT
  chunked(H, W, g.out_sizex, g.out_wg);     // fftMiddleOut: x < SMALL_HEIGHT, y < WIDTH
  return hi + 1;
}

// Highest element index + 1 touched by the in-place layouts (INPLACE=1 pads SIZEM by 16, INPLACE=2 does not).
u64 simulateInPlace(Geometry const& g, u32 inplace) {
  u64 const W = g.W, M = g.M, H = g.H;
  u64 const SIZEBLK = H, SIZEW = 16 * SIZEBLK + 16, SIZEM = W / 16 * SIZEW + (inplace == 1 ? 16 : 0);
  u64 hi = 0;
  for (u64 middle = 0; middle < M; ++middle) {
    for (u64 y = 0; y < H; ++y) {
      u64 const x = W - 1;                                                            // highest x of the row
      u64 const swz = (y / 16) ^ (y % 16);                                            // SWIZ(y % 16, y / 16)
      u64 const idx = x / 16 * SIZEW + middle * SIZEM + y % 16 * SIZEBLK + swz * 16 + x % 16;
      if (idx > hi) hi = idx;
    }
  }
  return hi + 1;
}

}  // namespace

int main() {
  const u32 widths[] = {256, 512, 1024, 2048, 4096};
  const u32 heights[] = {256, 512, 1024, 2048};
  const u32 pads[] = {0, 16, 32, 64, 128, 192, 256, 384, 512};
  const u32 wgs[] = {64, 128, 256};
  const u32 sizexs[] = {4, 8, 16, 32};

  u64 configs = 0, undersized = 0, legacyUndersized = 0, shrunk = 0, wasteful = 0;
  u64 firstBad[9] = {};
  bool haveBad = false;

  for (u32 W : widths) for (u32 H : heights) for (u32 M = 1; M <= 16; ++M) {
    // INPLACE layouts (pad is not used by them).
    for (u32 inplace = 1; inplace <= 2; ++inplace) {
      Geometry g{W, M, H, 0, 128, 16, 128, 16};
      u64 const need = simulateInPlace(g, inplace);
      u64 const have = middleDataElements(W, M, H, inplace, 0, 0, 0, 0, 0);
      ++configs;
      if (have < need) { ++undersized; if (!haveBad) { haveBad = true; firstBad[0] = W; firstBad[1] = M; firstBad[2] = H; firstBad[3] = inplace; } }
    }

    for (u32 pad : pads) {
      // Vary IN_* with default OUT_*, OUT_* with default IN_*, and both together.
      struct Combo { u32 iw, is, ow, os; };
      std::vector<Combo> combos;
      for (u32 wg : wgs) for (u32 sx : sizexs) {
        if (sx > wg) continue;
        combos.push_back({wg, sx, 128, 16});
        combos.push_back({128, 16, wg, sx});
        combos.push_back({wg, sx, wg, sx});
      }
      for (Combo const& c : combos) {
        // The kernels need the chunk grid to divide evenly.
        if (W % c.is || H % (c.iw / c.is) || H % c.os || W % (c.ow / c.os)) continue;
        Geometry g{W, M, H, pad, c.iw, c.is, c.ow, c.os};
        u64 const need = simulateOutOfPlace(g);
        u64 const have = middleDataElements(W, M, H, 0, int(pad), c.iw, c.is, c.ow, c.os);
        ++configs;
        if (have < need) {
          ++undersized;
          if (!haveBad) { haveBad = true; firstBad[0] = W; firstBad[1] = M; firstBad[2] = H; firstBad[3] = 0; firstBad[4] = pad; firstBad[5] = c.iw; firstBad[6] = c.is; firstBad[7] = c.ow; firstBad[8] = c.os; }
        }
        if (have < legacyElements(W, M, H, 0, pad)) ++shrunk;
        if (legacyElements(W, M, H, 0, pad) < need) ++legacyUndersized;
        if (pad == 0 && have != u64(W) * M * H) ++wasteful;            // unpadded layout needs exactly N
      }
    }
  }

  std::printf("configurations: %llu, undersized allocations: %llu, previously undersized: %llu\n",
              (unsigned long long) configs, (unsigned long long) undersized, (unsigned long long) legacyUndersized);

  int rc = 0;
  if (undersized) {
    std::printf("FAIL: W=%llu M=%llu H=%llu inplace=%llu PAD=%llu IN_WG=%llu IN_SIZEX=%llu OUT_WG=%llu OUT_SIZEX=%llu is undersized\n",
                (unsigned long long) firstBad[0], (unsigned long long) firstBad[1], (unsigned long long) firstBad[2],
                (unsigned long long) firstBad[3], (unsigned long long) firstBad[4], (unsigned long long) firstBad[5],
                (unsigned long long) firstBad[6], (unsigned long long) firstBad[7], (unsigned long long) firstBad[8]);
    rc = 1;
  }
  if (shrunk) { std::printf("FAIL: %llu allocations are smaller than the previous fixed-ratio sizing\n", (unsigned long long) shrunk); rc = 1; }
  if (wasteful) { std::printf("FAIL: %llu unpadded allocations differ from the exact size\n", (unsigned long long) wasteful); rc = 1; }
  if (!legacyUndersized) { std::printf("FAIL: the check cannot detect the defect it guards against\n"); rc = 1; }

  // The MIDDLE=1, PAD=512, IN_WG=64 case that the fixed ratios sized at 1.5 * N: it needs more.
  {
    u64 const have = middleDataElements(1024, 1, 256, 0, 512, 64, 4, 128, 16);
    u64 const legacy = legacyElements(1024, 1, 256, 0, 512);
    std::printf("W=1024 M=1 H=256 PAD=512 IN_WG=64 IN_SIZEX=4: %llu elements (previously %llu)\n",
                (unsigned long long) have, (unsigned long long) legacy);
    if (have <= legacy) { std::printf("FAIL: expected a larger allocation than the fixed ratio\n"); rc = 1; }
  }

  // Zero WG/SIZEX select the kernel defaults, so a zero must not divide by zero or shrink the allocation.
  if (middleDataElements(1024, 1, 256, 0, 512, 0, 0, 0, 0) != middleDataElements(1024, 1, 256, 0, 512, 128, 16, 128, 16)) {
    std::printf("FAIL: zero IN_WG/IN_SIZEX/OUT_WG/OUT_SIZEX must act as the defaults\n");
    rc = 1;
  }

  if (!rc) std::printf("data layout test passed\n");
  return rc;
}
