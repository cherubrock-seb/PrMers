// Size of the FFT/NTT data buffers (buf1, buf2, buf3, and the prepared transform buffers).
//
// Header-only and free of OpenCL/Gpu dependencies so that a host test can enumerate configurations against it.

#pragma once

#include <algorithm>
#include <cstdint>

// Largest PAD (in bytes) accepted by -use PAD=.  middleDataElements() sizes any value exactly, but the padded
// layouts grow quickly with PAD and nothing above 512 bytes has been evaluated.
constexpr int AEVUM_MAX_PAD = 512;

// Number of FFT elements (one T2, F2, GF31 or GF61 value each) that the data layouts in cl/middle.cl can address.
//
// The data buffers hold, at different points of one iteration, four different layouts, and the allocation has
// to cover the largest of them.  PAD_SIZE is PAD/16 (the pad in elements) and BIG_PAD_SIZE is
// (PAD_SIZE/2+1)*PAD_SIZE.
//
//   1. writeCarryFusedLine / readMiddleInLine (also fftP and fftW):
//        line * (WIDTH + PAD_SIZE) + line/SMALL_HEIGHT * BIG_PAD_SIZE + x,  line < MIDDLE*SMALL_HEIGHT, x < WIDTH.
//   2. writeMiddleInLine / readTailFusedLine, the chunked layout
//        chunk_y * (MIDDLE*IN_WG + PAD_SIZE) + chunk_x * (SMALL_HEIGHT*MIDDLE*IN_SIZEX + SMALL_HEIGHT/(IN_WG/IN_SIZEX)*PAD_SIZE + BIG_PAD_SIZE)
//          + i*IN_WG + lane,
//        chunk_x < WIDTH/IN_SIZEX, chunk_y < SMALL_HEIGHT/(IN_WG/IN_SIZEX), i < MIDDLE, lane < IN_WG.
//   3. writeTailFusedLine / readMiddleOutLine:
//        line * (SMALL_HEIGHT + PAD_SIZE) + (MIDDLE is 4, 8 or 16 ? line/MIDDLE * BIG_PAD_SIZE : 0) + x,
//        line < MIDDLE*WIDTH, x < SMALL_HEIGHT.
//   4. writeMiddleOutLine / readCarryFusedLine: the chunked layout of 2 with OUT_WG, OUT_SIZEX, and WIDTH and
//      SMALL_HEIGHT exchanged.
//
// Layouts 2 and 4 depend on IN_WG/IN_SIZEX and OUT_WG/OUT_SIZEX.  Their padding is PAD_SIZE per MIDDLE*WG
// elements, so it is large when MIDDLE is small and the work group is small: MIDDLE=1, PAD=512, IN_WG=64 needs
// 1.5 times the unpadded size plus the big pads, more than a fixed fraction of N granted for every PAD > 256.
//
// The INPLACE layouts need only a little more than the unpadded size and stay within the 3N/2 floor below.
//
// A fixed-ratio table (the sizes before this function existed) is kept as a floor, so that no configuration
// whose allocation was already large enough shrinks.
//
// `in_wg`, `in_sizex`, `out_wg` and `out_sizex` follow cl/middle.cl: zero selects the default (128, 16, 128, 16).
// A geometry that does not divide evenly is rounded up (the kernels do not support it; the result is then an upper
// bound rather than an exact size).
inline uint64_t middleDataElements(uint32_t W, uint32_t M, uint32_t H, uint32_t inplace, int pad,
                                   uint32_t in_wg, uint32_t in_sizex, uint32_t out_wg, uint32_t out_sizex) {
  const uint64_t N = uint64_t(W) * M * H;
  uint64_t need = N;

  const uint64_t padSize = pad > 0 ? uint64_t(pad) / 16 : 0;      // PAD_SIZE in middle.cl
  if (!inplace && padSize) {
    const uint64_t bigPad = (padSize / 2 + 1) * padSize;          // BIG_PAD_SIZE in middle.cl

    // Layout 1: one past the last element, i.e. last element + 1.
    need = std::max(need, uint64_t(M) * H * (W + padSize) - padSize + uint64_t(M - 1) * bigPad);

    // Layout 3.
    uint64_t l3 = uint64_t(M) * W * (H + padSize) - padSize;
    if (M == 4 || M == 8 || M == 16) { l3 += uint64_t(W - 1) * bigPad; }
    need = std::max(need, l3);

    // Layouts 2 and 4.  (`a` is the extent along the x chunks, `b` along the y chunks, `wg` the work-group size.)
    auto chunked = [&](uint64_t a, uint64_t b, uint64_t sizex, uint64_t wg) -> uint64_t {
      if (!sizex) sizex = 16;
      if (!wg) wg = 128;
      const uint64_t sizey = std::max<uint64_t>(wg / sizex, 1);
      const uint64_t nx = (a + sizex - 1) / sizex;                // chunk_x < nx
      const uint64_t ny = (b + sizey - 1) / sizey;                // chunk_y < ny
      const uint64_t strideX = b * M * sizex + b / sizey * padSize + bigPad;
      return (ny - 1) * (M * wg + padSize) + (nx - 1) * strideX + M * wg;
    };
    need = std::max(need, chunked(W, H, in_sizex, in_wg));        // fftMiddleIn
    need = std::max(need, chunked(H, W, out_sizex, out_wg));      // fftMiddleOut
  }

  // The fixed-ratio table, in units of FFT elements (the table was in units of sizeof(double) = two elements).
  // INPLACE=0, MIDDLE=4 with padding has an extra 5/4.
  uint64_t legacy;
  if (inplace) { legacy = 3 * N / 2; }
  else {
    legacy = pad <= 0 ? N : pad <= 128 ? 9 * N / 8 : pad <= 256 ? 5 * N / 4 : 3 * N / 2;
    if (pad > 0 && M == 4) { legacy = legacy * 5 / 4; }
  }
  return std::max(need, legacy);
}

// Turn a count of FFT elements into a buffer size in units of sizeof(double), per element type.
#define FP64_DATA_SIZE(elems)       ((elems) * 2)       // T2   is a double2, 16 bytes
#define FP32_DATA_SIZE(elems)       ((elems) * 1)       // F2   is a float2,   8 bytes
#define GF31_DATA_SIZE(elems)       ((elems) * 1)       // GF31 is a uint2,    8 bytes
#define GF61_DATA_SIZE(elems)       ((elems) * 2)       // GF61 is a ulong2,  16 bytes
#define TOTAL_DATA_SIZE(fft,elems)  ((int)(fft).FFT_FP64 * FP64_DATA_SIZE(elems) + (int)(fft).FFT_FP32 * FP32_DATA_SIZE(elems) + \
                                     (int)(fft).NTT_GF31 * GF31_DATA_SIZE(elems) + (int)(fft).NTT_GF61 * GF61_DATA_SIZE(elems))
