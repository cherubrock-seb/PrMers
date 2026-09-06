// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"

// Calculate the LDS bytes used by shufl
#if LDSPAD && SHUFL_BYTES == 16 && RADIX == 8
#define LDS_BYTES     ((WG * RADIX + 7) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 16 && RADIX == 4
#define LDS_BYTES     ((WG * RADIX + 12) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 8 && RADIX == 8
#define LDS_BYTES     ((WG * RADIX + 56) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 8 && RADIX == 4
#define LDS_BYTES     ((WG * RADIX + 12) * SHUFL_BYTES)
#else
#define LDS_BYTES     (WG * RADIX * SHUFL_BYTES)
#endif

#if FFT_FP64 | NTT_GF61

// Shufl two or more fft_WIDTHs or FFT_HEIGHTs operating on 64-bit values using LDS_BYTES of LDS memory.
// Care is taken that each simultaneous workgroup does not interfere with the LDS memory of other simultaneous workgroups --
// even when operating on differernt sized data elements as can happen in an M31+M61 NTT.
// WG = workgroup size of a single fft_WIDTH or fft_HEIGHT
// n = sizeof array u (nW or nH).  n * WG = WIDTH or HEIGHT
// numWG = number of fft_WIDTHs or fft_HEIGHTs being processed simultaneously
// lowMe = me % WG
// NOTE: shufl routines perform a bar(WG) at the start but not at the end.  After calling shufl, a bar(WG) is required
// before next LDS memory usage.  All routines that use LDS memory MUST OBEY THIS PROTOCOL of bar() before LDS use and
// only bar(WG) required before next use.  ALSO NOTE: the first shufl call does not need to do bar(WG).  A relatively
// minor optimization would be to special case the first shufl call.
void OVERLOAD shufl64(local T2 *lds2, T2 *u, u32 f, u32 numWG, u32 lowMe) {

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  int force_default = 0;
#if NOWG2                       // For timing tests only.  Option to not turn off LDS bank conflict code when numWG > 1.  I've not found a GPU where this is beneficial.
  if (numWG > 1) force_default = 1;
#endif
#if NOLDS2                      // For timing tests only.  Option to not turn off LDS bank for second shufl calls.  I've not found a GPU where this is beneficial.
  if (f != 1) force_default = 1;
#endif

  // If SHUFL_BYTES is 16 we can write the complete T2 value to LDS memory with one instruction.
  if (SHUFL_BYTES == 16) {
    local T2* lds = ((local T2*) lds2);
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T2);

#if LDSPAD
    // Special case first n == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 8, 72..., 16...   lds[64..127] = +1
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad 1 value every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe & 7) * (WG + 1) + (lowMe / 8) * 8 + i] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 8                     + ((lowMe / 8) & 7) * (WG + 1) + (lowMe & 7)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 32 + (lowMe / 64) * 8 + ((lowMe / 8) & 7) * (WG + 1) + (lowMe & 7)]; }
      return;
    }

    // Special case second n == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // No padding of LDS blocks is needed to eliminate bank conflicts!  Groups of 8 threads are already in separate LDS banks.
    // We could however save a bar() by writing to same locations that the previous shufl wrote to.
    if (0 && f == 8 && RADIX == 8) {
      // for (u32 i = 0; i < RADIX; ++i) { lds[something] = u[i]; }
      bar(WG);
      //for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[something]; }
      return;
    }

    // Special case first n == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1, 65..., 16...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad 1 value every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 2) & 3) * (WG + 1) + (lowMe / 8) * 8 + (lowMe & 1) * 4 + i] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG / 4 + (lowMe / 32) * 8 + ((lowMe / 8) & 3) * (WG + 1) + (lowMe & 7)]; }
      return;
    }

    // Special case second n == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16... 4..  lds[64..127] = +1
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 16... 1..   output[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[lowMe / 4 * (WG + 4) + i * 4 + (lowMe & 3)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      return;
    }
#endif

#if LDSSWIZ
    // Special case first n == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.  Swizzle on the first 8 threads written to LDS (multiples of 1) and the first 8 threads read from LDS (multiples of 64).
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 7)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 7)]; }
      return;
    }

    // Special case second n == 8 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // No swizzle of LDS blocks is needed to eliminate bank conflicts.  The first 8 threads written to LDS (multiples of 64) and
    // the first 8 threads read from LDS (multiples of 64) are already in separate LDS banks.
    // We can however save a bar() by writing to same locations that previous shufl wrote to.
    if (!force_default && f == 8 && RADIX == 8) {
      for (u32 i = 0; i < RADIX; ++i) { lds[i * WG + lowMe] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[lowMe / 8 * 64 + i * 8 + (lowMe & 7)]; }
      return;
    }

    // Special case first n == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 8 threads written to LDS (4 multiples of 1 and 2 multiples of 4) and the first 8 threads read from LDS (4 multiples of 64 and 2 multiples of 1).
    if (!force_default && f == 1 && n == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 7)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 7)]; }
      return;
    }

    // Special case second n == 4 to eliminate LDS bank conflicts.  We're writing 16 bytes at a time, which means groups of 8 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 8 threads written to LDS (4 multiples of 64 and 2 multiples of 1) and the first 8 threads read from LDS (4 multiples of 64 and 2 multiples of 4).
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 4)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 4)]; }
      return;
    }
#endif

    // Otherwise, execute the original shufl code
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }
  }

  // If SHUFL_BYTES is 8 we split the T2 values into two T values.  These are written to LDS memory with two instructions.
  else if (SHUFL_BYTES == 8) {
    local T* lds = ((local T*) lds2);
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T);

#if LDSPAD
    // Special case first n == 8 code to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 1, 65..., 16, 80...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad one value after every row to eliminate bank conflicts.
    // AEVUM_GWOLT_1K_FIRST_SHUFL_LDSPAD
    // gpuowl 6cf0dc first radix-8 padded shuffle.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i)
        lds[i * (WG + 2) + lowMe] = u[i].x;

      bar(WG);
      for (u32 i = 0; i < RADIX; ++i)
        u[i].x = lds[i * WG / 8
                   + (lowMe / 8)
                   + (lowMe & 7) * (WG + 2)];

      bar(WG);
      for (u32 i = 0; i < RADIX; ++i)
        lds[i * (WG + 2) + lowMe] = u[i].y;

      bar(WG);
      for (u32 i = 0; i < RADIX; ++i)
        u[i].y = lds[i * WG / 8
                   + (lowMe / 8)
                   + (lowMe & 7) * (WG + 2)];

      return;
    }

    // Special case second n == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every row to eliminate bank conflicts.
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i].x; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i].y; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      return;
    }

    // Special case first n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 3.., 16...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      return;
    }

    // Special case second n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 4...   lds[64..127] = +1
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      return;
    }
#endif

#if LDSSWIZ
    // Special case first n == 8 code to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 1 and 2 multiples of 8) and the first 16 threads read from LDS (8 multiples of 64 and 2 multiples of 1).
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      return;
    }

    // Special case second n == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 64 and 2 multiples of 1) and the first 16 threads read from LDS (8 multiples of 64 and 2 multiples of 8).
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i].x; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i].y; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      return;
    }

    // Special case first n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 1 and 4 multiples of 4) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 1).
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i].x; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i].y; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      return;
    }

    // Special case second n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 64 and 4 multiples of 1) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 16).
    if (!force_default && f == 4 && n == RADIX) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i].x; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i].y; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      return;
    }
#endif

    // Execute the original shufl code
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG + lowMe]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG + lowMe]; }
  }

  // If SHUFL_BYTES is 4 we split the T2 values into 4 int values.  These are written to LDS memory using four instructions.
  // NOT OPTIMIZED TO REDUCE LDS BANK CONFLICTS!!
  else if (SHUFL_BYTES == 4) {
    // Lower LDS requirements may let the optimizer use fewer VGPRs and increase occupancy for WIDTHs >= 1024.
    // Alas, the increased occupancy does not offset extra code needed for shufl_int (the assembly
    // code generated is not pretty).  This might not be true for nVidia or future ROCm optimizers.
    local int* lds = (local int*) lds2;
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(int);

    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).x; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG + lowMe]; u[i] = as_double2(tmp); }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).y; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG + lowMe]; u[i] = as_double2(tmp); }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).z; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG + lowMe]; u[i] = as_double2(tmp); }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = as_int4(u[i]).w; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG + lowMe]; u[i] = as_double2(tmp); }
  }
}

#endif


#if FFT_FP32 | NTT_GF31

// Shufl two or more fft_WIDTHs or FFT_HEIGHTs using two 4-byte floats.
void OVERLOAD shufl32(local F2 *lds2, F2 *u, u32 f, u32 numWG, u32 lowMe) {

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  //GW - would a 16 byte implementation be useful?  Less LDS conflict work?

  int force_default = 0;
#if NOWG2
  if (numWG > 1) force_default = 1;
#endif
#if NOLDS2
  if (f != 1) force_default = 1;
#endif

  // If SHUFL_BYTES is 8 or more we can write the complete F2 value to LDS memory with one instruction.
  if (SHUFL_BYTES >= 8) {
    local F2* lds = ((local F2*) lds2);
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F2);

#if LDSPAD
    // Special case first n == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...448, 1, 65..., 16, 80...   lds[64..127] = +2
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 448, 1, 65...   output[64..127] = +8
    // Pad one value after every row to eliminate bank conflicts.
    // gpuowl 6cf0dc first radix-8 padded shuffle.
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);

      for (u32 i = 0; i < RADIX; ++i)
        lds[i * (WG + 2) + lowMe] = u[i];

      bar(WG);

      for (u32 i = 0; i < RADIX; ++i)
        u[i] = lds[i * WG / 8
                 + (lowMe / 8)
                 + (lowMe & 7) * (WG + 2)];

      return;
    }

    // Special case second n == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Pad 8 values after every 64 values to eliminate bank conflicts.
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { lds[ (lowMe / 8)      * (WG + 8)                     + i * 8 + (lowMe & 7)] = u[i]; }
      else          for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 8) & 7) * (WG + 8) + (lowMe / 64) * 64 + i * 8 + (lowMe & 7)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * (WG + 8) + lowMe]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + lowMe / 64 * (WG + 8) + (lowMe & 63)]; }
      return;
    }

    // Special case first n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0, 64, ...192, 1.., 2.., 3.., 16...   lds[64..127] = +4
    // Read from LDS in the desired output order.  In the example:  output[0..63] = 0, 64, ... 192, 1, 65...   output[64..127] = +16
    // Pad one value after every row to eliminate bank conflicts.
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 1) + (lowMe / 16) * 16 + (lowMe & 3) * 4 + i] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 1) + (lowMe & 15)]; }
      return;
    }

    // Special case second n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS that does not use much padding and generates good code because all the lowMe calcs can be computed up front.
    // In the example:  lds[0..63] = 0...192, 16..., 32..., 48..., 4...   lds[64..127] = +1
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0...192, 16... 32.. 48.. 1...  lds[64..127] = +4
    // Pad 4 values after every row to eliminate bank conflicts.
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[((lowMe / 4) & 3) * (WG + 4) + (lowMe / 16) * 16 + i * 4 + (lowMe & 3)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 16                     +  (lowMe / 16)      * (WG + 4) + (lowMe & 15)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * 64 + (lowMe / 64) * 16 + ((lowMe / 16) & 3) * (WG + 4) + (lowMe & 15)]; }
      return;
    }
#endif

#if LDSSWIZ
    // Special case first n == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=512:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 1, 65...   lds[64..127] = +8
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 1 and 2 multiples of 8) and the first 26 threads read from LDS (multiples of 64 and two multiples of 1).
    if (!force_default && f == 1 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 8 + i) ^ (lowMe & 15)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ (((i & 1) * 8) + ((lowMe / 8) & 7))]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ (((lowMe / 8) & 15))]; }
      return;
    }

    // Special case second n == 8 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=512:  u[0] = 0, 64, ... 448, 1, 65...   u[1] = +8
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 448, 8, 72...   lds[64..127] = +1
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (8 multiples of 64 and 2 multiples of 1) and the first 16 threads read from LDS (8 multiples of 64 and two multiples of 8).
    if (!force_default && f == 8 && RADIX == 8) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 8 * 64 + i * 8 + (lowMe & 7)) ^ (lowMe & 8)] = u[i]; }
      bar(WG);
      if (WG == 64) for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((i & 1) * 8)]; }
      else          for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 8) & 8)]; }
      return;
    }

    // Special case first n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are in order.  For example, WIDTH=256:  u[0] = 0, 1, 2...  u[1] = +64...
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 1, 65...   lds[64..127] = +16
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 1 and 4 multiples of 4) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 1).
    if (!force_default && f == 1 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe * 4 + i) ^ (lowMe & 15)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 15)]; }
      return;
    }

    // Special case second n == 4 to eliminate LDS bank conflicts.  We're writing 8 bytes at a time, which means groups of 16 must have unique LDS banks.
    // Input values are the output from previous shufl.  For example, WIDTH=256:  u[0] = 0, 64, ... 192, 1, 65...   u[1] = +16
    // Output to LDS in the order we expect to read.  In the example:  lds[0..63] = 0, 64, ... 192, 16, 80 ...   lds[64..127] = +4
    // Swizzle LDS blocks to eliminate bank conflicts.
    // Swizzle on the first 16 threads written to LDS (4 multiples of 64 and 4 multiples of 1) and the first 16 threads read from LDS (4 multiples of 64 and 4 multiples of 16).
    if (!force_default && f == 4 && RADIX == 4) {
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { lds[(lowMe / 4 * 16 + i * 4 + (lowMe & 3)) ^ (lowMe & 12)] = u[i]; }
      bar(WG);
      for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[(i * WG + lowMe) ^ ((lowMe / 4) & 12)]; }
      return;
    }
#endif

    // Execute the original shufl code
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i] = lds[i * WG + lowMe]; }
  }

  // If SHUFL_BYTES is 4 we split the F2 values into 2 int values.  These are written to LDS memory using two instructions.
  // NOT OPTIMIZED TO REDUCE LDS BANK CONFLICTS!!
  else if (SHUFL_BYTES == 4) {
    // Accessing lds memory as ints might be faster than F2 accesses (halving LDS memory requirements)
    local F* lds = ((local F*) lds2);
    if (numWG > 1) lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F);

    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].x; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].x = lds[i * WG + lowMe]; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { lds[i * f + (lowMe & ~mask) * RADIX + (lowMe & mask)] = u[i].y; }
    bar(WG);
    for (u32 i = 0; i < RADIX; ++i) { u[i].y = lds[i * WG + lowMe]; }
  }
}

#endif




// AEVUM_GWOLT_SHUFL_FFT2_TYPED
// Typed Aevum backport of gpuowl 6cf0dc fused shuffle + fft2.
// Used only by SIZE=1K / RADIX=8 / f=8.

#if FFT_FP64

void OVERLOAD shufl_and_fft2(local T2 *lds2, T2 *u,
                             u32 f, u32 numWG, u32 lowMe) {
  assert(RADIX == 8);
  assert(f == 8);

  const u32 mask = f - 1;

#if SHUFL_BYTES == 16
  local T2 *lds = lds2;
  if (numWG > 1)
    lds += ((u32)get_local_id(0) / WG) * LDS_BYTES / sizeof(T2);

  bar(WG);
  for (u32 i = 0; i < RADIX; ++i)
    lds[i*f + (lowMe & ~mask)*RADIX + (lowMe & mask)] = u[i];

  bar(WG);
  for (u32 i = 0; i < RADIX; ++i) {
    T2 a = lds[i*(WG/2) + lowMe%(WG/2)];
    T2 b = lds[4*WG + i*(WG/2) + lowMe%(WG/2)];
    u[i] = lowMe < WG/2 ? a+b : a-b;
  }

#elif SHUFL_BYTES == 8
  local T *lds = (local T *)lds2;
  if (numWG > 1)
    lds += ((u32)get_local_id(0) / WG) * LDS_BYTES / sizeof(T);

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i].x;

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    T a=lds[i*(WG/2)+lowMe%(WG/2)];
    T b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i].x = lowMe < WG/2 ? a+b : a-b;
  }

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i].y;

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    T a=lds[i*(WG/2)+lowMe%(WG/2)];
    T b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i].y = lowMe < WG/2 ? a+b : a-b;
  }
#else
#error Unsupported FP64 SHUFL_BYTES
#endif
}

#endif


#if FFT_FP32

void OVERLOAD shufl_and_fft2(local F2 *lds2, F2 *u,
                             u32 f, u32 numWG, u32 lowMe) {
  assert(RADIX == 8);
  assert(f == 8);

  const u32 mask = f - 1;

#if SHUFL_BYTES >= 8
  local F2 *lds = lds2;
  if (numWG > 1)
    lds += ((u32)get_local_id(0) / WG) * LDS_BYTES / sizeof(F2);

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i];

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    F2 a=lds[i*(WG/2)+lowMe%(WG/2)];
    F2 b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i] = lowMe < WG/2 ? a+b : a-b;
  }
#else
#error Unsupported FP32 SHUFL_BYTES
#endif
}

#endif


#if NTT_GF31

void OVERLOAD shufl_and_fft2(local GF31 *lds2, GF31 *u,
                             u32 f, u32 numWG, u32 lowMe) {
  assert(RADIX == 8);
  assert(f == 8);

  const u32 mask = f - 1;

#if SHUFL_BYTES >= 8
  local GF31 *lds = lds2;
  if (numWG > 1)
    lds += ((u32)get_local_id(0) / WG) * LDS_BYTES / sizeof(GF31);

#if LDSPAD
  // AEVUM_GWOLT_1K_LDSPAD_FIX
  // gpuowl 6cf0dc: padded second radix-8 shuffle fused with fft2.
  if (f == 8 && RADIX == 8) {
    bar(WG);

    for (u32 i = 0; i < RADIX; ++i)
      lds[i * (WG + 8) + lowMe] = u[i];

    bar(WG);

    for (u32 i = 0; i < RADIX; ++i) {
      GF31 a =
        lds[(i / 2) * (WG / 64) * 8
          + (((i & 1) * (WG / 2)) / 64) * 8
          + ((lowMe % (WG / 2)) / 64) * 8
          + ((lowMe / 8) & 7) * (WG + 8)
          + (lowMe & 7)];

      GF31 b =
        lds[(i / 2 + 4) * (WG / 64) * 8
          + (((i & 1) * (WG / 2)) / 64) * 8
          + ((lowMe % (WG / 2)) / 64) * 8
          + ((lowMe / 8) & 7) * (WG + 8)
          + (lowMe & 7)];

      u[i] = lowMe < WG / 2 ? addq(a, b) : subq(a, b);
    }

    return;
  }
#endif

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i];

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    GF31 a=lds[i*(WG/2)+lowMe%(WG/2)];
    GF31 b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i] = lowMe < WG/2 ? addq(a,b) : subq(a,b);
  }
#else
#error Unsupported GF31 SHUFL_BYTES
#endif
}

#endif


#if NTT_GF61

void OVERLOAD shufl_and_fft2(local GF61 *lds2, GF61 *u,
                             u32 f, u32 numWG, u32 lowMe) {
  assert(RADIX == 8);
  assert(f == 8);

  const u32 mask = f - 1;

#if SHUFL_BYTES == 16
  local GF61 *lds = lds2;
  if (numWG > 1)
    lds += ((u32)get_local_id(0) / WG) * LDS_BYTES / sizeof(GF61);

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i];

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    GF61 a=lds[i*(WG/2)+lowMe%(WG/2)];
    GF61 b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i] = lowMe < WG/2 ? addq(a,b) : subq(a,b);
  }

#elif SHUFL_BYTES == 8
  local Z61 *lds = (local Z61 *)lds2;
  if (numWG > 1)
    lds += ((u32)get_local_id(0) / WG) * LDS_BYTES / sizeof(Z61);

#if LDSPAD
  // gpuowl 6cf0dc: padded second radix-8 shuffle fused with fft2.
  if (f == 8 && RADIX == 8) {
    bar(WG);

    for (u32 i = 0; i < RADIX; ++i)
      lds[i * (WG + 8) + lowMe] = u[i].x;

    bar(WG);

    for (u32 i = 0; i < RADIX; ++i) {
      Z61 a =
        lds[(i / 2) * (WG / 64) * 8
          + (((i & 1) * (WG / 2)) / 64) * 8
          + ((lowMe % (WG / 2)) / 64) * 8
          + ((lowMe / 8) & 7) * (WG + 8)
          + (lowMe & 7)];

      Z61 b =
        lds[(i / 2 + 4) * (WG / 64) * 8
          + (((i & 1) * (WG / 2)) / 64) * 8
          + ((lowMe % (WG / 2)) / 64) * 8
          + ((lowMe / 8) & 7) * (WG + 8)
          + (lowMe & 7)];

      u[i].x = lowMe < WG / 2 ? addq(a, b) : subq(a, b);
    }

    bar(WG);

    for (u32 i = 0; i < RADIX; ++i)
      lds[i * (WG + 8) + lowMe] = u[i].y;

    bar(WG);

    for (u32 i = 0; i < RADIX; ++i) {
      Z61 a =
        lds[(i / 2) * (WG / 64) * 8
          + (((i & 1) * (WG / 2)) / 64) * 8
          + ((lowMe % (WG / 2)) / 64) * 8
          + ((lowMe / 8) & 7) * (WG + 8)
          + (lowMe & 7)];

      Z61 b =
        lds[(i / 2 + 4) * (WG / 64) * 8
          + (((i & 1) * (WG / 2)) / 64) * 8
          + ((lowMe % (WG / 2)) / 64) * 8
          + ((lowMe / 8) & 7) * (WG + 8)
          + (lowMe & 7)];

      u[i].y = lowMe < WG / 2 ? addq(a, b) : subq(a, b);
    }

    return;
  }
#endif

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i].x;

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    Z61 a=lds[i*(WG/2)+lowMe%(WG/2)];
    Z61 b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i].x = lowMe < WG/2 ? addq(a,b) : subq(a,b);
  }

  bar(WG);
  for (u32 i=0; i<RADIX; ++i)
    lds[i*f+(lowMe&~mask)*RADIX+(lowMe&mask)] = u[i].y;

  bar(WG);
  for (u32 i=0; i<RADIX; ++i) {
    Z61 a=lds[i*(WG/2)+lowMe%(WG/2)];
    Z61 b=lds[4*WG+i*(WG/2)+lowMe%(WG/2)];
    u[i].y = lowMe < WG/2 ? addq(a,b) : subq(a,b);
  }
#else
#error Unsupported GF61 SHUFL_BYTES
#endif
}

#endif


#if FFT_FP64

void OVERLOAD shufl(local T2 *lds, T2 *u, u32 f, u32 numWG, u32 lowMe) {
  shufl64(lds, u, f, numWG, lowMe);
}

void OVERLOAD chainMul4(T2 *u, T2 w) {
  u[1] = cmul(u[1], w);

  T2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  double a = mul2(base.y);
  base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

#if 1
// This version of chainMul8 tries to minimize roundoff error even if more F64 ops are used.
// Trial and error looking at Z values on a WIDTH=512 FFT was used to determine when to switch from fancy to non-fancy powers of w.
void OVERLOAD chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

  T2 w2;
  // Rocm optimizer behaves weirdly. Using multiple mul2s instead of one mul2 in csqTrigFancy makes double-wide single-kernel variant 0 tailSquare inexplicably slower
  if (DOING_WIDTH || VARIANT != 0) {
    w2 = csqTrigFancy(w);
  } else {
    w2 = U2(mulminus2(w.y) * w.y, mul2(fma(w.x, w.y, w.y)));
  }
  u[2] = cmulFancy(u[2], w2);

  T2 w3;
  // Rocm optimizer behaves weirdly yet again. Using mul2 instead of 2.0* makes double-wide single-kernel tailSquare inexplicably slower
  // even though it is one fewer F64 op.
  if (DOING_WIDTH || VARIANT != 0) {
    w3 = ccubeTrigFancy(w2, w);
  } else {
    double a = 2*w2.y;
    w3 = U2(fma(a, -w.y, w.x), fma(a, w.x, a - w.y));
  }
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  T2 base = cmulFancy(w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

#else
// This version of chainMul8 minimizes F64 ops even if that increases roundoff error.
// This version is faster on a Radeon 7 with worse roundoff.  However, FFT_width is even faster with better roundoff.
// This version is the same speed on a TitanV probably due to its great F64 throughput.
// This version is slower on R7Pro due to a rocm optimizer issue in double-wide single-kernel tailSquare using BCAST.  I could not find a work-around.
// Other GPUs???  This version might be useful.  If we decide to make this available, it will need a new width and height fft spec number.
// Consequently, an increase in the BPW table and increase work for -ztune and -tune.
void OVERLOAD chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

  T2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  T2 w3 = ccubeTrigDefancy(w2, w);
  u[3] = cmul(u[3], w3);

  T2 w4 = csqTrigDefancy(w2);
  u[4] = cmul(u[4], w4);

  T2 w6 = csqTrig(w3);
  T2 w5, w7; cmul_a_by_fancyb_and_conjfancyb(&w7, &w5, w6, w);
  u[5] = cmul(u[5], w5);
  u[6] = cmul(u[6], w6);
  u[7] = cmul(u[7], w7);
}
#endif

void OVERLOAD chainMul(T2 *u, T2 w) {
  // Do a length 4 chain mul, w must not be in Fancy format
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul, w must be in Fancy format
  if (RADIX == 8) chainMul8(u, w);
}


#if AMDGPU && VARIANT == 0

int bcast4(int x)  { return __builtin_amdgcn_mov_dpp(x, 0, 0xf, 0xf, false); }
int bcast8(int x)  { return __builtin_amdgcn_ds_swizzle(x, 0x0018); }
int bcast16(int x) { return __builtin_amdgcn_ds_swizzle(x, 0x0010); }
int bcast64(int x) { return __builtin_amdgcn_readfirstlane(x); }

int bcastAux(int x, u32 span) {
  return span == 4 ? bcast4(x) : span == 8 ? bcast8(x) : span == 16 ? bcast16(x) : span == 64 ? bcast64(x) : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  for (int i = 0; i < 4; ++i) { s[i] = bcastAux(s[i], span); }
  return as_double2(s);
}

#endif

void OVERLOAD fft_RADIX(T2 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 5
  fft5(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD tabMul(Trig trig, T2 *u, u32 f, u32 me) {
#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

// Compute trigs from scratch every time.  This can't possibly be a good idea on any GPUs.
#if 0
  T2 w = slowTrig_N(ND / RADIX / WG * p, ND / RADIX);
  T2 base = w;
  for (int i = 1; i < RADIX; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
  return;
#endif

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at these short n=4 and n=8 lengths are very accurate.

  if (TABMUL_CHAIN) {
    T2 w = TFLOAD(&trig[p]);
    chainMul(u, w);
    return;
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN) {
    T2 w = TFLOAD(&trig[p]);

    if (RADIX >= 8) {
      u[1] = cmulFancy(u[1], w);
    } else {
      u[1] = cmul(u[1], w);
    }

    for (u32 i = 2; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(Trig trig, T2 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN) {
      T2 w  = TFLOAD(&trig[p]);
      T2 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      T2 base  = csqTrig(w);
      T2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

    if (!TABMUL_CHAIN) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN) {
      T2 w  = TFLOAD(&trig[p]);
      T2 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      T2 base  = csqTrig(w);
      T2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

    if (!TABMUL_CHAIN) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(Trig trig, T2 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at n=8 lengths are very accurate.

  if (TABMUL_CHAIN) {
    u32 p = me & ~(f - 1);
    T2 w = TFLOAD(&trig[p]);

//    u[1] = cmulFancy(u[1], w);				// GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
//    T2 w2 = csqTrigFancy(w);
//    u[2] = cmulFancy(u[2], w2);
//    T2 w3 = ccubeTrigFancy(w2, w);
//    u[3] = cmulFancy(u[3], w3);
//    w3.x += 1;
//    T2 base = cmulFancy(w3, w);
//    for (int i = 4; i < 8; ++i) {
//      u[i] = cmul(u[i], base);
//      base = cmulFancy(base, w);
//    }

    u[1] = cmul(u[1], w);				// GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
    T2 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    T2 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    T2 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

//GW:  Can any of these be Fancy? Yes, u[1] and u[2]
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { T2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(Trig trig, T *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  Trig trig2 = (Trig) trig1;
  T2 sine_over_cosines = TFLOAD(&trig2[me]);
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = TFLOAD(&trig1[2*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 4; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = TFLOAD(&trig1[3*WG + me]);             // Sine/cosines for later tabMuls
    preloads[5] = TFLOAD(&trig1[4*WG + 4*WG + me]);      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(Trig trig, T *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    Trig trig2 = (Trig) (trig1 + (i-1)*WG);
    T2 sine_over_cosines = TFLOAD(&trig2[me]);
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = TFLOAD(&trig1[6*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 8; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
    T cosine1_SQRT1_2 = preloads[1] * M_SQRT1_2;
    X2_via_FMA(u[0], preloads[1], u[1]);
    X2_via_FMA(u[2], preloads[1], u[3]);
    X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
    X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

  } else {

    // Apply cosine to u[1]
    u[1] = u[1] * preloads[1];

    // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8
    X2(u[0], u[1]);
    X2(u[2], u[3]);
    X2ad(u[4], u[5], M_SQRT1_2);
    X2ad(u[6], u[7], M_SQRT1_2);
  }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}


void OVERLOAD fft_common(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, u32 lowMe, int callnum) {

  // This line mimics shufl -- partition lds for variant 2
  local T2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T2);

// Variant 0 uses broadcast instructions.  Only available on AMD GPUs.

#if VARIANT == 0

#if WG * RADIX > 1024
#error VARIANT == 0 only supported for FFT size <= 1024
#endif
#if !AMDGPU
#error VARIANT == 0 only supported by AMD GPUs
#endif

// There is a slight difference between fft_WIDTH and fft_HEIGHT.  Tail square computes the trig values
// to be broadcast, while fft_WIDTH does not.  Compute the trig values now for fft_WIDTH,
#if DOING_WIDTH
#if RADIX == 8
  w = fancyTrig_N(ND / (WG * RADIX) * lowMe);
#else
  w = slowTrig_N(ND / (WG * RADIX) * lowMe, ND / RADIX);
#endif
#endif

  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    w = bcast(w, s);
    chainMul(u, w);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

// Variant 2 uses more FMA instructions than the original FFT code.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

// Variant 2 code for SIZE=256, RADIX=4
#elif WG == 64 && RADIX == 4 && VARIANT == 2

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);

// Variant 2 code for SIZE=512, RADIX=8
#elif WG == 64 && RADIX == 8 && VARIANT == 2

  T preloads[10];                       // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + SAVE_ONE_MUL*2*WG*8;   // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, SAVE_ONE_MUL);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, SAVE_ONE_MUL);  // We'd rather set save_one_more_mul to 1

// Variant 2 code for SIZE=1024, RADIX=4
#elif WG == 256 && RADIX == 4 && VARIANT == 2

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 64, numWG, lowMe, 1);

// Custom code for SIZE=4K, RADIX=8
#elif WG == 512 && RADIX == 8 && VARIANT == 2

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1


// Custom code for SIZE=256, RADIX=8, threads=32.  Performed as 4 * 8 * 8.  Radix-8 allows fewer
// shufls and tabmuls than radix-4.  Fewer instructions, but more registers.
// Uses only 32 threads which is fine on nVidia, lousy on radeon VII (use WMUL=2, TAIL_KERNELS=2).
//
// Details for memory layout, trig data, and shufls:
// Mem:   0     1...  31
//        32
//        ...
//        196
//        224   ...  255
// Only do a radix-4 fft.  Non-standard TABMUL:    (64 3/4 cmuls in blocks of 1 duplicated trig values)
// trig powers are: 0*0 0*1 .. 0*31
//                  0*32    .. 0*63
//                  1*0 1*1 .. 1*31
//                  1*32    .. 1*63
//                  2*0 2*1 .. 2*31
//                  2*32    .. 2*63
//                  3*0 3*1 .. 3*31
//                  3*32    .. 3*63           total trig data (6*32*16=3KB)
// non-standard shufl out:
//        0  64 .. 192   1... 7...
//        8
//        16
//        ...
//        48
//        56
// standard TABMUL:   (8 7/8 cmuls in blocks of 4 duplicated trig values)
// trig powers are: 0000 0000 .. 0000*7
//                  0000 4444 .. 4444*7
//                  0000 8888 .. 8888*7
//                  0000 12 ...
//                  0000 16 ...
//                  0000 20 ...
//                  0000 24 ...
//                  0000 28 ...           total trig data (7*8*16=896B)
// standard shufl out:
//        0  64 .. 192   8...  56...
//        1
//        2
//        ...
//        6
//        7
//
// FP64 and FP32 could benefit by starting the next fft8 after the radix-4 tabmul (easy FMA opportunities).

// Code for SIZE=256, RADIX=8
#elif WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Custom code for SIZE=1024, RADIX=8, threads=128.  Performed as 8 * 8 * 2 * 8.  Radix-8 allows fewer
// shufls and tabmuls than radix-4.  Fewer instructions, but more registers.  If we process 4 (or 8)
// independent width lines then we should be able to avoid the mul by w^0 in the next to last radix-8 step.
//
// Details for memory layout, trig data, and shufls:
// Mem:   0     1 ...  127
//        128
//        256
//        384
//        512
//        640
//        768
//        896   ...  1023
// standard TABMUL:    (128 7/8 cmuls in blocks of 1 duplicated trig values)
// trig powers are: 0 0 0  .. 0*127
//                  0 1 2  .. 1*127
//                  0 2 4  .. 2*127
//                  0 3 6  .. 3*127
//                  0 4 8  .. 4*127
//                  0 5 10 .. 5*127
//                  0 6 12 .. 6*127
//                  0 7 14 .. 7*127           total trig data (7*128*16=14KB)
// standard shufl out:
//        0   128 .. 896   1... 15...
//        16
//        32
//        48
//        64
//        80
//        96
//        112
// standard TABMUL:   (16 7/8 cmuls in blocks of 8 duplicated trig values)
// trig powers are: 00000000 00000000 .. 00000000*15
//                  00000000 11111111 .. 11111111*15
//                  00000000 22222222 .. 22222222*15
//                  00000000 33333333 .. 33333333*15
//                  00000000 44444444 .. 44444444*15
//                  00000000 55555555 .. 55555555*15
//                  00000000 66666666 .. 66666666*15
//                  00000000 77777777 .. 77777777*15           total trig data (7*16*16=1.75KB)
// shufl out with an fft2:
//        0  128 .. 896  16 .. 112...  8...
//        1
//        2
//        3
//        4
//        5
//        6
//        7

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

  // Old / original version

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD shufl(local F2 *lds, F2 *u, u32 f, u32 numWG, u32 lowMe) {
  shufl32(lds, u, f, numWG, lowMe);
}

void OVERLOAD fft_RADIX(F2 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(F2 *u, F2 w) {
  u[1] = cmul(u[1], w);

  F2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  F a = mul2(base.y);
  base = (F2)(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(F2 *u, F2 w) {
  u[1] = cmulFancy(u[1], w);
                                                  //GWBUG - see FP64 version for many possible optimizations
  F2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  F2 w3 = ccubeTrigFancy(w2, w);
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  F2 base = cmulFancy(w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

void OVERLOAD chainMul(F2 *u, F2 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigFP32 trig, F2 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN32) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN32) {
    if (RADIX >= 8) {
      u[1] = cmulFancy(u[1], TFLOAD(&trig[p]));
    } else {
      u[1] = cmul(u[1], TFLOAD(&trig[p]));
    }
    for (u32 i = 2; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(TrigFP32 trig, F2 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great SP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN32) {
      F2 w  = TFLOAD(&trig[p]);
      F2 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      F2 base  = csqTrig(w);
      F2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN32) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great SP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN32) {
      F2 w  = TFLOAD(&trig[p]);
      F2 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      F2 base  = csqTrig(w);
      F2 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN32) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(TrigFP32 trig, F2 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great SP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at n=8 lengths are very accurate.

  if (TABMUL_CHAIN32) {
    u32 p = me & ~(f - 1);
    F2 w = TFLOAD(&trig[p]);

//    u[1] = cmulFancy(u[1], w);				// GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
//    T2 w2 = csqTrigFancy(w);
//    u[2] = cmulFancy(u[2], w2);
//    T2 w3 = ccubeTrigFancy(w2, w);
//    u[3] = cmulFancy(u[3], w3);
//    w3.x += 1;
//    T2 base = cmulFancy(w3, w);
//    for (int i = 4; i < 8; ++i) {
//      u[i] = cmul(u[i], base);
//      base = cmulFancy(base, w);
//    }

    u[1] = cmul(u[1], w);				// GW: - this should use Fancy, but tabmul8_4a does not and it could for half of the data
    F2 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    F2 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    F2 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN32) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

//GW:  Can any of these be Fancy? Yes, u[1] and u[2]
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Some OpenCL compilers are having trouble with fma on floats.  Specifically, line "X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);".
// Since we're not enabling FP32 variant 2 by default, don't include these "more FMA" routines.

#if ENABLE_FP32_VARIANT_2

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { F2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(TrigFP32 trig, F *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  TrigFP32 trig2 = (TrigFP32) trig1;
  F2 sine_over_cosines = TFLOAD(&trig2[me]);
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = TFLOAD(&trig1[2*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(local F2 *lds, TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me) {
  local F *lds1 = (local F *) lds;
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 4; ++i) {
    F sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      TrigFP32 trig2 = (TrigFP32) (trig1 + i*WG);
      F2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = TFLOAD(&trig1[3*WG + me]);             // Sine/cosines for later tabMuls
    preloads[5] = TFLOAD(&trig1[4*WG + 4*WG + me]);      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(TrigFP32 trig, F *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    TrigFP32 trig2 = (TrigFP32) (trig1 + (i-1)*WG);
    F2 sine_over_cosines = TFLOAD(&trig2[me]);
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = TFLOAD(&trig1[6*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(local F2 *lds, TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me) {
  local F *lds1 = (local F *) lds;
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 8; ++i) {
    F sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      TrigFP32 trig2 = (TrigFP32) (trig1 + i*WG);
      F2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
    F cosine1_SQRT1_2 = preloads[1] * (float) M_SQRT1_2;
    X2_via_FMA(u[0], preloads[1], u[1]);
    X2_via_FMA(u[2], preloads[1], u[3]);
    X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
    X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

  } else {

    // Apply cosine to u[1]
    u[1] = u[1] * preloads[1];

    // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8
    X2(u[0], u[1]);
    X2(u[2], u[3]);
    X2ad(u[4], u[5], M_SQRT1_2);
    X2ad(u[6], u[7], M_SQRT1_2);
  }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

// Variant 2 code uses more FMA instructions than the original fft version.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

void OVERLOAD fft_common(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe, int callnum) {

  // This line mimics shufl -- partition lds
  local F2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F2);

// Variant 2 code for SIZE=256, RADIX=4
#if ENABLE_FP32_VARIANT_2 && WG == 64 && RADIX == 4 && VARIANT == 2

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);

// Variant 2 code for SIZE=512, RADIX=8
#elif ENABLE_FP32_VARIANT_2 && WG == 64 && RADIX == 8 && VARIANT == 2

  F preloads[10];                        // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + SAVE_ONE_MUL*2*WG*8;    // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, SAVE_ONE_MUL);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, SAVE_ONE_MUL);

// Variant 2 code for SIZE=1024, RADIX=4
#elif ENABLE_FP32_VARIANT_2 && WG == 256 && RADIX == 4 && VARIANT == 2

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 64, numWG, lowMe, 1);

// Variant 2 code for SIZE=4K, RADIX=8
#elif ENABLE_FP32_VARIANT_2 && WG == 512 && RADIX == 8 && VARIANT == 2

  F preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

// Code for SIZE=256, RADIX=8
#elif WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

  // Old / original version

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD shufl(local GF31 *lds, GF31 *u, u32 f, u32 numWG, u32 lowMe) {
  shufl32((local F2 *) lds, (F2 *) u, f, numWG, lowMe);
}

void OVERLOAD fft_RADIX(GF31 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(GF31 *u, GF31 w) {
  u[1] = cmul(u[1], w);

  GF31 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(GF31 *u, GF31 w) {
  u[1] = cmul(u[1], w);

  GF31 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmul(base, w);
  }
}

void OVERLOAD chainMul(GF31 *u, GF31 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigGF31 trig, GF31 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN31) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN31) {
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(TrigGF31 trig, GF31 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN31) {
      GF31 w  = TFLOAD(&trig[p]);
      GF31 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF31 base  = csqTrig(w);
      GF31 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN31) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN31) {
      GF31 w  = TFLOAD(&trig[p]);
      GF31 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF31 base  = csqTrig(w);
      GF31 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN31) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(TrigGF31 trig, GF31 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN31) {
    u32 p = me & ~(f - 1);
    GF31 w = TFLOAD(&trig[p]);

    u[1] = cmul(u[1], w);
    GF31 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    GF31 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    GF31 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN31) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

void OVERLOAD fft_common(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) {

// Code for SIZE=256, RADIX=8
#if WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif

}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD shufl(local GF61 *lds, GF61 *u, u32 f, u32 numWG, u32 lowMe) {
  shufl64((local T2 *) lds, (T2 *) u, f, numWG, lowMe);
}

void OVERLOAD fft_RADIX(GF61 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(GF61 *u, GF61 w) {
  u[1] = cmul(u[1], w);

  GF61 base = csq(w);
  u[2] = cmul(u[2], base);

  base = cmul(base, w);                 //GWBUG - see FP64 version for possible optimization
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(GF61 *u, GF61 w) {
  u[1] = cmul(u[1], w);

  GF61 w2 = csq(w);
  u[2] = cmul(u[2], w2);

  GF61 base = cmul(w2, w);              //GWBUG - see FP64 version for many possible optimizations
  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmul(base, w);
  }
}

void OVERLOAD chainMul(GF61 *u, GF61 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigGF61 trig, GF61 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN61) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN61) {
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

// Tabmul after doing an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4a(TrigGF61 trig, GF61 *u, u32 f, u32 me) {

  if (f == 1) {                      // fft8_4 is performed first
    u32 p = me;

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    if (TABMUL_CHAIN61) {
      GF61 w  = TFLOAD(&trig[p]);
      GF61 w2 = TFLOAD(&trig[WG + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF61 base  = csqTrig(w);
      GF61 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN61) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG + p]));
      }
    }
  }

  else {                      // fft8_4 is performed after an initial fft8

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.
// Perform two length=4 chain muls.

    u32 p = me / 8;                 // Generate index into condensed trig table that does not have duplicated trig values
    trig += 7 * WG;                 // Skip over the trig values used in the first tabmul
    if (TABMUL_CHAIN61) {
      GF61 w  = TFLOAD(&trig[p]);
      GF61 w2 = TFLOAD(&trig[WG/8 + p]);
      u[2] = cmul(u[2], w);
      u[3] = cmul(u[3], w2);
      GF61 base  = csqTrig(w);
      GF61 base2 = csqTrig(w2);
      u[4] = cmul(u[4], base);
      u[5] = cmul(u[5], base2);
      base  = ccubeTrig(base, w);
      base2 = ccubeTrig(base2, w2);
      u[6] = cmul(u[6], base);
      u[7] = cmul(u[7], base2);
    }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

    if (!TABMUL_CHAIN61) {
      for (u32 i = 2; i < RADIX; ++i) {
        u[i] = cmul(u[i], TFLOAD(&trig[(i-2)*WG/8 + p]));
      }
    }
  }
}

// Later tabmuls after starting with an fft4 when RADIX=8.  See the SIZE=256 code for example memory and trig layout.
void OVERLOAD tabMul8_4b(TrigGF61 trig, GF61 *u, u32 f, u32 me) {

// This code uses chained complex multiplies which could be faster on GPUs with great throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN61) {
    u32 p = me & ~(f - 1);
    GF61 w = TFLOAD(&trig[p]);

    u[1] = cmul(u[1], w);
    GF61 w2 = csqTrig(w);
    u[2] = cmul(u[2], w2);
    GF61 w3 = ccubeTrig(w2, w);
    u[3] = cmul(u[3], w3);
    GF61 base = cmul(w3, w);
    for (int i = 4; i < 8; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN61) {
    u32 p = (me/4) & ~(f/4 - 1);     // Generate index into condensed trig table that does not have duplicated trig values
    trig += 6 * WG;                  // Skip over the trig values used in tabmul8_4a

    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*(WG/4) + p]));
    }
  }
}

void OVERLOAD fft_common(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) {

// Code for SIZE=256, RADIX=8
#if WG == 32 && NW == 8

  fft8_4(u);
  tabMul8_4a(trig, u, 1, lowMe);
  shufl(lds, u, 1, 4, numWG, lowMe);

  fft8(u);
  tabMul8_4b(trig, u, 4, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  fft8(u);

// Code for SIZE=1024, RADIX=8
#elif WG == 128 && RADIX == 8

  fft8(u);
  tabMul(trig, u, 1, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  fft8(u);
  tabMul(trig, u, 8, lowMe);
  shufl_and_fft2(lds, u, 8, numWG, lowMe);

  if (lowMe < WG / 2) fft8_16a(u); else fft8_16b(u);

#else

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif

}

#endif
