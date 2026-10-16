// Copyright (C) Mihai Preda

#include "math.cl"
#include "trig.cl"

#ifndef AEVUM_PFA_GF31_COSBASE
#define AEVUM_PFA_GF31_COSBASE 0
#endif
#if AEVUM_PFA_GF31_COSBASE != 0 && AEVUM_PFA_GF31_COSBASE != 1
#error AEVUM_PFA_GF31_COSBASE must be 0 or 1
#endif
#if AEVUM_PFA_GF31_COSBASE && PFA_RADIX != 7 && PFA_RADIX != 9
#error AEVUM_PFA_GF31_COSBASE requires PFA7 or PFA9
#endif

#ifndef AEVUM_PFA7_NUSS8
#define AEVUM_PFA7_NUSS8 0
#endif
#if AEVUM_PFA7_NUSS8 != 0 && AEVUM_PFA7_NUSS8 != 1
#error AEVUM_PFA7_NUSS8 must be 0 or 1
#endif
#if AEVUM_PFA7_NUSS8 && PFA_RADIX != 7
#error AEVUM_PFA7_NUSS8 requires PFA7
#endif


#if MIDDLE == 3
#include "fft3.cl"
#elif MIDDLE == 4
#include "fft4.cl"
#elif MIDDLE == 5
#include "fft5.cl"
#elif MIDDLE == 6
#include "fft6.cl"
#elif MIDDLE == 7
#include "fft7.cl"
#elif MIDDLE == 8
#include "fft8.cl"
#elif MIDDLE == 9
#include "fft9.cl"
#elif MIDDLE == 10
#include "fft10.cl"
#elif MIDDLE == 11
#include "fft11.cl"
#elif MIDDLE == 12
#include "fft12.cl"
#elif MIDDLE == 13
#include "fft13.cl"
#elif MIDDLE == 14
#include "fft14.cl"
#elif MIDDLE == 15
#include "fft15.cl"
#elif MIDDLE == 16
#include "fft16.cl"
#endif

#if !defined(MM_CHAIN) && !defined(MM2_CHAIN) && FFT_VARIANT_M == 0
#define MM_CHAIN 0
#define MM2_CHAIN 0
#endif

#if !defined(MM_CHAIN) && !defined(MM2_CHAIN) && FFT_VARIANT_M == 1
#define MM_CHAIN 1
#define MM2_CHAIN 2
#endif

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.

#define WADD(i, w) u[i] = cmul(u[i], w)
#define WSUB(i, w) u[i] = cmul_by_conjugate(u[i], w)
#define WADDF(i, w) u[i] = cmulFancy(u[i], w)
#define WSUBF(i, w) u[i] = cmulFancy(u[i], conjugate(w))

#if FFT_FP64

void OVERLOAD fft2(T2* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(T2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 3
  fft3(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 5
  fft5(u);
#elif MIDDLE == 6
  fft6(u);
#elif MIDDLE == 7
  fft7(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE == 10
  fft10(u);
#elif MIDDLE == 11
  fft11(u);
#elif MIDDLE == 12
  fft12(u);
#elif MIDDLE == 13
  fft13(u);
#elif MIDDLE == 14
  fft14(u);
#elif MIDDLE == 15
  fft15(u);
#elif MIDDLE == 16
  fft16(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

// Keep in sync with TrigBufCache.cpp, see comment there.
#define SHARP_MIDDLE 5

void OVERLOAD middleMul(T2 *u, u32 s, Trig trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

  if (WIDTH == SMALL_HEIGHT) trig += SMALL_HEIGHT;     // In this case we can share the MiddleMul2 trig table.  Skip over the MiddleMul trig table.
  T2 w = TFLOAD(&trig[s]);           // s / BIG_HEIGHT

  if (MIDDLE < SHARP_MIDDLE) {
    WADD(1, w);
#if MM_CHAIN == 0
    T2 base = csqTrig(w);
    for (u32 k = 2; k < MIDDLE; ++k) {
      WADD(k, base);
      base = cmul(base, w);
    }
#elif MM_CHAIN == 1
    for (u32 k = 2; k < MIDDLE; ++k) { WADD(k, slowTrig_N(WIDTH * k * s, WIDTH * k * SMALL_HEIGHT)); }
#else
#error MM_CHAIN must be 0 or 1
#endif

  } else { // MIDDLE >= 5

#if MM_CHAIN == 0
    WADDF(1, w);
    T2 base;
    base = csqTrigFancy(w);
    WADDF(2, base);
    base = ccubeTrigFancy(base, w);
    WADDF(3, base);
    base.x += 1;

    for (u32 k = 4; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif 0 && MM_CHAIN == 1        // This is fewer F64 ops, but may be slower on Radeon 7 -- probably the optimizer being weird.  It also has somewhat worse Z.
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      T2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    T2 w2;
    if ((MIDDLE - 2) % 3 > 0) {
      w2 = csqTrigFancy(w);
      WADDF(2, w2);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      T2 w3 = ccubeTrigFancy(w2, w);
      WADDF(3, w3);
    }

#elif MM_CHAIN == 1
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      T2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    if ((MIDDLE - 2) % 3 > 0) {
      WADDF(2, w);
      WADDF(2, w);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      WADDF(3, w);
      WADDF(3, csqTrigFancy(w));
    }
#else
#error MM_CHAIN must be 0 or 1.
#endif
  }
}

void OVERLOAD middleMul2(T2 *u, u32 x, u32 y, double factor, Trig trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  if (MIDDLE == 1) {
    WADD(0, slowTrig_N(x * y, ND) * factor);
    return;
  }

  trig += SMALL_HEIGHT;          // Skip over the MiddleMul trig table
  T2 w = TFLOAD(&trig[x]);       // x / (MIDDLE * WIDTH)

  if (MIDDLE < SHARP_MIDDLE) {
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, base); }
    WSUB(0, w);
    if (MIDDLE > 2) { WADD(2, w); }
    if (MIDDLE > 3) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);

#if 0                                   // Slower on Radeon 7, but proves the concept for use in GF61.  Might be worthwhile on poor FP64 GPUs

    Trig trig2 = trig + WIDTH;          // Skip over the fist MiddleMul2 trig table
    u32 desired_root = x * y;
    T2 base = cmulFancy(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT])) * factor;   //Optimization to do: put multiply by factor in trig2 table

    WADD(0, base);
    for (u32 k = 1; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif AMDGPU && MM2_CHAIN == 0          // Oddly, Radeon 7 is faster with this version that uses more F64 ops

    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    WADD(0, base);
    WADD(1, base);

    for (u32 k = 2; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }
    WSUBF(0, w);

#elif MM2_CHAIN == 0

    u32 mid = MIDDLE / 2;
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
    WADD(mid, base);

    T2 basehi, baselo;
    cmul_a_by_fancyb_and_conjfancyb(&basehi, &baselo, base, w);
    WADD(mid-1, baselo);
    WADD(mid+1, basehi);

    for (int i = mid-2; i >= 0; --i) {
      baselo = cmulFancy(baselo, conjugate(w));
      WADD(i, baselo);
    }

    for (int i = mid+2; i < MIDDLE; ++i) {
      basehi = cmulFancy(basehi, w);
      WADD(i, basehi);
    }

#elif MM2_CHAIN == 1
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      T2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
      WADD(mid, base1);

      T2 base2 = base1;
      for (u32 i = 1; i <= n; ++i) {
        base1 = cmulFancy(base1, conjugate(w));
        WADD(mid - i, base1);

        base2 = cmulFancy(base2, w);
        WADD(mid + i, base2);
      }
      if (!(sz & 1)) {
        base2 = cmulFancy(base2, w);
        WADD(mid + n + 1, base2);
      }
    }

#elif MM2_CHAIN == 2
    T2 base, base_minus1, base_plus1;
    for (u32 i = 1; ; i += 3) {
      if (i-1 == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        WADD(i-1, base_minus1);
        break;
      } else if (i == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        base = cmulFancy(base_minus1, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        break;
      } else {
        base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor;
        cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        WADD(i+1, base_plus1);
        if (i+1 == MIDDLE-1) break;
      }
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    local T *p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local T *p2 = lds + me;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  } else {
    local int *p1 = ((local int*) lds) + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local int *p2 = (local int*) lds + me;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[workgroupSize * i]; }
  }
}

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global T2 *out, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;

  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

#if PFA_RADIX
// FP32 analogue of pfaMiddleTwiddle(GF31/GF61) below: the WIDTH x SMALL_HEIGHT
// Cooley-Tukey twiddle w_(W*SH)^(x*y) that the binary axis of a Good-Thomas
// plan still needs between its width pass and its height pass.  The scalar is
// the same for every odd-row register, so it commutes with the radix-R
// transform, and the inverse direction reuses the forward table because the
// inverse binary passes run on component-swapped data (stock middleMul2 does
// the same).
// PFA FP32 middle trig layout (TrigBufCache genMiddleTrigFP32Pfa(smallH, width)):
//   trig[0 .. WIDTH)                   : w_WIDTH^k
//   trig[WIDTH .. WIDTH + SMALL_HEIGHT) : w_(WIDTH*SMALL_HEIGHT)^k
void OVERLOAD pfaMiddleTwiddle(F2 *u, u32 x, u32 y, TrigFP32 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  const u32 desired_root = x * y;
  const F2 w = cmul(TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT]));
  for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, w); }
}
#endif

void OVERLOAD fft2(F2* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(F2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
#elif PFA_RADIX == 7 && MIDDLE == 7
  fft7(u);
#elif PFA_RADIX == 9 && MIDDLE == 9
  fft9(u);
#elif MIDDLE == 16
  fft16(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

void OVERLOAD ifft_MIDDLE(F2 *u) {
#if PFA_RADIX == 7
  // The tail hands back SWAP_XY(result), so as with PFA9 the same forward
  // odd-axis DFT applied to swapped data is the inverse transform.  Only the
  // radix normalization is required here.
  fft7(u);
#pragma unroll
  for (u32 i = 0; i < 7; ++i) u[i] *= (1.0f / 7.0f);
#elif PFA_RADIX == 9
  // The tail hands back SWAP_XY(result) = i*conj(result), so, exactly as in the
  // stock power-of-two path, the FORWARD transform applied to the swapped data
  // is the inverse transform of the unswapped data.  An explicit inverse DFT
  // (conjugate, fft9, conjugate) here would apply a second forward 9-point DFT
  // to the real data and negate the odd-axis index (row k lands in row 9-k).
  // Only the 1/9 normalisation of the odd axis is needed; width and height are
  // scaled by fftMiddleOut's factor.  (The GF31/GF61 planes legitimately use
  // explicit inverse roots because their 9th roots lie in the base field and
  // commute with conjugation.)
  fft9(u);
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] *= (1.0f / 9.0f);
#else
  fft_MIDDLE(u);
#endif
}

// Keep in sync with TrigBufCache.cpp, see comment there.
#define SHARP_MIDDLE 5

void OVERLOAD middleMul(F2 *u, u32 s, TrigFP32 trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

  if (WIDTH == SMALL_HEIGHT) trig += SMALL_HEIGHT;     // In this case we can share the MiddleMul2 trig table.  Skip over the MiddleMul trig table.
  F2 w = TFLOAD(&trig[s]);           // s / BIG_HEIGHT

  if (MIDDLE < SHARP_MIDDLE) {
    WADD(1, w);
#if MM_CHAIN == 0
    F2 base = csqTrig(w);
    for (u32 k = 2; k < MIDDLE; ++k) {
      WADD(k, base);
      base = cmul(base, w);
    }
#elif MM_CHAIN == 1
    for (u32 k = 2; k < MIDDLE; ++k) { WADD(k, slowTrig_N(WIDTH * k * s, WIDTH * k * SMALL_HEIGHT)); }
#else
#error MM_CHAIN must be 0 or 1
#endif

  } else { // MIDDLE >= 5

#if MM_CHAIN == 0
    WADDF(1, w);
    F2 base;
    base = csqTrigFancy(w);
    WADDF(2, base);
    base = ccubeTrigFancy(base, w);
    WADDF(3, base);
    base.x += 1;

    for (u32 k = 4; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif 0 && MM_CHAIN == 1        // This is fewer F64 ops, but may be slower on Radeon 7 -- probably the optimizer being weird.  It also has somewhat worse Z.
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      F2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    F2 w2;
    if ((MIDDLE - 2) % 3 > 0) {
      w2 = csqTrigFancy(w);
      WADDF(2, w2);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      F2 w3 = ccubeTrigFancy(w2, w);
      WADDF(3, w3);
    }

#elif MM_CHAIN == 1
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      F2 base, base_minus1, base_plus1;
      base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k);
      cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
      WADD(k-1, base_minus1);
      WADD(k,   base);
      WADD(k+1, base_plus1);
    }

    WADDF(1, w);

    if ((MIDDLE - 2) % 3 > 0) {
      WADDF(2, w);
      WADDF(2, w);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      WADDF(3, w);
      WADDF(3, csqTrigFancy(w));
    }
#else
#error MM_CHAIN must be 0 or 1.
#endif
  }
}

void OVERLOAD middleMul2(F2 *u, u32 x, u32 y, float factor, TrigFP32 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  if (MIDDLE == 1) {
    WADD(0, slowTrig_N(x * y, ND) * factor);
    return;
  }

  trig += SMALL_HEIGHT;     // Skip over the MiddleMul trig table
  F2 w = TFLOAD(&trig[x]);           // x / (MIDDLE * WIDTH)

  if (MIDDLE < SHARP_MIDDLE) {
    F2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, base); }
    WSUB(0, w);
    if (MIDDLE > 2) { WADD(2, w); }
    if (MIDDLE > 3) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // F2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);

#if 0                                   // Slower on Radeon 7, but proves the concept for use in GF61.  Might be worthwhile on poor FP64 GPUs

    TrigFP32 trig2 = trig + WIDTH;          // Skip over the fist MiddleMul2 trig table
    u32 desired_root = x * y;
    F2 base = cmulFancy(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT])) * factor;   //Optimization to do: put multiply by factor in trig2 table

    WADD(0, base);
    for (u32 k = 1; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }

#elif AMDGPU && MM2_CHAIN == 0          // Oddly, Radeon 7 is faster with this version that uses more F64 ops

    F2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2) * factor;
    WADD(0, base);
    WADD(1, base);

    for (u32 k = 2; k < MIDDLE; ++k) {
      base = cmulFancy(base, w);
      WADD(k, base);
    }
    WSUBF(0, w);

#elif MM2_CHAIN == 0

    u32 mid = MIDDLE / 2;
    F2 base = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
    WADD(mid, base);

    F2 basehi, baselo;
    cmul_a_by_fancyb_and_conjfancyb(&basehi, &baselo, base, w);
    WADD(mid-1, baselo);
    WADD(mid+1, basehi);

    for (int i = mid-2; i >= 0; --i) {
      baselo = cmulFancy(baselo, conjugate(w));
      WADD(i, baselo);
    }

    for (int i = mid+2; i < MIDDLE; ++i) {
      basehi = cmulFancy(basehi, w);
      WADD(i, basehi);
    }

#elif MM2_CHAIN == 1
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      F2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1)) * factor;
      WADD(mid, base1);

      F2 base2 = base1;
      for (u32 i = 1; i <= n; ++i) {
        base1 = cmulFancy(base1, conjugate(w));
        WADD(mid - i, base1);

        base2 = cmulFancy(base2, w);
        WADD(mid + i, base2);
      }
      if (!(sz & 1)) {
        base2 = cmulFancy(base2, w);
        WADD(mid + n + 1, base2);
      }
    }

#elif MM2_CHAIN == 2
    F2 base, base_minus1, base_plus1;
    for (u32 i = 1; ; i += 3) {
      if (i-1 == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        WADD(i-1, base_minus1);
        break;
      } else if (i == MIDDLE-1) {
        base_minus1 = slowTrig_N(x * y + x * SMALL_HEIGHT * (i - 1), ND / MIDDLE * i) * factor;
        base = cmulFancy(base_minus1, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        break;
      } else {
        base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1)) * factor;
        cmul_a_by_fancyb_and_conjfancyb(&base_plus1, &base_minus1, base, w);
        WADD(i-1, base_minus1);
        WADD(i,   base);
        WADD(i+1, base_plus1);
        if (i+1 == MIDDLE-1) break;
      }
    }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local F *lds, F2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 16) {
    local F *p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local F *p2 = lds + me;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  }
}

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global F2 *out, F2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local F2 *lds, F2 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}
#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

#if PFA_RADIX
GF31 OVERLOAD pfaMulScalar(GF31 a, Z31 s) { return U2(mul(a.x, s), mul(a.y, s)); }

// Three-point DFT using w^2 + w + 1 = 0.  The direct expression used
// four GF scalar multiplies.  This form needs only one:
//   y1 = a0 - a2 + w*(a1-a2), y2 = a0 - a1 - w*(a1-a2).
void OVERLOAD pfaDft3(GF31 a0, GF31 a1, GF31 a2, Z31 w, GF31 *o0, GF31 *o1, GF31 *o2) {
  const GF31 wd = pfaMulScalar(sub(a1, a2), w);
  *o0 = add(add(a0, a1), a2);
  *o1 = add(sub(a0, a2), wd);
  *o2 = sub(sub(a0, a1), wd);
}

#if AEVUM_PFA_GF31_COSBASE
// Adapted from PRPLL 85173461443df1e5d615f193868ca4a3c1e39f40.
// Each input is in 0..M31. Folding a constant above M31/2 keeps every
// unsigned product below 2^61; at most four products plus a0 stay below
// 2^63, hence below modM31(u64)'s 0xFFFFFFFF7FFFFFFF input limit.
// Unlike reduction-only batching, the shared cosine base removes repeated
// products. Keep the GF61 schedule (including its gfx906 NUSS8 policy) intact.
// AEVUM_PFA_GF31_COSBASE_BEGIN
u64 aevumPfa31Mad(Z31 x, Z31 c, u64 acc) {
  return c > M31 / 2 ? acc + (u64)(M31 - x) * (M31 - c)
                     : acc + (u64)x * c;
}

void aevumPfa31CosBase(GF31 *a, bool inverse) {
#if PFA_RADIX == 7
  const Z31 c[PFA_RADIX] = {1U, 141971251U, 405239506U, 526531066U, 526531066U, 405239506U, 141971251U};
  const Z31 sine[PFA_RADIX] = {0U, 752284155U, 1347360268U, 1073059520U, 1074424127U, 800123379U, 1395199492U};
#elif PFA_RADIX == 9
  const Z31 c[PFA_RADIX] = {1U, 1693082228U, 586798891U, 1073741823U, 2015086175U, 2015086175U, 1073741823U, 586798891U, 1693082228U};
  const Z31 sine[PFA_RADIX] = {0U, 1219784641U, 277691671U, 439735912U, 1205390677U, 942092970U, 1707747735U, 1869791976U, 927699006U};
#endif
  const u32 h = (PFA_RADIX - 1) / 2;
  GF31 sum[(PFA_RADIX - 1) / 2], dif[(PFA_RADIX - 1) / 2], y[PFA_RADIX];
#pragma unroll
  for (u32 k = 1; k <= h; ++k) {
    sum[k - 1] = add(a[k], a[PFA_RADIX - k]);
    dif[k - 1] = sub(a[k], a[PFA_RADIX - k]);
  }
#if PFA_RADIX == 7
  GF31 total = add(add(sum[0], sum[1]), sum[2]);
  y[0] = add(a[0], total);
  const u64 bx = aevumPfa31Mad(total.x, c[1], a[0].x);
  const u64 by = aevumPfa31Mad(total.y, c[1], a[0].y);
#else
  GF31 total = add(add(sum[0], sum[1]), sum[3]);
  GF31 t = add(a[0], sum[2]);
  y[0] = add(t, total);
  const u64 bx = aevumPfa31Mad(total.x, c[1], aevumPfa31Mad(sum[2].x, c[3], a[0].x));
  const u64 by = aevumPfa31Mad(total.y, c[1], aevumPfa31Mad(sum[2].y, c[3], a[0].y));
#endif
#pragma unroll
  for (u32 j = 1; j <= h; ++j) {
    u64 pcx = bx, pcy = by, qsx = 0, qsy = 0;
#if PFA_RADIX == 9
    if (j == 3) {
      pcx = aevumPfa31Mad(total.x, c[3], t.x);
      pcy = aevumPfa31Mad(total.y, c[3], t.y);
    } else
#endif
    {
#pragma unroll
      for (u32 k = 1; k <= h; ++k) {
        const u32 m = j * k % PFA_RADIX;
#if PFA_RADIX == 9
        if (k == 3) continue;
#endif
        if (m == 1 || m == PFA_RADIX - 1) continue;
        const Z31 cd = (c[m] + M31 - c[1]) % M31;
        pcx = aevumPfa31Mad(sum[k - 1].x, cd, pcx);
        pcy = aevumPfa31Mad(sum[k - 1].y, cd, pcy);
      }
    }
#pragma unroll
    for (u32 k = 1; k <= h; ++k) {
      const u32 m = j * k % PFA_RADIX;
      if (m == 0) continue;
      const Z31 sk = inverse ? sine[PFA_RADIX - m] : sine[m];
      qsx = aevumPfa31Mad(dif[k - 1].x, sk, qsx);
      qsy = aevumPfa31Mad(dif[k - 1].y, sk, qsy);
    }
    const GF31 pc = U2(modM31(pcx), modM31(pcy));
    const GF31 qs = U2(modM31(qsx), modM31(qsy));
    y[j] = add(pc, qs);
    y[PFA_RADIX - j] = sub(pc, qs);
  }
#pragma unroll
  for (u32 k = 0; k < PFA_RADIX; ++k) a[k] = y[k];
}
// AEVUM_PFA_GF31_COSBASE_END
#endif

#if PFA_RADIX == 7
// Correctness-first direct seven-point DFT.  Keep this simple until profiling
// identifies the odd-axis transform as the dominant PFA7 cost.
#if AEVUM_PFA7_NUSS8 && !AEVUM_GF61_LIMB32 && !defined(AEVUM_APPLE_OPENCL12)
// Nussbaumer 7-point factorization: four cosine + four sine products.
// S already represents (root^k-root^-k)/2 in the prime field; unlike the
// FP64 schedule, no multiplication by the complex unit i is needed.
void OVERLOAD aevum_pfa7_nuss8(GF31 *u, Z31 w) {
  const GF31 x0 = u[0];
  const GF31 a1 = add(u[1], u[6]), a2 = add(u[2], u[5]), a3 = add(u[3], u[4]);
  const GF31 b1 = sub(u[1], u[6]), b2 = sub(u[2], u[5]), b3 = sub(u[3], u[4]);
  const GF31 total = add(add(a1, a2), a3);
  const GF31 m2 = pfaMulScalar(sub(a1, a3), (Z31)215942690U);
  const GF31 tc = add(pfaMulScalar(sub(a2, a3), (Z31)2100158082U), m2);
  const GF31 mc = add(pfaMulScalar(sub(a2, a1), (Z31)168617125U), m2);
  const GF31 center = add(x0, pfaMulScalar(total, (Z31)357913941U));
  u[0] = add(x0, total);
  u[1] = sub(center, tc);
  u[2] = add(center, mc);
  u[3] = sub(add(center, tc), mc);

  const bool forward = w == (Z31)PFA7_ROOT31;
  const GF31 m6 = pfaMulScalar(add(b3, b1),
      forward ? (Z31)305738695U : (Z31)1841744952U);
  const GF31 tq = add(pfaMulScalar(add(b2, b3),
      forward ? (Z31)1858146229U : (Z31)289337418U), m6);
  const GF31 sq = add(pfaMulScalar(sub(b1, b2),
      forward ? (Z31)2131082370U : (Z31)16401277U), m6);
  const GF31 q0 = pfaMulScalar(add(sub(b2, b3), b1),
      forward ? (Z31)1058022850U : (Z31)1089460797U);
  const GF31 q1 = sub(q0, tq), q2 = add(q0, sq), q3 = sub(sub(sq, q0), tq);
  u[6] = sub(u[1], q1); u[1] = add(u[1], q1);
  u[5] = sub(u[2], q2); u[2] = add(u[2], q2);
  u[4] = sub(u[3], q3); u[3] = add(u[3], q3);
}
#endif

void OVERLOAD pfaDft7(GF31 *u, Z31 w) {
#if AEVUM_PFA_GF31_COSBASE
  aevumPfa31CosBase(u, w != (Z31)PFA7_ROOT31);
#elif AEVUM_PFA7_NUSS8 && !AEVUM_GF61_LIMB32 && !defined(AEVUM_APPLE_OPENCL12)
  aevum_pfa7_nuss8(u, w);
#else
 const Z31 c1 = (Z31)141971251U;
 const Z31 c2 = (Z31)405239506U;
 const Z31 c3 = (Z31)526531066U;
 const Z31 s1 = w == (Z31)PFA7_ROOT31 ? (Z31)752284155U : (Z31)1395199492U;
 const Z31 s2 = w == (Z31)PFA7_ROOT31 ? (Z31)1347360268U : (Z31)800123379U;
 const Z31 s3 = w == (Z31)PFA7_ROOT31 ? (Z31)1073059520U : (Z31)1074424127U;

 const GF31 x0 = u[0];
 const GF31 x1 = u[1];
 const GF31 x2 = u[2];
 const GF31 x3 = u[3];
 const GF31 x4 = u[4];
 const GF31 x5 = u[5];
 const GF31 x6 = u[6];

 const GF31 a1 = add(x1, x6);
 const GF31 a2 = add(x2, x5);
 const GF31 a3 = add(x3, x4);
 const GF31 b1 = sub(x1, x6);
 const GF31 b2 = sub(x2, x5);
 const GF31 b3 = sub(x3, x4);

 GF31 t1 = x0;
 t1 = add(t1, pfaMulScalar(a1, c1));
 t1 = add(t1, pfaMulScalar(a2, c2));
 t1 = add(t1, pfaMulScalar(a3, c3));
 GF31 q1 = pfaMulScalar(b1, s1);
 q1 = add(q1, pfaMulScalar(b2, s2));
 q1 = add(q1, pfaMulScalar(b3, s3));

 GF31 t2 = x0;
 t2 = add(t2, pfaMulScalar(a1, c2));
 t2 = add(t2, pfaMulScalar(a2, c3));
 t2 = add(t2, pfaMulScalar(a3, c1));
 GF31 q2 = pfaMulScalar(b1, s2);
 q2 = sub(q2, pfaMulScalar(b2, s3));
 q2 = sub(q2, pfaMulScalar(b3, s1));

 GF31 t3 = x0;
 t3 = add(t3, pfaMulScalar(a1, c3));
 t3 = add(t3, pfaMulScalar(a2, c1));
 t3 = add(t3, pfaMulScalar(a3, c2));
 GF31 q3 = pfaMulScalar(b1, s3);
 q3 = sub(q3, pfaMulScalar(b2, s1));
 q3 = add(q3, pfaMulScalar(b3, s2));

 u[0] = add(add(x0, a1), add(a2, a3));
 u[1] = add(t1, q1);
 u[6] = sub(t1, q1);
 u[2] = add(t2, q2);
 u[5] = sub(t2, q2);
 u[3] = add(t3, q3);
 u[4] = sub(t3, q3);
#endif
}
#endif

void OVERLOAD pfaForwardMiddle(GF31 *u) {
#if PFA_RADIX == 3
  GF31 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z31)1513477735U, &o0, &o1, &o2);
  u[0] = o0; u[1] = o1; u[2] = o2;
#elif PFA_RADIX == 7
  pfaDft7(u, (Z31)PFA7_ROOT31);
#elif PFA_RADIX == 9
#if AEVUM_PFA_GF31_COSBASE
  aevumPfa31CosBase(u, false);
#else
  const Z31 w9  = (Z31)765383222U;
  const Z31 w92 = (Z31)864490562U;
  const Z31 w94 = (Z31)1072993205U;
  GF31 t[9], o[9];
#pragma unroll
  for (u32 n1 = 0; n1 < 3; ++n1)
    pfaDft3(u[n1], u[n1+3], u[n1+6], (Z31)1513477735U,
            &t[n1*3], &t[n1*3+1], &t[n1*3+2]);
  t[4] = pfaMulScalar(t[4], w9);
  t[5] = pfaMulScalar(t[5], w92);
  t[7] = pfaMulScalar(t[7], w92);
  t[8] = pfaMulScalar(t[8], w94);
#pragma unroll
  for (u32 k2 = 0; k2 < 3; ++k2) {
    GF31 a, b, c;
    pfaDft3(t[k2], t[3+k2], t[6+k2], (Z31)1513477735U, &a, &b, &c);
    o[k2] = a; o[k2+3] = b; o[k2+6] = c;
  }
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = o[i];
#endif
#endif
}

void OVERLOAD pfaInverseMiddle(GF31 *u) {
#if PFA_RADIX == 3
  GF31 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z31)634005911U, &o0, &o1, &o2);
#ifdef AEVUM_PFA_MIDOUT_INV31
  u[0] = o0; u[1] = o1; u[2] = o2;
#else
  const Z31 inv = (Z31)1431655765U;
  u[0] = pfaMulScalar(o0, inv); u[1] = pfaMulScalar(o1, inv); u[2] = pfaMulScalar(o2, inv);
#endif
#elif PFA_RADIX == 7
  pfaDft7(u, (Z31)PFA7_INVROOT31);
#pragma unroll
  for (u32 i = 0; i < 7; ++i) u[i] = pfaMulScalar(u[i], (Z31)PFA7_INV7_31);
#elif PFA_RADIX == 9
#if AEVUM_PFA_GF31_COSBASE
  aevumPfa31CosBase(u, true);
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = pfaMulScalar(u[i], (Z31)1908874353U);
#else
  const Z31 w9  = (Z31)473297587U;
  const Z31 w92 = (Z31)309107220U;
  const Z31 w94 = (Z31)809695498U;
  const Z31 inv9 = (Z31)1908874353U;
  GF31 t[9], o[9];
#pragma unroll
  for (u32 n1 = 0; n1 < 3; ++n1)
    pfaDft3(u[n1], u[n1+3], u[n1+6], (Z31)634005911U,
            &t[n1*3], &t[n1*3+1], &t[n1*3+2]);
  t[4] = pfaMulScalar(t[4], w9);
  t[5] = pfaMulScalar(t[5], w92);
  t[7] = pfaMulScalar(t[7], w92);
  t[8] = pfaMulScalar(t[8], w94);
#pragma unroll
  for (u32 k2 = 0; k2 < 3; ++k2) {
    GF31 a, b, c;
    pfaDft3(t[k2], t[3+k2], t[6+k2], (Z31)634005911U, &a, &b, &c);
    o[k2] = a; o[k2+3] = b; o[k2+6] = c;
  }
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = pfaMulScalar(o[i], inv9);
#endif
#endif
}
#endif

#if PFA_RADIX
// WIDTH x SMALL_HEIGHT Cooley-Tukey twiddle for the binary axis of a
// Good-Thomas plan.  The odd radix-R axis is coprime to the binary axis and
// needs no twiddles against it, but WIDTH and SMALL_HEIGHT are both powers of
// two, so the W*SH binary transform still needs w_(W*SH)^(x*y) between its
// width pass (fftP / fftW) and its height pass (tail).  The scalar depends only
// on (x, y), so it is identical for every odd-row register and commutes with
// the radix-R transform.  The inverse direction reuses the same forward table
// because the inverse binary passes run on component-swapped (conjugated)
// data, exactly as stock middleMul2 does.
// PFA middle trig layout (TrigBufCache genMiddleTrigGF31(smallH, 1, width)):
//   trig[0 .. WIDTH)                   : w_WIDTH^k
//   trig[WIDTH .. WIDTH + SMALL_HEIGHT) : w_(WIDTH*SMALL_HEIGHT)^k
void OVERLOAD pfaMiddleTwiddle(GF31 *u, u32 x, u32 y, TrigGF31 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  const u32 desired_root = x * y;
  GF31 w = cmul(TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT]));
#ifdef AEVUM_PFA_MIDOUT_INV31
  w = pfaMulScalar(w, (Z31)AEVUM_PFA_MIDOUT_INV31);
#endif
  for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, w); }
}
#endif

void OVERLOAD fft2(GF31* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(GF31 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 16
  fft16(u);
#elif PFA_RADIX == 3 && MIDDLE == 3
  pfaForwardMiddle(u);
#elif PFA_RADIX == 7 && MIDDLE == 7
  pfaForwardMiddle(u);
#elif PFA_RADIX == 9 && MIDDLE == 9
  pfaForwardMiddle(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

void OVERLOAD ifft_MIDDLE(GF31 *u) {
#if PFA_RADIX
  pfaInverseMiddle(u);
#else
  fft_MIDDLE(u);
#endif
}

void OVERLOAD middleMul(GF31 *u, u32 s, TrigGF31 trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

#if !MIDDLE_CHAIN           // Read all trig values from memory

  for (u32 k = 1; k < MIDDLE; ++k) {
    WADD(k, TFLOAD(&trig[s]));
    s += SMALL_HEIGHT;
  }

#else

  GF31 w = TFLOAD(&trig[s]);         // s / BIG_HEIGHT
  WADD(1, w);
  if (MIDDLE == 2) return;

#if SHOULD_BE_FASTER
  GF31 sq = csqTrig(w);
  WADD(2, sq);
  GF31 base = ccubeTrig(sq, w);                         // GWBUG: compute w^4 as csqTriq(sq), w^6 as ccubeTrig(w2, w4), and w^5 and w^7 as cmul_a_by_b_and_conjb
  for (u32 k = 3; k < MIDDLE; ++k) {
#else
  GF31 base = csq(w);
  for (u32 k = 2; k < MIDDLE; ++k) {
#endif
    WADD(k, base);
    base = cmul(base, w);
  }

#endif

}

void OVERLOAD middleMul2(GF31 *u, u32 x, u32 y, TrigGF31 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  // First trig table comes after the MiddleMul trig table.  Second trig table comes after the first MiddleMul2 trig table.
  TrigGF31 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF31 trig2 = trig1 + WIDTH;
  // The first trig table can be shared with MiddleMul trig table if WIDTH = HEIGHT.
  if (WIDTH == SMALL_HEIGHT) trig1 = trig;

  GF31 w = TFLOAD(&trig1[x]);         // x / (MIDDLE * WIDTH)
  u32 desired_root = x * y;
  GF31 base = cmul(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig1[desired_root / SMALL_HEIGHT]));

  WADD(0, base);
  for (u32 k = 1; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local Z31 *lds, GF31 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 16) {
    local Z31 *p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local Z31 *p2 = lds + me;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  }
}

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global GF31 *out, GF31 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local GF31 *lds, GF31 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#ifndef AEVUM_GF61_MIDDLE_BASE_TABLE
#define AEVUM_GF61_MIDDLE_BASE_TABLE 0
#endif
#if AEVUM_GF61_MIDDLE_BASE_TABLE < 0 || AEVUM_GF61_MIDDLE_BASE_TABLE > 3
#error AEVUM_GF61_MIDDLE_BASE_TABLE must be 0..3
#endif

#if PFA_RADIX
#ifndef AEVUM_PFA7_GF61_COSBASE
#define AEVUM_PFA7_GF61_COSBASE 0
#endif
#ifndef AEVUM_PFA7_GF61_SINBASE
#define AEVUM_PFA7_GF61_SINBASE 0
#endif
#if AEVUM_PFA7_GF61_SINBASE != 0 && AEVUM_PFA7_GF61_SINBASE != 1
#error AEVUM_PFA7_GF61_SINBASE must be 0 or 1
#endif
#if AEVUM_PFA7_GF61_SINBASE && !AEVUM_PFA7_GF61_COSBASE
#error AEVUM_PFA7_GF61_SINBASE requires AEVUM_PFA7_GF61_COSBASE
#endif
#ifndef AEVUM_PFA7_GF61_BOUNDED_BASE
#define AEVUM_PFA7_GF61_BOUNDED_BASE 0
#endif
#if AEVUM_PFA7_GF61_BOUNDED_BASE != 0 && AEVUM_PFA7_GF61_BOUNDED_BASE != 1
#error AEVUM_PFA7_GF61_BOUNDED_BASE must be 0 or 1
#endif
#if AEVUM_PFA7_GF61_BOUNDED_BASE && (!AEVUM_PFA7_GF61_COSBASE || !AEVUM_PFA7_GF61_SINBASE)
#error AEVUM_PFA7_GF61_BOUNDED_BASE requires COSBASE=1 and SINBASE=1
#endif
#if AEVUM_PFA7_GF61_COSBASE != 0 && AEVUM_PFA7_GF61_COSBASE != 1
#error AEVUM_PFA7_GF61_COSBASE must be 0 or 1
#endif
#if AEVUM_PFA7_GF61_COSBASE && (FFT_TYPE != FFT3161 || PFA_RADIX != 7 || WIDTH != 512 || MIDDLE != 7 || SMALL_HEIGHT != 512 || AEVUM_GF61_LIMB32)
#error AEVUM_PFA7_GF61_COSBASE requires the guarded native Type1 PFA7 plan
#endif

GF61 OVERLOAD pfaMulScalar(GF61 a, Z61 s) { return U2(mul(a.x, s), mul(a.y, s)); }

// Same one-multiply radix-3 identity as the GF31 path above.
void OVERLOAD pfaDft3(GF61 a0, GF61 a1, GF61 a2, Z61 w, GF61 *o0, GF61 *o1, GF61 *o2) {
  const GF61 wd = pfaMulScalar(sub(a1, a2), w);
  *o0 = add(add(a0, a1), a2);
  *o1 = add(sub(a0, a2), wd);
  *o2 = sub(sub(a0, a1), wd);
}

#if PFA_RADIX == 7
#if AEVUM_PFA7_NUSS8 && !AEVUM_GF61_LIMB32 && !defined(AEVUM_APPLE_OPENCL12)
// Nussbaumer 7-point factorization: four cosine + four sine products.
// S already represents (root^k-root^-k)/2 in the prime field; unlike the
// FP64 schedule, no multiplication by the complex unit i is needed.
void OVERLOAD aevum_pfa7_nuss8(GF61 *u, Z61 w) {
  const GF61 x0 = u[0];
  const GF61 a1 = add(u[1], u[6]), a2 = add(u[2], u[5]), a3 = add(u[3], u[4]);
  const GF61 b1 = sub(u[1], u[6]), b2 = sub(u[2], u[5]), b3 = sub(u[3], u[4]);
  const GF61 total = add(add(a1, a2), a3);
  const GF61 m2 = pfaMulScalar(sub(a1, a3), (Z61)916476061952082211UL);
  const GF61 tc = add(pfaMulScalar(sub(a2, a3), (Z61)1789062672330551777UL), m2);
  const GF61 mc = add(pfaMulScalar(sub(a2, a1), (Z61)399695725068940037UL), m2);
  const GF61 center = add(x0, pfaMulScalar(total, (Z61)384307168202282325UL));
  u[0] = add(x0, total);
  u[1] = sub(center, tc);
  u[2] = add(center, mc);
  u[3] = sub(add(center, tc), mc);

  const bool forward = w == (Z61)PFA7_ROOT61;
  const GF61 m6 = pfaMulScalar(add(b3, b1),
      forward ? (Z61)2007045408887657666UL : (Z61)298797600326036285UL);
  const GF61 tq = add(pfaMulScalar(add(b2, b3),
      forward ? (Z61)1848645854929387642UL : (Z61)457197154284306309UL), m6);
  const GF61 sq = add(pfaMulScalar(sub(b1, b2),
      forward ? (Z61)755994754610342594UL : (Z61)1549848254603351357UL), m6);
  const GF61 q0 = pfaMulScalar(add(sub(b2, b3), b1),
      forward ? (Z61)374686897387646219UL : (Z61)1931156111826047732UL);
  const GF61 q1 = sub(q0, tq), q2 = add(q0, sq), q3 = sub(sub(sq, q0), tq);
  u[6] = sub(u[1], q1); u[1] = add(u[1], q1);
  u[5] = sub(u[2], q2); u[2] = add(u[2], q2);
  u[4] = sub(u[3], q3); u[3] = add(u[3], q3);
}
#endif

#if AEVUM_PFA7_GF61_COSBASE
// Shared cosine base from PRPLL 8517346/f597e42, with the native generator-5
// root and its inverse. This is NOT the older three-product DOT128 schedule:
// cbase removes one product from each cosine row (16 vector products total).
typedef struct { u128 x, y; } AevumPfa61Quick;

AevumPfa61Quick aevumPfa61Mad(GF61 a, Z61 b, AevumPfa61Quick c) {
  AevumPfa61Quick r = {mad64(a.x,b,c.x), mad64(a.y,b,c.y)};
  return r;
}
AevumPfa61Quick aevumPfa61Mul(GF61 a, Z61 b) {
  AevumPfa61Quick r = {mul64(a.x,b), mul64(a.y,b)};
  return r;
}
void aevumPfa61Pair(AevumPfa61Quick a, AevumPfa61Quick b, GF61* plus, GF61* minus) {
  // A sine dot can exceed 2^125. Use the full 128-bit weak reduction, whose
  // outputs are at most 2*M61+epsilon; then combine in 64 bits.
  const GF61 pc = U2(weakModM61(a.x,128), weakModM61(a.y,128));
  const GF61 qs = U2(weakModM61(b.x,128), weakModM61(b.y,128));
  *plus = modM61(pc+qs);
  *minus = modM61q(pc-qs,3);
}
#if AEVUM_PFA7_GF61_BOUNDED_BASE
// PRPLL f361b5e/1418b96: choose bases for the dot bounds as well as product
// count. Native generator-5 C3/S3 coefficients permit six 125-bit folds.
void aevumPfa61BoundedPair(AevumPfa61Quick a, AevumPfa61Quick b, u32 sine_bound,
                    GF61* plus, GF61* minus) {
  // a < 2*M61^2, b < sine_bound*M61^2, sine_bound = 2 or 3.
  // Thus pc < 3*M61, qs < (sine_bound+1)*M61; both combinations stay
  // nonnegative and below 7*M61 before modM61. No full 128-bit fold is needed.
  const GF61 pc = U2(weakModM61(a.x,125), weakModM61(a.y,125));
  const GF61 qs = U2(weakModM61(b.x,125), weakModM61(b.y,125));
  *plus = modM61(pc+qs);
  *minus = modM61q(pc-qs,sine_bound+1);
}
void aevumPfa7BoundedBase61(GF61* u, Z61 w) {
  const Z61 C3 = (Z61)784002893271222362UL;
  const Z61 C1_C3 = (Z61)989671222192671703UL;
  const Z61 C2_C3 = (Z61)117084611814202137UL;
  const Z61 S3 = (Z61)381307857222696375UL;
  const Z61 S1_S3 = (Z61)292176640490986129UL;
  const Z61 S2_S3 = (Z61)450576194449256153UL;
  const Z61 NS2_S3 = (Z61)1092651100319045048UL;
  const Z61 NS1_S3 = (Z61)1251050654277315072UL;
  // Inverse roots only reverse the output pairs. Use the same bounded
  // positive coefficients in both directions; w is constant at each caller.
  const bool inverse = w != (Z61)PFA7_ROOT61;
  // Inputs 0..M61+epsilon. Quick sums 0..2*M61+epsilon and positive
  // differences M61-epsilon..3*M61+epsilon fit in 64 bits.
  const GF61 a1=u[1]+u[6], a2=u[2]+u[5], a3=u[3]+u[4];
  const GF61 b1=u[1]-u[6]+U2(2*M61,2*M61);
  const GF61 b2=u[2]-u[5]+U2(2*M61,2*M61);
  const GF61 b3=u[3]-u[4]+U2(2*M61,2*M61);
  const GF61 sum=modM61(a1+a2+a3);
  const AevumPfa61Quick cbase={mad64(sum.x,C3,u[0].x),mad64(sum.y,C3,u[0].y)};
  u[0]=modM61(u[0]+sum);
  // The intermediate u64 sum can wrap, but the true sum minus 2*M61 lies
  // in M61-epsilon..7*M61+epsilon. Unsigned wraparound preserves that value.
  const GF61 dsum=modM61(b1+b2+b3-U2(2*M61,2*M61));
  const AevumPfa61Quick sbase=aevumPfa61Mul(dsum,S3);

  // One output pair at a time; six wide dots, each below 3*M61^2. No matrix
  // or dynamically indexed temporaries, and no 128-bit add/sub of two dots.
  AevumPfa61Quick pc=cbase, qs=sbase;
  pc=aevumPfa61Mad(a1,C1_C3,pc); qs=aevumPfa61Mad(b1,S1_S3,qs);
  pc=aevumPfa61Mad(a2,C2_C3,pc); qs=aevumPfa61Mad(b2,S2_S3,qs);
  aevumPfa61BoundedPair(pc,qs,2,inverse?&u[6]:&u[1],inverse?&u[1]:&u[6]);

  pc=cbase; qs=sbase;
  pc=aevumPfa61Mad(a1,C2_C3,pc); qs=aevumPfa61Mad(b1,NS2_S3,qs);
  pc=aevumPfa61Mad(a3,C1_C3,pc); qs=aevumPfa61Mad(b3,S1_S3,qs);
  // This row computes negated sines, so its plus/minus outputs are exchanged.
  aevumPfa61BoundedPair(pc,qs,2,inverse?&u[2]:&u[5],inverse?&u[5]:&u[2]);

  pc=cbase; qs=sbase;
  pc=aevumPfa61Mad(a2,C1_C3,pc); qs=aevumPfa61Mad(b2,NS1_S3,qs);
  pc=aevumPfa61Mad(a3,C2_C3,pc); qs=aevumPfa61Mad(b3,S2_S3,qs);
  aevumPfa61BoundedPair(pc,qs,3,inverse?&u[4]:&u[3],inverse?&u[3]:&u[4]);
}
#endif

void aevumPfa7CosBase61(GF61* u, Z61 w) {
  const Z61 C1 = (Z61)1773674115463894065UL;
  const Z61 C2_C1 = (Z61)1433256398835224385UL;
  const Z61 C3_C1 = (Z61)1316171787021022248UL;
  const Z61 S1 = w==(Z61)PFA7_ROOT61 ? (Z61)673484497713682504UL : (Z61)1632358511500011447UL;
  const Z61 S2 = w==(Z61)PFA7_ROOT61 ? (Z61)831884051671952528UL : (Z61)1473958957541741423UL;
  const Z61 S3 = w==(Z61)PFA7_ROOT61 ? (Z61)381307857222696375UL : (Z61)1924535151990997576UL;
  // Inputs 0..M61+epsilon. Quick sums 0..2*M61+epsilon and positive
  // differences M61-epsilon..3*M61+epsilon fit in 64 bits.
  const GF61 a1=u[1]+u[6], a2=u[2]+u[5], a3=u[3]+u[4];
  const GF61 b1=u[1]-u[6]+U2(2*M61,2*M61);
  const GF61 b2=u[2]-u[5]+U2(2*M61,2*M61);
  const GF61 b3=u[3]-u[4]+U2(2*M61,2*M61);
  const GF61 sum=modM61(a1+a2+a3);
  const AevumPfa61Quick cbase={mad64(sum.x,C1,u[0].x),mad64(sum.y,C1,u[0].y)};
  u[0]=modM61(u[0]+sum);

  // One output pair at a time; six wide dots, each below 2^126. No matrix
  // or dynamically indexed temporaries, and no 128-bit add/sub of two dots.
#if AEVUM_PFA7_GF61_SINBASE
  // PRPLL sine-base decomposition (gwoltman/gpuowl@1418b96f814a),
  // rederived for Aevum's generator-5 GF61 roots.  The single shared
  // sbase eliminates one constant product per sine row: 16 -> 14 total
  // scalar products over the radix-7 cosine+sine DFT, with six wide dots.
  // b1+b2+b3 may exceed 64 bits (up to 9*M61); fold the first two before
  // adding b3 so the unsigned addition cannot wrap modulo 2^64.
  const GF61 dsum=modM61(modM61(b1+b2)+b3);
  const AevumPfa61Quick sbase=aevumPfa61Mul(dsum,S1);
  const bool fwd=w==(Z61)PFA7_ROOT61;
  const Z61 S2_S1=fwd ? (Z61)158399553958270024UL : (Z61)2147443455255423927UL;
  const Z61 S3_S1=fwd ? (Z61)2013666368722707822UL : (Z61)292176640490986129UL;
  const Z61 NS2_S1=fwd ? (Z61)800474459828058919UL : (Z61)1505368549385635032UL;
  const Z61 NS3_S1=fwd ? (Z61)1251050654277315072UL : (Z61)1054792354936378879UL;

  AevumPfa61Quick pc=cbase, qs=sbase;
  pc=aevumPfa61Mad(a2,C2_C1,pc); qs=aevumPfa61Mad(b2,S2_S1,qs);
  pc=aevumPfa61Mad(a3,C3_C1,pc); qs=aevumPfa61Mad(b3,S3_S1,qs);
  aevumPfa61Pair(pc,qs,&u[1],&u[6]);

  // The second and third sine rows use the NEGATED sine sums, so every
  // term remains positive in the wide accumulator.  Swap the two outputs
  // to restore the exact original transform orientation.
  pc=cbase; qs=sbase;
  pc=aevumPfa61Mad(a1,C2_C1,pc); qs=aevumPfa61Mad(b1,NS2_S1,qs);
  pc=aevumPfa61Mad(a2,C3_C1,pc); qs=aevumPfa61Mad(b2,S3_S1,qs);
  aevumPfa61Pair(pc,qs,&u[5],&u[2]);

  pc=cbase; qs=sbase;
  pc=aevumPfa61Mad(a1,C3_C1,pc); qs=aevumPfa61Mad(b1,NS3_S1,qs);
  pc=aevumPfa61Mad(a3,C2_C1,pc); qs=aevumPfa61Mad(b3,NS2_S1,qs);
  aevumPfa61Pair(pc,qs,&u[4],&u[3]);
#else
  AevumPfa61Quick pc=cbase, qs=aevumPfa61Mul(b1,S1);
  pc=aevumPfa61Mad(a2,C2_C1,pc); qs=aevumPfa61Mad(b2,S2,qs);
  pc=aevumPfa61Mad(a3,C3_C1,pc); qs=aevumPfa61Mad(b3,S3,qs);
  aevumPfa61Pair(pc,qs,&u[1],&u[6]);

  pc=cbase; qs=aevumPfa61Mul(b1,S2);
  pc=aevumPfa61Mad(a1,C2_C1,pc); qs=aevumPfa61Mad(b2,M61-S3,qs);
  pc=aevumPfa61Mad(a2,C3_C1,pc); qs=aevumPfa61Mad(b3,M61-S1,qs);
  aevumPfa61Pair(pc,qs,&u[2],&u[5]);

  pc=cbase; qs=aevumPfa61Mul(b1,S3);
  pc=aevumPfa61Mad(a1,C3_C1,pc); qs=aevumPfa61Mad(b2,M61-S1,qs);
  pc=aevumPfa61Mad(a3,C2_C1,pc); qs=aevumPfa61Mad(b3,S2,qs);
  aevumPfa61Pair(pc,qs,&u[3],&u[4]);
#endif
}
#endif

void OVERLOAD pfaDft7(GF61 *u, Z61 w) {
#if AEVUM_PFA7_GF61_BOUNDED_BASE
  aevumPfa7BoundedBase61(u,w);
#elif AEVUM_PFA7_GF61_COSBASE
  aevumPfa7CosBase61(u,w);
#elif AEVUM_PFA7_NUSS8 && !AEVUM_GF61_LIMB32 && !defined(AEVUM_APPLE_OPENCL12)
  aevum_pfa7_nuss8(u, w);
#else
 const Z61 c1 = (Z61)1773674115463894065UL;
 const Z61 c2 = (Z61)901087505085424499UL;
 const Z61 c3 = (Z61)784002893271222362UL;
 const Z61 s1 = w == (Z61)PFA7_ROOT61 ? (Z61)673484497713682504UL : (Z61)1632358511500011447UL;
 const Z61 s2 = w == (Z61)PFA7_ROOT61 ? (Z61)831884051671952528UL : (Z61)1473958957541741423UL;
 const Z61 s3 = w == (Z61)PFA7_ROOT61 ? (Z61)381307857222696375UL : (Z61)1924535151990997576UL;

 const GF61 x0 = u[0];
 const GF61 x1 = u[1];
 const GF61 x2 = u[2];
 const GF61 x3 = u[3];
 const GF61 x4 = u[4];
 const GF61 x5 = u[5];
 const GF61 x6 = u[6];

 const GF61 a1 = add(x1, x6);
 const GF61 a2 = add(x2, x5);
 const GF61 a3 = add(x3, x4);
 const GF61 b1 = sub(x1, x6);
 const GF61 b2 = sub(x2, x5);
 const GF61 b3 = sub(x3, x4);

 GF61 t1 = x0;
 t1 = add(t1, pfaMulScalar(a1, c1));
 t1 = add(t1, pfaMulScalar(a2, c2));
 t1 = add(t1, pfaMulScalar(a3, c3));
 GF61 q1 = pfaMulScalar(b1, s1);
 q1 = add(q1, pfaMulScalar(b2, s2));
 q1 = add(q1, pfaMulScalar(b3, s3));

 GF61 t2 = x0;
 t2 = add(t2, pfaMulScalar(a1, c2));
 t2 = add(t2, pfaMulScalar(a2, c3));
 t2 = add(t2, pfaMulScalar(a3, c1));
 GF61 q2 = pfaMulScalar(b1, s2);
 q2 = sub(q2, pfaMulScalar(b2, s3));
 q2 = sub(q2, pfaMulScalar(b3, s1));

 GF61 t3 = x0;
 t3 = add(t3, pfaMulScalar(a1, c3));
 t3 = add(t3, pfaMulScalar(a2, c1));
 t3 = add(t3, pfaMulScalar(a3, c2));
 GF61 q3 = pfaMulScalar(b1, s3);
 q3 = sub(q3, pfaMulScalar(b2, s1));
 q3 = add(q3, pfaMulScalar(b3, s2));

 u[0] = add(add(x0, a1), add(a2, a3));
 u[1] = add(t1, q1);
 u[6] = sub(t1, q1);
 u[2] = add(t2, q2);
 u[5] = sub(t2, q2);
 u[3] = add(t3, q3);
 u[4] = sub(t3, q3);
#endif
}
#endif

void OVERLOAD pfaForwardMiddle(GF61 *u) {
#if PFA_RADIX == 3
  GF61 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z61)1669582390241348315UL, &o0, &o1, &o2);
  u[0] = o0; u[1] = o1; u[2] = o2;
#elif PFA_RADIX == 7
  pfaDft7(u, (Z61)PFA7_ROOT61);
#elif PFA_RADIX == 9
  const Z61 w9  = (Z61)1102844585000305877UL;
  const Z61 w92 = (Z61)594418010121383343UL;
  const Z61 w94 = (Z61)569931187132395942UL;
  GF61 t[9], o[9];
#pragma unroll
  for (u32 n1 = 0; n1 < 3; ++n1)
    pfaDft3(u[n1], u[n1+3], u[n1+6], (Z61)1669582390241348315UL,
            &t[n1*3], &t[n1*3+1], &t[n1*3+2]);
  t[4] = pfaMulScalar(t[4], w9);
  t[5] = pfaMulScalar(t[5], w92);
  t[7] = pfaMulScalar(t[7], w92);
  t[8] = pfaMulScalar(t[8], w94);
#pragma unroll
  for (u32 k2 = 0; k2 < 3; ++k2) {
    GF61 a, b, c;
    pfaDft3(t[k2], t[3+k2], t[6+k2], (Z61)1669582390241348315UL, &a, &b, &c);
    o[k2] = a; o[k2+3] = b; o[k2+6] = c;
  }
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = o[i];
#endif
}

void OVERLOAD pfaInverseMiddle(GF61 *u) {
#if PFA_RADIX == 3
  GF61 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z61)636260618972345635UL, &o0, &o1, &o2);
#ifdef AEVUM_PFA_MIDOUT_INV61
  u[0] = o0; u[1] = o1; u[2] = o2;
#else
  const Z61 inv = (Z61)1537228672809129301UL;
  u[0] = pfaMulScalar(o0, inv); u[1] = pfaMulScalar(o1, inv); u[2] = pfaMulScalar(o2, inv);
#endif
#elif PFA_RADIX == 7
#ifdef AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE
  pfaDft7(u, (Z61)PFA7_INVROOT61);
#else
  pfaDft7(u, (Z61)PFA7_INVROOT61);
#pragma unroll
  for (u32 i = 0; i < 7; ++i) u[i] = pfaMulScalar(u[i], (Z61)PFA7_INV7_61);
#endif
#elif PFA_RADIX == 9
  const Z61 w9  = (Z61)2252987116782656529UL;
  const Z61 w92 = (Z61)633067237080992132UL;
  const Z61 w94 = (Z61)1764280891523348030UL;
  const Z61 inv9 = (Z61)2049638230412172401UL;
  GF61 t[9], o[9];
#pragma unroll
  for (u32 n1 = 0; n1 < 3; ++n1)
    pfaDft3(u[n1], u[n1+3], u[n1+6], (Z61)636260618972345635UL,
            &t[n1*3], &t[n1*3+1], &t[n1*3+2]);
  t[4] = pfaMulScalar(t[4], w9);
  t[5] = pfaMulScalar(t[5], w92);
  t[7] = pfaMulScalar(t[7], w92);
  t[8] = pfaMulScalar(t[8], w94);
#pragma unroll
  for (u32 k2 = 0; k2 < 3; ++k2) {
    GF61 a, b, c;
    pfaDft3(t[k2], t[3+k2], t[6+k2], (Z61)636260618972345635UL, &a, &b, &c);
    o[k2] = a; o[k2+3] = b; o[k2+6] = c;
  }
#ifdef AEVUM_PFA9_MIDOUT_FUSE_INV_SCALE
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = o[i];
#else
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = pfaMulScalar(o[i], inv9);
#endif
#endif
}
#endif

#if PFA_RADIX
// WIDTH x SMALL_HEIGHT Cooley-Tukey twiddle of the binary axis; see the GF31
// pfaMiddleTwiddle above for the rationale and the PFA middle trig layout.
void OVERLOAD pfaMiddleTwiddle(GF61 *u, u32 x, u32 y, TrigGF61 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  const u32 desired_root = x * y;
#if defined(AEVUM_PFA_MIDOUT_INV61)
  const GF61 w = pfaMulScalar(
      cmul(TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]),
           TFLOAD(&trig[desired_root / SMALL_HEIGHT])),
      (Z61)AEVUM_PFA_MIDOUT_INV61);
#elif defined(AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE)
#if AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE == 1
  const GF61 w = cmul(
      pfaMulScalar(TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]), (Z61)PFA7_INV7_61),
      TFLOAD(&trig[desired_root / SMALL_HEIGHT]));
#elif AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE == 2
  const GF61 w = cmul(
      TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]),
      pfaMulScalar(TFLOAD(&trig[desired_root / SMALL_HEIGHT]), (Z61)PFA7_INV7_61));
#else
#error INVALID AEVUM_PFA7_MIDOUT_TWIDDLE_PRESCALE
#endif
#elif defined(AEVUM_PFA9_MIDOUT_FUSE_INV_SCALE)
  GF61 w = cmul(TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT]));
  w = pfaMulScalar(w, (Z61)2049638230412172401UL);
#else
  const GF61 w = cmul(TFLOAD(&trig[WIDTH + desired_root % SMALL_HEIGHT]), TFLOAD(&trig[desired_root / SMALL_HEIGHT]));
#endif
  for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, w); }
}
#endif

void OVERLOAD fft2(GF61* u) { X2(u[0], u[1]); }

void OVERLOAD fft_MIDDLE(GF61 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 16
  fft16(u);
#elif PFA_RADIX == 3 && MIDDLE == 3
  pfaForwardMiddle(u);
#elif PFA_RADIX == 7 && MIDDLE == 7
  pfaForwardMiddle(u);
#elif PFA_RADIX == 9 && MIDDLE == 9
  pfaForwardMiddle(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

void OVERLOAD ifft_MIDDLE(GF61 *u) {
#if PFA_RADIX
  pfaInverseMiddle(u);
#else
  fft_MIDDLE(u);
#endif
}

void OVERLOAD middleMul(GF61 *u, u32 s, TrigGF61 trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) return;

#if !MIDDLE_CHAIN           // Read all trig values from memory

  for (u32 k = 1; k < MIDDLE; ++k) {
    WADD(k, TFLOAD(&trig[s]));
    s += SMALL_HEIGHT;
  }

#else

  GF61 w = TFLOAD(&trig[s]);         // s / BIG_HEIGHT
  WADD(1, w);
  if (MIDDLE == 2) return;

#if SHOULD_BE_FASTER
  GF61 sq = csqTrig(w);
  WADD(2, sq);
  GF61 base = ccubeTrig(sq, w);                         // GWBUG: compute w^4 as csqTriq(sq), w^6 as ccubeTrig(w2, w4), and w^5 and w^7 as cmul_a_by_b_and_conjb
  for (u32 k = 3; k < MIDDLE; ++k) {
#else
  GF61 base = csq(w);
  for (u32 k = 2; k < MIDDLE; ++k) {
#endif
    WADD(k, base);
    base = cmul(base, w);
  }

#endif

}

void OVERLOAD middleMul2(GF61 *u, u32 x, u32 y, TrigGF61 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  // First trig table comes after the MiddleMul trig table.  Second trig table comes after the first MiddleMul2 trig table.
  TrigGF61 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF61 trig2 = trig1 + WIDTH;
  // The first trig table can be shared with MiddleMul trig table if WIDTH = HEIGHT.
  if (WIDTH == SMALL_HEIGHT) trig1 = trig;

  GF61 w = TFLOAD(&trig1[x]);                      // x / (MIDDLE * WIDTH)
  u32 desired_root = x * y;
  GF61 base = cmul(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]), TFLOAD(&trig1[desired_root / SMALL_HEIGHT]));

  WADD(0, base);
  for (u32 k = 1; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }
}

void OVERLOAD middleMul2BaseTable(GF61 *u, u32 x, u32 y, u32 baseIndex, TrigGF61 trig) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  assert(baseIndex < WIDTH * SMALL_HEIGHT);

  TrigGF61 trig1 = trig + SMALL_HEIGHT * (MIDDLE - 1);
  if (WIDTH == SMALL_HEIGHT) trig1 = trig;
  TrigGF61 trigBase = trig + SMALL_HEIGHT * (MIDDLE - 1) + WIDTH + SMALL_HEIGHT;

  GF61 w = TFLOAD(&trig1[x]);
  GF61 base = TFLOAD(&trigBase[baseIndex]);

  WADD(0, base);
  for (u32 k = 1; k < MIDDLE; ++k) {
    base = cmul(base, w);
    WADD(k, base);
  }
}

// Do a partial transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local Z61 *lds, GF61 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    local Z61 *p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local Z61 *p2 = lds + me;
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = p2[workgroupSize * i]; }
  } else {
    local int *p1 = ((local int*) lds) + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local int *p2 = (local int*) lds + me;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[workgroupSize * i]; }
  }
}

// Do a partial transpose during fftMiddleIn/Out and write the results to global memory
void OVERLOAD middleShuffleWrite(global GF61 *out, GF61 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  out += (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { out[i * workgroupSize] = u[i]; }
}

// Do an in-place 16x16 transpose during fftMiddleIn/Out
void OVERLOAD middleShuffle(local GF61 *lds, GF61 *u) {
  u32 me = get_local_id(0);
  u32 y = me / 16;
  u32 x = me % 16;
  for (int i = 0; i < MIDDLE; ++i) {
    lds[x * 16 + y ^ x] = u[i];         // Swizzling with XOR should reduce LDS bank conflicts, formerly "lds[x * 16 + y] = u[i];"
    bar();
    u[i] = lds[y * 16 + x ^ y];         // Formerly "u[i] = lds[me];"
    if (++i == MIDDLE) break;
    lds[y * 16 + x ^ y] = u[i];
    bar();
    u[i] = lds[x * 16 + y ^ x];
  }
}

#endif


#undef WADD
#undef WADDF
#undef WSUB
#undef WSUBF
