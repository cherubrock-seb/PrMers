// Copyright (C) Mihai Preda

#include "math.cl"
#include "trig.cl"

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
#elif MIDDLE == 16
  fft16(u);
#else
#error UNRECOGNIZED MIDDLE
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

void OVERLOAD pfaForwardMiddle(GF31 *u) {
#if PFA_RADIX == 3
  GF31 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z31)1513477735U, &o0, &o1, &o2);
  u[0] = o0; u[1] = o1; u[2] = o2;
#elif PFA_RADIX == 9
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
}

void OVERLOAD pfaInverseMiddle(GF31 *u) {
#if PFA_RADIX == 3
  GF31 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z31)634005911U, &o0, &o1, &o2);
  const Z31 inv = (Z31)1431655765U;
  u[0] = pfaMulScalar(o0, inv); u[1] = pfaMulScalar(o1, inv); u[2] = pfaMulScalar(o2, inv);
#elif PFA_RADIX == 9
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

#if PFA_RADIX
GF61 OVERLOAD pfaMulScalar(GF61 a, Z61 s) { return U2(mul(a.x, s), mul(a.y, s)); }

// Same one-multiply radix-3 identity as the GF31 path above.
void OVERLOAD pfaDft3(GF61 a0, GF61 a1, GF61 a2, Z61 w, GF61 *o0, GF61 *o1, GF61 *o2) {
  const GF61 wd = pfaMulScalar(sub(a1, a2), w);
  *o0 = add(add(a0, a1), a2);
  *o1 = add(sub(a0, a2), wd);
  *o2 = sub(sub(a0, a1), wd);
}

void OVERLOAD pfaForwardMiddle(GF61 *u) {
#if PFA_RADIX == 3
  GF61 o0, o1, o2;
  pfaDft3(u[0], u[1], u[2], (Z61)1669582390241348315UL, &o0, &o1, &o2);
  u[0] = o0; u[1] = o1; u[2] = o2;
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
  const Z61 inv = (Z61)1537228672809129301UL;
  u[0] = pfaMulScalar(o0, inv); u[1] = pfaMulScalar(o1, inv); u[2] = pfaMulScalar(o2, inv);
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
#pragma unroll
  for (u32 i = 0; i < 9; ++i) u[i] = pfaMulScalar(o[i], inv9);
#endif
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
