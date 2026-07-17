// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "fft-middle.cl"
#include "middle.cl"

#if !INPLACE                   // Original implementation (not in place)

#if FFT_FP64

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, CP(T2) in, Trig trig) {
  T2 u[MIDDLE];

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquareFP64 that launched dependents before writing FP64 data

  readMiddleOutLine(u, in, y, x);

  middleMul(u, x, trig);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

  middleMul2(u, y, x, factor, trig);

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local T lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
  out += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if FFT_FP32

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, CP(T2) in, Trig trig) {
  F2 u[MIDDLE];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquareFP32 that launched dependents before writing FP64 data

  readMiddleOutLine(u, inF2, y, x);

  middleMul(u, x, trigF2);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS * 2).
  const float factor = 1.0f / (NWORDS * 2);

  middleMul2(u, y, x, factor, trigF2);

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local F lds[OUT_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
  outF2 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  outF2 += mx * SIZEY + my;
#endif

  writeMiddleOutLine(outF2, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(OUT_WG) fftMiddleOutGF31(P(T2) out, CP(T2) in, Trig trig) {
  GF31 u[MIDDLE];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquareGF31 that launched dependents before writing GF31 data

  readMiddleOutLine(u, in31, y, x);

  middleMul(u, x, trig31);

  fft_MIDDLE(u);

  middleMul2(u, y, x, trig31);

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z31 lds[OUT_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
  out31 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out31 += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out31, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(OUT_WG) fftMiddleOutGF61(P(T2) out, CP(T2) in, Trig trig) {
  GF61 u[MIDDLE];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquare61 that launched dependents before writing GF61 data

  readMiddleOutLine(u, in61, y, x);

  middleMul(u, x, trig61);

  fft_MIDDLE(u);

  middleMul2(u, y, x, trig61);

  dependentLaunch();       // Next kernel will be carryfused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z61 lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
  out61 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out61 += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out61, u, gy, gx);
}

#if defined(AEVUM_APPLE_OPENCL12)

// The stock GF61 middle-out kernel is too large for cl2Metal when MIDDLE=8.
// Four stages are fully scalar and flatten MIDDLE into the group id.  Only the
// isolated fft_MIDDLE stage retains GF61 u[MIDDLE], a pipeline shape already
// accepted by Apple for fftMiddleInGF61FftApple.  The existing raw GF61 scratch
// plane holds exactly one transform and no additional allocation is required.
inline u32 appleMiddleOutGF61TmpIndex(u32 g, u32 k, u32 me) {
  return g * MIDDLE * OUT_WG + k * OUT_WG + me;
}

KERNEL(OUT_WG) fftMiddleOutGF61LoadScalarApple(P(GF61) tmp,
                                               CP(T2) in) {
#if PAD_SIZE != 0
#error Apple scalar GF61 middle-out requires PAD_SIZE=0
#endif
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  const u32 flatGroup = get_group_id(0);
  const u32 k = flatGroup % MIDDLE;
  const u32 g = flatGroup / MIDDLE;
  const u32 sizeY = OUT_WG / OUT_SIZEX;
  const u32 groupsX = SMALL_HEIGHT / OUT_SIZEX;
  const u32 gx = g % groupsX;
  const u32 gy = g / groupsX;
  const u32 me = get_local_id(0);
  const u32 mx = me % OUT_SIZEX;
  const u32 my = me / OUT_SIZEX;
  const u32 x = gx * OUT_SIZEX + mx;
  const u32 y = gy * sizeY + my;

  dependentLaunchWait();
  tmp[appleMiddleOutGF61TmpIndex(g, k, me)] =
      in61[(u64) y * MIDDLE * SMALL_HEIGHT + x + (u64) k * SMALL_HEIGHT];
}

KERNEL(OUT_WG) fftMiddleOutGF61MulScalarApple(P(GF61) tmp,
                                              Trig trig) {
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);
  const u32 flatGroup = get_group_id(0);
  const u32 k = flatGroup % MIDDLE;
  if (k == 0) return;
  const u32 g = flatGroup / MIDDLE;
  const u32 groupsX = SMALL_HEIGHT / OUT_SIZEX;
  const u32 gx = g % groupsX;
  const u32 me = get_local_id(0);
  const u32 mx = me % OUT_SIZEX;
  const u32 x = gx * OUT_SIZEX + mx;
  GF61 value = tmp[appleMiddleOutGF61TmpIndex(g, k, me)];

#if !MIDDLE_CHAIN
  value = cmul(value, TFLOAD(&trig61[x + (k - 1) * SMALL_HEIGHT]));
#else
  GF61 w = TFLOAD(&trig61[x]);
  GF61 factor = U2((u64) 1, (u64) 0);
  GF61 power = w;
  if (k & 1u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 2u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 4u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 8u) factor = cmul(factor, power);
  value = cmul(value, factor);
#endif
  tmp[appleMiddleOutGF61TmpIndex(g, k, me)] = value;
}

KERNEL(OUT_WG) fftMiddleOutGF61FftApple(P(GF61) tmp) {
  GF61 u[MIDDLE];
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  for (u32 k = 0; k < MIDDLE; ++k)
    u[k] = tmp[appleMiddleOutGF61TmpIndex(g, k, me)];
  fft_MIDDLE(u);
  for (u32 k = 0; k < MIDDLE; ++k)
    tmp[appleMiddleOutGF61TmpIndex(g, k, me)] = u[k];
}

KERNEL(OUT_WG) fftMiddleOutGF61Mul2ScalarApple(P(GF61) tmp,
                                               Trig trig) {
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);
  const u32 flatGroup = get_group_id(0);
  const u32 k = flatGroup % MIDDLE;
  const u32 g = flatGroup / MIDDLE;
  const u32 sizeY = OUT_WG / OUT_SIZEX;
  const u32 groupsX = SMALL_HEIGHT / OUT_SIZEX;
  const u32 gx = g % groupsX;
  const u32 gy = g / groupsX;
  const u32 me = get_local_id(0);
  const u32 mx = me % OUT_SIZEX;
  const u32 my = me / OUT_SIZEX;
  const u32 x = gx * OUT_SIZEX + mx;
  const u32 y = gy * sizeY + my;

  TrigGF61 trig1 = trig61 + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF61 trig2 = trig1 + WIDTH;
  if (WIDTH == SMALL_HEIGHT) trig1 = trig61;
  GF61 w = TFLOAD(&trig1[x]);
  const u32 desiredRoot = x * y;
  GF61 factor = cmul(TFLOAD(&trig2[desiredRoot % SMALL_HEIGHT]),
                     TFLOAD(&trig1[desiredRoot / SMALL_HEIGHT]));

  GF61 power = w;
  if (k & 1u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 2u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 4u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 8u) factor = cmul(factor, power);

  const u32 index = appleMiddleOutGF61TmpIndex(g, k, me);
  tmp[index] = cmul(tmp[index], factor);
}

KERNEL(OUT_WG) fftMiddleOutGF61WriteScalarApple(P(T2) out,
                                                CP(GF61) tmp) {
#if PAD_SIZE != 0
#error Apple scalar GF61 middle-out requires PAD_SIZE=0
#endif
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  const u32 flatGroup = get_group_id(0);
  const u32 k = flatGroup % MIDDLE;
  const u32 g = flatGroup / MIDDLE;
  const u32 sizeY = OUT_WG / OUT_SIZEX;
  const u32 groupsX = SMALL_HEIGHT / OUT_SIZEX;
  const u32 gx = g % groupsX;
  const u32 gy = g / groupsX;
  const u32 me = get_local_id(0);
  const u32 transposedMe =
      (me % OUT_SIZEX) * sizeY + me / OUT_SIZEX;
  const u64 outputIndex =
      (u64) gy * MIDDLE * OUT_WG +
      (u64) gx * MIDDLE * WIDTH * OUT_SIZEX +
      (u64) k * OUT_WG + transposedMe;

  dependentLaunch();
  out61[outputIndex] = tmp[appleMiddleOutGF61TmpIndex(g, k, me)];
}

// Signature-compatible placeholder used instead of the rejected monolithic
// kernel on Apple large plans.
KERNEL(OUT_WG) fftMiddleOutGF61ApplePlaceholder(P(T2) out,
                                                 CP(T2) in, Trig trig) {
  (void) out;
  (void) in;
  (void) trig;
}

#endif  // AEVUM_APPLE_OPENCL12

#endif



#else           // in place transpose

#if FFT_FP64

KERNEL(256) fftMiddleOut(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  T2 u[MIDDLE];

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

  dependentLaunchWait();   // Previous kernel was tailSquareFP64 that launched dependents before writing FP64 data

  readMiddleOutLine(u, in, y, x);

  middleMul(u, x, trig);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

  middleMul2(u, y, x, factor, trig);

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local T2 lds[256];
  middleShuffle(lds, u);

  writeMiddleOutLine(out, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if FFT_FP32

KERNEL(256) fftMiddleOut(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  F2 u[MIDDLE];

  P(F2) inF2 = (P(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

  dependentLaunchWait();   // Previous kernel was tailSquareFP32 that launched dependents before writing FP64 data

  readMiddleOutLine(u, inF2, y, x);

  middleMul(u, x, trigF2);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS * 2).
  const float factor = 1.0f / (NWORDS * 2);

  middleMul2(u, y, x, factor, trigF2);

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local F2 lds[256];
  middleShuffle(lds, u);

  writeMiddleOutLine(outF2, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(256) fftMiddleOutGF31(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  GF31 u[MIDDLE];

  P(GF31) in31 = (P(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

  dependentLaunchWait();   // Previous kernel was tailSquareGF31 that launched dependents before writing GF31 data

  readMiddleOutLine(u, in31, y, x);

  middleMul(u, x, trig31);

  fft_MIDDLE(u);

  middleMul2(u, y, x, trig31);

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local GF31 lds[256];
  middleShuffle(lds, u);

  writeMiddleOutLine(out31, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(256) fftMiddleOutGF61(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  GF61 u[MIDDLE];

  P(GF61) in61 = (P(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

  dependentLaunchWait();   // Previous kernel was tailSquareGF61 that launched dependents before writing GF61 data

  readMiddleOutLine(u, in61, y, x);

  middleMul(u, x, trig61);

  fft_MIDDLE(u);

  middleMul2(u, y, x, trig61);

  dependentLaunch();       // Next kernel will be carryfused which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local GF61 lds[256];
  middleShuffle(lds, u);

  writeMiddleOutLine(out61, u, y, x);
}

#endif

#endif
