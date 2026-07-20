// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "fft-middle.cl"
#include "middle.cl"

#if !INPLACE                  // Original implementation (not in place)

#if FFT_FP64

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, Trig trig) {
  T2 u[MIDDLE];

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

#if FFT_TYPE == FFT64
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP64 data
#endif

  readMiddleInLine(u, in, y, x);

  middleMul2(u, x, y, 1, trig);

  fft_MIDDLE(u);

  middleMul(u, y, trig);

  dependentLaunch();       // Next kernel will be tailSquareFP64 which must dependentLaunchWait before reading data

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local T lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  out += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out += mx * SIZEY + my;
#endif

  writeMiddleInLine(out, u, gy, gx);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, Trig trig) {
  F2 u[MIDDLE];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

#if FFT_TYPE == FFT32
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP32 data
#endif

  readMiddleInLine(u, inF2, y, x);

#if PFA_RADIX
  fft_MIDDLE(u);
#else
  middleMul2(u, x, y, 1, trigF2);
  fft_MIDDLE(u);
  middleMul(u, y, trigF2);
#endif

  dependentLaunch();       // Next kernel will be tailSquareFP32 which must dependentLaunchWait before reading data

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local F lds[IN_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  outF2 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  outF2 += mx * SIZEY + my;
#endif

  writeMiddleInLine(outF2, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(IN_WG) fftMiddleInGF31(P(T2) out, CP(T2) in, Trig trig) {
  GF31 u[MIDDLE];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;
  
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

#if FFT_TYPE == FFT31
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in31, y, x);

#if PFA_RADIX
  fft_MIDDLE(u);
#else
  middleMul2(u, x, y, trig31);
  fft_MIDDLE(u);
  middleMul(u, y, trig31);
#endif

  dependentLaunch();       // Next kernel will be tailSquareGF31 which must dependentLaunchWait before reading data

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z31 lds[IN_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  out31 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out31 += mx * SIZEY + my;
#endif

  writeMiddleInLine(out31, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(IN_WG) fftMiddleInGF61(P(T2) out, CP(T2) in, Trig trig) {
  GF61 u[MIDDLE];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;
  
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  u32 x = startx + mx;
  u32 y = starty + my;

#if FFT_TYPE == FFT61
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in61, y, x);

#if PFA_RADIX
  fft_MIDDLE(u);
#else
  middleMul2(u, x, y, trig61);
  fft_MIDDLE(u);
  middleMul(u, y, trig61);
#endif

  dependentLaunch();       // Next kernel will be tailSquareGF61 which must dependentLaunchWait before reading data

#if MIDDLE_IN_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z61 lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  out61 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out61 += mx * SIZEY + my;
#endif

  writeMiddleInLine(out61, u, gy, gx);
}

#endif


#if NTT_GF61 && defined(AEVUM_APPLE_OPENCL12)

// Apple's OpenCL-to-Metal compiler rejects the complete GF61 middle-in
// pipeline even though its individual arithmetic and LDS operations are
// supported. Keep the stock kernel above for every non-Apple backend and
// decompose only the Apple OpenCL 1.2 path into in-order global stages.
// The final stage retains the original LDS transpose exactly.

u64 appleMiddleInGF61BlockOffset(u32 chunk_y, u32 chunk_x) {
#if PAD_SIZE > 0
  u64 BIG_PAD_SIZE = (PAD_SIZE / 2 + 1) * PAD_SIZE;
  u32 SIZEY = IN_WG / IN_SIZEX;
  return (u64) chunk_y * (MIDDLE * IN_WG + PAD_SIZE) +
         (u64) chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX +
                           SMALL_HEIGHT / SIZEY * PAD_SIZE + BIG_PAD_SIZE);
#else
  return (u64) chunk_y * MIDDLE * IN_WG +
         (u64) chunk_x * MIDDLE * SMALL_HEIGHT * IN_SIZEX;
#endif
}

void appleReadMiddleInGF61Block(GF61 *u, CP(GF61) in61, u32 chunk_y, u32 chunk_x) {
  in61 += appleMiddleInGF61BlockOffset(chunk_y, chunk_x) + get_local_id(0);
  for (int i = 0; i < MIDDLE; ++i) { u[i] = in61[i * IN_WG]; }
}

void appleWriteMiddleInGF61Block(P(GF61) out61, GF61 *u, u32 chunk_y, u32 chunk_x) {
  out61 += appleMiddleInGF61BlockOffset(chunk_y, chunk_x) + get_local_id(0);
  for (int i = 0; i < MIDDLE; ++i) { out61[i * IN_WG] = u[i]; }
}

GF61 appleReadMiddleInGF61Scalar(CP(GF61) in61, u32 y, u32 x, u32 k) {
#if PAD_SIZE > 0
  u64 BIG_PAD_SIZE = (PAD_SIZE / 2 + 1) * PAD_SIZE;
  u64 line = (u64) y * WIDTH + (u64) y * PAD_SIZE +
             (u64) (y / SMALL_HEIGHT) * BIG_PAD_SIZE + x;
  u64 stride = (u64) SMALL_HEIGHT * (WIDTH + PAD_SIZE) + BIG_PAD_SIZE;
  return FFTLOAD(&in61[line + (u64) k * stride]);
#else
  return FFTLOAD(&in61[(u64) y * WIDTH + x + (u64) k * SMALL_HEIGHT * WIDTH]);
#endif
}

GF61 appleReadMiddleInGF61BlockScalar(CP(GF61) in61, u32 chunk_y, u32 chunk_x, u32 k) {
  u64 offset = appleMiddleInGF61BlockOffset(chunk_y, chunk_x) +
               (u64) k * IN_WG + get_local_id(0);
  return in61[offset];
}

void appleWriteMiddleInGF61BlockScalar(P(GF61) out61, GF61 value,
                                       u32 chunk_y, u32 chunk_x, u32 k) {
  u64 offset = appleMiddleInGF61BlockOffset(chunk_y, chunk_x) +
               (u64) k * IN_WG + get_local_id(0);
  out61[offset] = value;
}

// The Apple compiler rejected the previous combined load+middleMul2 kernel.
// Flatten MIDDLE into the group index so each work-item owns one scalar GF61
// value instead of a private GF61 u[MIDDLE] array.
KERNEL(IN_WG) fftMiddleInGF61LoadScalarApple(P(T2) out, CP(T2) in) {
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);

  u32 flat_group = get_group_id(0);
  u32 k = flat_group % MIDDLE;
  u32 g = flat_group / MIDDLE;
  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 N = WIDTH / IN_SIZEX;
  u32 gx = g % N;
  u32 gy = g / N;
  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;
  u32 x = gx * IN_SIZEX + mx;
  u32 y = gy * SIZEY + my;

#if FFT_TYPE == FFT61
  dependentLaunchWait();
#endif
  GF61 value = appleReadMiddleInGF61Scalar(in61, y, x, k);
  appleWriteMiddleInGF61BlockScalar(out61, value, gy, gx, k);
}

// Generate only the middleMul2 factor. The original source buffer is no longer
// needed after the scalar load, so the host stores these factors in that scratch.
KERNEL(IN_WG) fftMiddleInGF61Mul2FactorScalarApple(P(T2) out, Trig trig) {
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 flat_group = get_group_id(0);
  u32 k = flat_group % MIDDLE;
  u32 g = flat_group / MIDDLE;
  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 N = WIDTH / IN_SIZEX;
  u32 gx = g % N;
  u32 gy = g / N;
  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;
  u32 x = gx * IN_SIZEX + mx;
  u32 y = gy * SIZEY + my;

  TrigGF61 trig1 = trig61 + SMALL_HEIGHT * (MIDDLE - 1);
  TrigGF61 trig2 = trig1 + WIDTH;
  if (WIDTH == SMALL_HEIGHT) trig1 = trig61;

  GF61 w = TFLOAD(&trig1[x]);
  u32 desired_root = x * y;
  GF61 factor = cmul(TFLOAD(&trig2[desired_root % SMALL_HEIGHT]),
                     TFLOAD(&trig1[desired_root / SMALL_HEIGHT]));

  // k < 16 for every supported middle size. Fixed binary powering avoids a
  // variable-length loop in Apple's OpenCL-to-Metal compiler.
  GF61 power = w;
  if (k & 1u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 2u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 4u) factor = cmul(factor, power);
  power = cmul(power, power);
  if (k & 8u) factor = cmul(factor, power);

  appleWriteMiddleInGF61BlockScalar(out61, factor, gy, gx, k);
}

// In-place element-wise apply stage.  P()/CP() expand to restrict-qualified
// pointers, therefore the former (out, data, factor) signature was undefined
// when Apple dispatch passed the same cl_mem for out and data.  A single io
// pointer states the real aliasing contract and prevents Metal from optimizing
// the read under a false no-alias assumption.
KERNEL(IN_WG) fftMiddleInGF61ApplyScalarApple(P(T2) io, CP(T2) factor) {
  P(GF61) io61 = (P(GF61)) (io + DISTGF61);
  CP(GF61) factor61 = (CP(GF61)) (factor + DISTGF61);

  u32 flat_group = get_group_id(0);
  u32 k = flat_group % MIDDLE;
  u32 g = flat_group / MIDDLE;
  u32 N = WIDTH / IN_SIZEX;
  u32 gx = g % N;
  u32 gy = g / N;

  GF61 value = appleReadMiddleInGF61BlockScalar(io61, gy, gx, k);
  GF61 w = appleReadMiddleInGF61BlockScalar(factor61, gy, gx, k);
  appleWriteMiddleInGF61BlockScalar(io61, cmul(value, w), gy, gx, k);
}

KERNEL(IN_WG) fftMiddleInGF61FftApple(P(T2) out, CP(T2) in) {
  GF61 u[MIDDLE];
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  u32 N = WIDTH / IN_SIZEX;
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;
  appleReadMiddleInGF61Block(u, in61, gy, gx);
  fft_MIDDLE(u);
  appleWriteMiddleInGF61Block(out61, u, gy, gx);
}

// In-place post-middle multiply.  As above, use one restrict-qualified
// pointer because Apple dispatch intentionally reads and writes the same plane.
KERNEL(IN_WG) fftMiddleInGF61MulApple(P(T2) io, Trig trig) {
  GF61 u[MIDDLE];
  P(GF61) io61 = (P(GF61)) (io + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 N = WIDTH / IN_SIZEX;
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;
  u32 me = get_local_id(0);
  u32 my = me / IN_SIZEX;
  u32 y = gy * SIZEY + my;

  appleReadMiddleInGF61Block(u, io61, gy, gx);
  middleMul(u, y, trig61);
  dependentLaunch();
  appleWriteMiddleInGF61Block(io61, u, gy, gx);
}

KERNEL(IN_WG) fftMiddleInGF61TransposeApple(P(T2) out, CP(T2) in) {
  GF61 u[MIDDLE];
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  u32 N = WIDTH / IN_SIZEX;
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  appleReadMiddleInGF61Block(u, in61, gy, gx);

  // Keep the stock GF61 LDS transpose. Splitting the arithmetic from this
  // stage reduces Metal pipeline complexity without changing the algorithm.
  local Z61 lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  appleWriteMiddleInGF61Block(out61, u, gy, gx);
}

#endif






#else           // in place transpose

#if FFT_FP64

KERNEL(256) fftMiddleIn(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  T2 u[MIDDLE];

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = g / 131072;                       // A super tiny benefit (much smaller than margin of error) on TitanV
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT64
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP64 data
#endif

  readMiddleInLine(u, in, y, x);

  middleMul2(u, x, y, 1, trig);

  fft_MIDDLE(u);

  middleMul(u, y, trig);

  dependentLaunch();       // Next kernel will be tailSquareFP64 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local T2 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(in + zerohack, u, y, x);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

KERNEL(256) fftMiddleIn(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  F2 u[MIDDLE];

  P(F2) inF2 = (P(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16 (for FP64, FP32 untimed)
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT32
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing FP32 data
#endif

  readMiddleInLine(u, inF2, y, x);

#if PFA_RADIX
  fft_MIDDLE(u);
#else
  middleMul2(u, x, y, 1, trigF2);
  fft_MIDDLE(u);
  middleMul(u, y, trigF2);
#endif

  dependentLaunch();       // Next kernel will be tailSquareFP32 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local F2 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(inF2 + zerohack, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(256) fftMiddleInGF31(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  GF31 u[MIDDLE];

  P(GF31) in31 = (P(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16 (for FP64, GF31 untimed)
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT31
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in31, y, x);

#if PFA_RADIX
  fft_MIDDLE(u);
#else
  middleMul2(u, x, y, trig31);
  fft_MIDDLE(u);
  middleMul(u, y, trig31);
#endif

  dependentLaunch();       // Next kernel will be tailSquareGF31 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local GF31 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(in31 + zerohack, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(256) fftMiddleInGF61(P(T2) out, P(T2) in, Trig trig) {
  assert(out == in);
  GF61 u[MIDDLE];

  P(GF61) in61 = (P(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 g = get_group_id(0);
#if INPLACE == 1                                   // nVidia friendly padding
  u32 N = SMALL_HEIGHT / 16;
  u32 starty = g % N * 16;
  u32 startx = g / N * 16;
  u32 zerohack = 0;                                // Need to test if g / 131072 is of any benefit
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / 16;
  u32 startx = g % N * 16;
  u32 starty = g / N * 16;
  u32 zerohack = (MIDDLE >= 16) ? 0 : g / 131072;  // Rocm optimizer goes bonkers if zerohack used when MIDDLE=16 (for FP64, GF61 untimed)
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % 16;
  u32 y = starty + me / 16;

#if FFT_TYPE == FFT61
  dependentLaunchWait();   // Previous kernel was carryfused that launched dependents before writing GF31 data
#endif

  readMiddleInLine(u, in61, y, x);

#if PFA_RADIX
  fft_MIDDLE(u);
#else
  middleMul2(u, x, y, trig61);
  fft_MIDDLE(u);
  middleMul(u, y, trig61);
#endif

  dependentLaunch();       // Next kernel will be tailSquareGF61 which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local GF61 lds[256];
  middleShuffle(lds, u);

  writeMiddleInLine(in61 + zerohack, u, y, x);
}

#endif

#endif
