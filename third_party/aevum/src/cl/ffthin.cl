// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftheight.cl"
#include "middle.cl"

#if FFT_FP64

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[LDS_BYTES / sizeof(T2)];

  T2 u[NH];
  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  readTailFusedLine(in, u, g, me);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrig, w, 1, me);

  write(G_H, NH, u, out, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local F2 lds[LDS_BYTES / sizeof(F2)];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH];
  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  readTailFusedLine(inF2, u, g, me);

#if NH == 8
  F2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  F2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrigF2, 1, me);

  write(G_H, NH, u, outF2, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHinGF31(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[LDS_BYTES / sizeof(GF31)];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH];
  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  readTailFusedLine(in31, u, g, me);

  fft_HEIGHT(lds, u, smallTrig31, 1, me);

  write(G_H, NH, u, out31, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHinGF61(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF61 lds[LDS_BYTES / sizeof(GF61)];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH];
  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  readTailFusedLine(in61, u, g, me);

  fft_HEIGHT(lds, u, smallTrig61, 1, me);

  write(G_H, NH, u, out61, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}


#if defined(AEVUM_APPLE_OPENCL12)

// Apple OpenCL-to-Metal rejects the stock fftHinGF61 pipeline at
// clCreateKernel.  These kernels are a global-memory decomposition of the
// exact same fft_HEIGHT operation.  The data remains FFT3161 GF(M61^2), with
// the same twiddles, radix order and final layout.  Only the work is split
// across several smaller kernels on Apple.
inline u32 appleFftHinGF61LineBase(u32 line) {
  return SMALL_HEIGHT * transPos(line, MIDDLE, WIDTH);
}

inline u32 appleFftHinGF61Index(u32 line, u32 slot, u32 me) {
  return appleFftHinGF61LineBase(line) + slot * G_H + me;
}

// Scalar equivalent of readTailFusedLine for PAD=0.  One workgroup handles
// one private-vector slot of one line and writes directly into the final
// fully-height-transformed layout used by tailMulLow.
KERNEL(G_H) fftHinGF61LoadScalarApple(P(T2) out, CP(T2) in) {
#if PAD_SIZE != 0
#error Apple staged fftHinGF61 requires PAD_SIZE=0
#endif
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  const u32 group = get_group_id(0);
  const u32 line = group / NH;
  const u32 slot = group - line * NH;
  const u32 me = get_local_id(0);
  const u32 sizeY = IN_WG / IN_SIZEX;
  const u32 x = line % WIDTH;
  const u32 chunkX = x / IN_SIZEX;
  const u32 xWithin = x % IN_SIZEX;
  const u32 middleI = line / WIDTH;
  const u32 y = slot * G_H + me;
  const u32 chunkY = y / sizeY;
  const u32 src = chunkX * (SMALL_HEIGHT * MIDDLE * IN_SIZEX) +
                  xWithin * sizeY + middleI * IN_WG + (me % sizeY) +
                  chunkY * (MIDDLE * IN_WG);
  out61[appleFftHinGF61Index(line, slot, me)] = in61[src];
}

// First/following private radix of fft_common.
KERNEL(G_H) fftHinGF61FftRadixApple(P(T2) data) {
  P(GF61) data61 = (P(GF61)) (data + DISTGF61);
  const u32 line = get_group_id(0);
  const u32 me = get_local_id(0);
  GF61 u[NH];
  for (u32 i = 0; i < NH; ++i) u[i] = data61[appleFftHinGF61Index(line, i, me)];
  fft_RADIX(u);
  for (u32 i = 0; i < NH; ++i) data61[appleFftHinGF61Index(line, i, me)] = u[i];
}

// Exact scalar tabMul from fft_common for one line/slot.
KERNEL(G_H) fftHinGF61FftTwiddleApple(P(T2) data, u32 f, Trig smallTrig) {
  P(GF61) data61 = (P(GF61)) (data + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);
  const u32 group = get_group_id(0);
  const u32 line = group / NH;
  const u32 i = group - line * NH;
  if (i == 0) return;
  const u32 me = get_local_id(0);
  const u32 p = me & ~(f - 1);
  const u32 index = appleFftHinGF61Index(line, i, me);
  GF61 value = data61[index];
#if TABMUL_CHAIN61
  GF61 w = TFLOAD(&smallTrig61[p]);
  GF61 multiplier = w;
  for (u32 j = 1; j < i; ++j) multiplier = cmul(multiplier, w);
  value = cmul(value, multiplier);
#else
  value = cmul(value, TFLOAD(&smallTrig61[(i - 1) * G_H + p]));
#endif
  data61[index] = value;
}

// Race-free global equivalent of shufl.  Source and destination are distinct.
KERNEL(G_H) fftHinGF61FftShuffleApple(CP(T2) src, P(T2) dst, u32 f) {
  CP(GF61) src61 = (CP(GF61)) (src + DISTGF61);
  P(GF61) dst61 = (P(GF61)) (dst + DISTGF61);
  const u32 group = get_group_id(0);
  const u32 line = group / NH;
  const u32 outI = group - line * NH;
  const u32 outMe = get_local_id(0);
  const u32 logical = outI * G_H + outMe;
  const u32 remainder = logical & (f - 1);
  const u32 srcI = (logical / f) % NH;
  const u32 srcMe = (logical / (f * NH)) * f + remainder;
  dst61[appleFftHinGF61Index(line, outI, outMe)] =
      src61[appleFftHinGF61Index(line, srcI, srcMe)];
}

// Final radix and normalization into the requested output buffer.
KERNEL(G_H) fftHinGF61FftFinalApple(CP(T2) src, P(T2) out) {
  CP(GF61) src61 = (CP(GF61)) (src + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  const u32 line = get_group_id(0);
  const u32 me = get_local_id(0);
  GF61 u[NH];
  for (u32 i = 0; i < NH; ++i) u[i] = src61[appleFftHinGF61Index(line, i, me)];
  fft_RADIX(u);
  for (u32 i = 0; i < NH; ++i) out61[appleFftHinGF61Index(line, i, me)] = u[i];
}

// The host instantiates this instead of the stock kernel on Apple so Metal is
// never asked to create the rejected monolithic pipeline.
KERNEL(G_H) fftHinGF61ApplePlaceholder(P(T2) out, CP(T2) in, Trig smallTrig) {
  
}

#endif

#endif
