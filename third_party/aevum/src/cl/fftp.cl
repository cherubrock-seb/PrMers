// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftwidth.cl"
#include "weight.cl"
#include "middle.cl"

#if FFT_TYPE == FFT64

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTab THREAD_WEIGHTS) {
  local T2 lds[LDS_BYTES / sizeof(T2)];
  T2 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  for (u32 i = 0; i < NW; ++i) {
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x * w1, in[p].y * w2);
  }

  fft_WIDTH(lds, u, smallTrig, 1, me);

  writeCarryFusedLine(u, out, g, me);
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(F2) out, CP(Word2) in, TrigFP32 smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local F2 lds[LDS_BYTES / sizeof(F2)];
  F2 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  u32 me_frac_bits = fracBits(me * BIG_HEIGHT * 2);
  u32 line_frac_bits = fracBits(g * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y), base_frac_bits > line_frac_bits);

  const u32 frac_bits_bigstep = fracBits(G_W * BIG_HEIGHT * 2);

  u32 frac_bits = base_frac_bits;
  for (u32 i = 0; i < NW; ++i) {
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x * w1, in[p].y * w2);
    // Generate frac_bits for next pair
    frac_bits += frac_bits_bigstep;
  }

  fft_WIDTH(lds, u, smallTrig, 1, me);

  writeCarryFusedLine(u, out, g, me);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(GF31) out, CP(Word2) in, TrigGF31 smallTrig) {
  local GF31 lds[LDS_BYTES / sizeof(GF31)];
  GF31 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 31;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Convert and weight inputs
    u[i] = U2(shl(make_Z31(in[p].x), weight_shift0), shl(make_Z31(in[p].y), weight_shift1));      // Form a GF31 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH(lds, u, smallTrig, 1, me);

  writeCarryFusedLine(u, out, g, me);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(GF61) out, CP(Word2) in, TrigGF61 smallTrig) {
  local GF61 lds[LDS_BYTES / sizeof(GF61)];
  GF61 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits	combo.a[0]
#define weight_shift	combo.a[1]
#define combo_counter	combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    u[i] = U2(shl(make_Z61(in[p].x), weight_shift0), shl(make_Z61(in[p].y), weight_shift1));      // Form a GF61 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  fft_WIDTH(lds, u, smallTrig, 1, me);

  writeCarryFusedLine(u, out, g, me);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTab THREAD_WEIGHTS) {
  local T2 lds[LDS_BYTES / sizeof(T2)];
  local GF31 *lds31 = (local GF31 *) lds;
  T2 u[NW];
  GF31 u31[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

  in += g * WIDTH;

  T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 31;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP64 weights and the second GF31 weight shift
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    u[i] = U2(in[p].x * w1, in[p].y * w2);
    u31[i] = U2(shl(make_Z31(in[p].x), weight_shift0), shl(make_Z31(in[p].y), weight_shift1));      // Form a GF31 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH(lds, u, smallTrig, 1, me);
  writeCarryFusedLine(u, out, g, me);

  fft_WIDTH(lds31, u31, smallTrig31, 1, me);
  writeCarryFusedLine(u31, out31, g, me);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local F2 ldsF2[LDS_BYTES / sizeof(F2)];
  local GF31 *lds31 = (local GF31 *) ldsF2;
  F2 uF2[NW];
  GF31 u31[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  u32 me_frac_bits = fracBits(me * BIG_HEIGHT * 2);
  u32 line_frac_bits = fracBits(g * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y), base_frac_bits > line_frac_bits);

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 31;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP32 weights and the second GF31 weight shift
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    uF2[i] = U2(in[p].x * w1, in[p].y * w2);
    u31[i] = U2(shl(make_Z31(in[p].x), weight_shift0), shl(make_Z31(in[p].y), weight_shift1));      // Form a GF31 from each pair of input words

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH(ldsF2, uF2, smallTrigF2, 1, me);
  writeCarryFusedLine(uF2, outF2, g, me);

  fft_WIDTH(lds31, u31, smallTrig31, 1, me);
  writeCarryFusedLine(u31, out31, g, me);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local GF61 lds61[LDS_BYTES / sizeof(GF61)];
  local F2 *ldsF2 = (local F2 *) lds61;
  F2 uF2[NW];
  GF61 u61[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  u32 me_frac_bits = fracBits(me * BIG_HEIGHT * 2);
  u32 line_frac_bits = fracBits(g * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y), base_frac_bits > line_frac_bits);

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  weight_shift = weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP32 weights and the second GF61 weight shift
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    uF2[i] = U2(in[p].x * w1, in[p].y * w2);
    u61[i] = U2(shl(make_Z61(in[p].x), weight_shift0), shl(make_Z61(in[p].y), weight_shift1));      // Form a GF61 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  fft_WIDTH(ldsF2, uF2, smallTrigF2, 1, me);
  writeCarryFusedLine(uF2, outF2, g, me);

  fft_WIDTH(lds61, u61, smallTrig61, 1, me);
  writeCarryFusedLine(u61, out61, g, me);
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161


#if defined(AEVUM_APPLE_SPLIT_FFTP)

// Apple OpenCL lowers every kernel entry point separately to a Metal compute
// pipeline. The combined FFT3161 fftP entry is too large for the legacy Apple
// translator even though clBuildProgram accepts the source. Keep the exact
// same data layout but expose one compact entry point per CRT modulus.
KERNEL(G_W) fftP31Apple(P(ulong2) outRaw, CP(Word2) in, global const ulong2 *trigRaw) {
  local GF31 lds31[LDS_BYTES / sizeof(GF31)];
  GF31 u31[NW];

  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);

  P(GF31) out31 = (P(GF31)) (outRaw + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (trigRaw + DISTWTRIGGF31);
  in += g * WIDTH;

  const u32 word_index = (me * BIG_HEIGHT + g) * 2;
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  union { uint2 a; u64 b; } combo;
  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep =
      (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) +
       make_u64((G_W * BIG_HEIGHT * 2 - 1) * bigword_weight_shift_minus1, 0)) %
      (31ULL << 32);
  combo.b = comboFracBits(word_index) +
            make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  combo.a[1] %= 31;

  for (u32 i = 0; i < NW; ++i) {
    const u32 p = G_W * i + me;
    const u32 shift0 = combo.a[1];
    combo.b += combo_step;
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
    const u32 shift1 = combo.a[1];
    u31[i] = U2(shl(make_Z31(in[p].x), shift0),
                shl(make_Z31(in[p].y), shift1));
    combo.b += combo_bigstep;
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
  }

  fft_WIDTH(lds31, u31, smallTrig31, 1, me);
  writeCarryFusedLine(u31, out31, g, me);
}


// The grouped GF31 fftP entry still uses LDS and is the only Apple fftP
// component whose result remained unstable in the M2 stage trace.  Use the
// same exact scalar weighting plus global radix/twiddle/shuffle decomposition
// as GF61.  Every destination is written exactly once, no local memory is used,
// and the final GF31 plane is returned to the original output buffer.
KERNEL(256) fftP31WeightScalarApple(P(ulong2) outRaw, CP(Word2) in) {
  const u32 p = get_global_id(0);
  const u32 line = p / WIDTH;
  const u32 x = p - line * WIDTH;
  const u32 word_index = (line + BIG_HEIGHT * x) * 2;

  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift =
      (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 =
      (bigword_weight_shift + 30) % 31;

  union { uint2 a; u64 b; } combo;
  combo.b = comboFracBits(word_index) +
            make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  combo.a[1] %= 31;
  const u32 shift0 = combo.a[1];

  combo.b += make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
  const u32 shift1 = combo.a[1];

  P(GF31) out31 = (P(GF31)) (outRaw + DISTGF31);
  out31[p] = U2(shl(make_Z31(in[p].x), shift0),
                 shl(make_Z31(in[p].y), shift1));
}

KERNEL(G_W) fftP31WidthRadixApple(P(ulong2) ioRaw) {
  GF31 u31[NW];
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  P(GF31) io31 = (P(GF31)) (ioRaw + DISTGF31) + g * WIDTH + me;

  for (u32 i = 0; i < NW; ++i) u31[i] = io31[i * G_W];
  fft_RADIX(u31);
  for (u32 i = 0; i < NW; ++i) io31[i * G_W] = u31[i];
}

#define DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(NAME, STAGE)                         \
KERNEL(G_W) NAME(P(ulong2) outRaw, CP(ulong2) inRaw,                           \
                 global const ulong2 *trigRaw) {                               \
  const u32 p = get_global_id(0);                                               \
  const u32 line = p / WIDTH;                                                   \
  const u32 x = p - line * WIDTH;                                               \
  const u32 i = x / G_W;                                                        \
  const u32 me = x - i * G_W;                                                   \
  CP(GF31) in31 = (CP(GF31)) (inRaw + DISTGF31);                               \
  P(GF31) out31 = (P(GF31)) (outRaw + DISTGF31);                               \
  TrigGF31 trig31 = (TrigGF31) (trigRaw + DISTWTRIGGF31);                      \
  GF31 v = in31[p];                                                             \
  const u32 mask = (STAGE) - 1;                                                 \
  const u32 trigBase = me & ~mask;                                              \
  if (i != 0) v = cmul(v, TFLOAD(&trig31[(i - 1) * G_W + trigBase]));          \
  const u32 dst = i * (STAGE) + (me & ~mask) * NW + (me & mask);               \
  out31[line * WIDTH + dst] = v;                                                \
}

DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle1Apple,   1)
DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle4Apple,   4)
DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle8Apple,   8)
DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle16Apple, 16)
DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle64Apple, 64)
DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle256Apple, 256)
DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE(fftP31TwiddleShuffle512Apple, 512)

#undef DEFINE_APPLE_GF31_TWIDDLE_SHUFFLE

// Apple v0.3.56: exact composition of WidthRadix + TwiddleShuffle.  Each
// work-item owns NW values, performs the private radix, applies the stage
// twiddles, then scatters to unique destinations in the alternate transform
// buffer.  No LDS or cross-work-item barrier is used.
#define DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(NAME, STAGE)                       \
KERNEL(G_W) NAME(P(ulong2) outRaw, CP(ulong2) inRaw,                           \
                 global const ulong2 *trigRaw) {                               \
  const u32 line = get_group_id(0);                                             \
  const u32 me = get_local_id(0);                                                \
  CP(GF31) in31 = (CP(GF31)) (inRaw + DISTGF31) + line * WIDTH;                 \
  P(GF31) out31 = (P(GF31)) (outRaw + DISTGF31) + line * WIDTH;                 \
  TrigGF31 trig31 = (TrigGF31) (trigRaw + DISTWTRIGGF31);                      \
  GF31 u[NW];                                                                   \
  for (u32 i = 0; i < NW; ++i) u[i] = in31[i * G_W + me];                      \
  fft_RADIX(u);                                                                 \
  const u32 mask = (STAGE) - 1;                                                 \
  const u32 trigBase = me & ~mask;                                              \
  for (u32 i = 0; i < NW; ++i) {                                                \
    GF31 v = u[i];                                                              \
    if (i != 0) v = cmul(v, TFLOAD(&trig31[(i - 1) * G_W + trigBase]));        \
    const u32 dst = i * (STAGE) + (me & ~mask) * NW + (me & mask);             \
    out31[dst] = v;                                                             \
  }                                                                             \
}

DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused1Apple,   1)
DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused4Apple,   4)
DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused8Apple,   8)
DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused16Apple, 16)
DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused64Apple, 64)
DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused256Apple, 256)
DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED(fftP31WidthStageFused512Apple, 512)

#undef DEFINE_APPLE_GF31_WIDTH_STAGE_FUSED

// More aggressive first stage: exact WeightScalar + fused stage-1 composition.
// It is preloaded separately; Apple can fall back to the stage-only path if its
// Metal pipeline compiler rejects the combined weighting arithmetic.
KERNEL(G_W) fftP31WeightStage1FusedApple(P(ulong2) outRaw, CP(Word2) in,
                                         global const ulong2 *trigRaw) {
  const u32 line = get_group_id(0);
  const u32 me = get_local_id(0);
  P(GF31) out31 = (P(GF31)) (outRaw + DISTGF31) + line * WIDTH;
  TrigGF31 trig31 = (TrigGF31) (trigRaw + DISTWTRIGGF31);
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift =
      (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 step = (bigword_weight_shift + 30) % 31;
  GF31 u[NW];
  for (u32 i = 0; i < NW; ++i) {
    const u32 x = i * G_W + me;
    const u32 p = line * WIDTH + x;
    const u32 word_index = (line + BIG_HEIGHT * x) * 2;
    union { uint2 a; u64 b; } combo;
    combo.b = comboFracBits(word_index) +
              make_u64(word_index * step, 0xFFFFFFFF);
    combo.a[1] %= 31;
    const u32 shift0 = combo.a[1];
    combo.b += make_u64(step, FRAC_BPW_HI);
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
    const u32 shift1 = combo.a[1];
    u[i] = U2(shl(make_Z31(in[p].x), shift0),
              shl(make_Z31(in[p].y), shift1));
  }
  fft_RADIX(u);
  for (u32 i = 0; i < NW; ++i) {
    GF31 v = u[i];
    if (i != 0) v = cmul(v, TFLOAD(&trig31[(i - 1) * G_W + me]));
    out31[i + me * NW] = v;
  }
}

KERNEL(G_W) fftP31WidthFinalApple(P(ulong2) outRaw, CP(ulong2) inRaw) {
  GF31 u31[NW];
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  CP(GF31) in31 = (CP(GF31)) (inRaw + DISTGF31) + g * WIDTH + me;
  P(GF31) out31 = (P(GF31)) (outRaw + DISTGF31) + g * WIDTH + me;

  for (u32 i = 0; i < NW; ++i) u31[i] = in31[i * G_W];
  fft_RADIX(u31);
  for (u32 i = 0; i < NW; ++i) out31[i * G_W] = u31[i];
}

// Apple's Metal pipeline compiler also rejects the grouped GF61 weighting
// kernel.  Use one work-item per Word2.  With PAD=0 (the Apple default), the
// carry-fused line index is exactly the flat input index.  The logical FFT word
// order is the transpose q = line + BIG_HEIGHT * x, where line=p/WIDTH and
// x=p%WIDTH.  This is algebraically identical to the grouped NW loop above but
// removes the private GF61[NW] array, the loop, and writeCarryFusedLine.
KERNEL(256) fftP61WeightScalarApple(P(ulong2) outRaw, CP(Word2) in) {
  const u32 p = get_global_id(0);          // Flat Word2 / carry-fused index.
  const u32 line = p / WIDTH;
  const u32 x = p - line * WIDTH;
  const u32 word_index = (line + BIG_HEIGHT * x) * 2;

  const u32 log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift =
      (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 =
      (bigword_weight_shift + 60) % 61;

  union { uint2 a; u64 b; } combo;
  combo.b = comboFracBits(word_index) +
            make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);
  combo.a[1] %= 61;
  const u32 shift0 = combo.a[1];

  combo.b += make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo.a[1] = adjust_m61_weight_shift(combo.a[1]);
  const u32 shift1 = combo.a[1];

  P(GF61) out61 = (P(GF61)) (outRaw + DISTGF61);
  out61[p] = U2(shl(make_Z61(in[p].x), shift0),
                 shl(make_Z61(in[p].y), shift1));
}

// On the tested Apple OpenCL-to-Metal stack, this particular fftP width stage
// exceeded the pipeline compiler when GF61 arithmetic, twiddles and LDS shuffle
// were all in one entry point. This is not a general prohibition on LDS: the
// middle/tail kernels keep their original LDS paths. Reproduce fft_common here
// with compact stages:
//
//   1. one in-place radix kernel operates on the NW values owned by a lane;
//   2. one scalar kernel applies the twiddle and writes the exact shufl
//      permutation into the alternate big buffer.
//
// The scalar destination below is the flat LDS index used by shufl's generic
// implementation.  Interpreting that flat index as [register][lane] gives
// exactly the u[] state consumed by the following radix stage.
KERNEL(G_W) fftP61WidthRadixApple(P(ulong2) ioRaw) {
  GF61 u61[NW];
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  P(GF61) io61 = (P(GF61)) (ioRaw + DISTGF61) + g * WIDTH + me;

  for (u32 i = 0; i < NW; ++i) u61[i] = io61[i * G_W];
  fft_RADIX(u61);
  for (u32 i = 0; i < NW; ++i) io61[i * G_W] = u61[i];
}

#define DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(NAME, STAGE)                         \
KERNEL(G_W) NAME(P(ulong2) outRaw, CP(ulong2) inRaw,                           \
                 global const ulong2 *trigRaw) {                               \
  const u32 p = get_global_id(0);                                               \
  const u32 line = p / WIDTH;                                                   \
  const u32 x = p - line * WIDTH;                                               \
  const u32 i = x / G_W;                                                        \
  const u32 me = x - i * G_W;                                                   \
  CP(GF61) in61 = (CP(GF61)) (inRaw + DISTGF61);                               \
  P(GF61) out61 = (P(GF61)) (outRaw + DISTGF61);                               \
  TrigGF61 trig61 = (TrigGF61) (trigRaw + DISTWTRIGGF61);                      \
  GF61 v = in61[p];                                                             \
  const u32 mask = (STAGE) - 1;                                                 \
  const u32 trigBase = me & ~mask;                                              \
  if (i != 0) v = cmul(v, TFLOAD(&trig61[(i - 1) * G_W + trigBase]));          \
  const u32 dst = i * (STAGE) + (me & ~mask) * NW + (me & mask);               \
  out61[line * WIDTH + dst] = v;                                                \
}

DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle1Apple,   1)
DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle4Apple,   4)
DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle8Apple,   8)
DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle16Apple, 16)
DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle64Apple, 64)
DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle256Apple, 256)
DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE(fftP61TwiddleShuffle512Apple, 512)

#undef DEFINE_APPLE_GF61_TWIDDLE_SHUFFLE

#define DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(NAME, STAGE)                       \
KERNEL(G_W) NAME(P(ulong2) outRaw, CP(ulong2) inRaw,                           \
                 global const ulong2 *trigRaw) {                               \
  const u32 line = get_group_id(0);                                             \
  const u32 me = get_local_id(0);                                                \
  CP(GF61) in61 = (CP(GF61)) (inRaw + DISTGF61) + line * WIDTH;                 \
  P(GF61) out61 = (P(GF61)) (outRaw + DISTGF61) + line * WIDTH;                 \
  TrigGF61 trig61 = (TrigGF61) (trigRaw + DISTWTRIGGF61);                      \
  GF61 u[NW];                                                                   \
  for (u32 i = 0; i < NW; ++i) u[i] = in61[i * G_W + me];                      \
  fft_RADIX(u);                                                                 \
  const u32 mask = (STAGE) - 1;                                                 \
  const u32 trigBase = me & ~mask;                                              \
  for (u32 i = 0; i < NW; ++i) {                                                \
    GF61 v = u[i];                                                              \
    if (i != 0) v = cmul(v, TFLOAD(&trig61[(i - 1) * G_W + trigBase]));        \
    const u32 dst = i * (STAGE) + (me & ~mask) * NW + (me & mask);             \
    out61[dst] = v;                                                             \
  }                                                                             \
}

DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused1Apple,   1)
DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused4Apple,   4)
DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused8Apple,   8)
DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused16Apple, 16)
DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused64Apple, 64)
DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused256Apple, 256)
DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED(fftP61WidthStageFused512Apple, 512)

#undef DEFINE_APPLE_GF61_WIDTH_STAGE_FUSED

KERNEL(G_W) fftP61WeightStage1FusedApple(P(ulong2) outRaw, CP(Word2) in,
                                         global const ulong2 *trigRaw) {
  const u32 line = get_group_id(0);
  const u32 me = get_local_id(0);
  P(GF61) out61 = (P(GF61)) (outRaw + DISTGF61) + line * WIDTH;
  TrigGF61 trig61 = (TrigGF61) (trigRaw + DISTWTRIGGF61);
  const u32 log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift =
      (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 step = (bigword_weight_shift + 60) % 61;
  GF61 u[NW];
  for (u32 i = 0; i < NW; ++i) {
    const u32 x = i * G_W + me;
    const u32 p = line * WIDTH + x;
    const u32 word_index = (line + BIG_HEIGHT * x) * 2;
    union { uint2 a; u64 b; } combo;
    combo.b = comboFracBits(word_index) +
              make_u64(word_index * step, 0xFFFFFFFF);
    combo.a[1] %= 61;
    const u32 shift0 = combo.a[1];
    combo.b += make_u64(step, FRAC_BPW_HI);
    combo.a[1] = adjust_m61_weight_shift(combo.a[1]);
    const u32 shift1 = combo.a[1];
    u[i] = U2(shl(make_Z61(in[p].x), shift0),
              shl(make_Z61(in[p].y), shift1));
  }
  fft_RADIX(u);
  for (u32 i = 0; i < NW; ++i) {
    GF61 v = u[i];
    if (i != 0) v = cmul(v, TFLOAD(&trig61[(i - 1) * G_W + me]));
    out61[i + me * NW] = v;
  }
}

// Final radix can write to a different big buffer.  This avoids an extra copy
// when the odd number of width stages leaves the current GF61 state in the
// alternate buffer (WIDTH=256, NW=4 on the M2 smoke plan).
KERNEL(G_W) fftP61WidthFinalApple(P(ulong2) outRaw, CP(ulong2) inRaw) {
  GF61 u61[NW];
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  CP(GF61) in61 = (CP(GF61)) (inRaw + DISTGF61) + g * WIDTH + me;
  P(GF61) out61 = (P(GF61)) (outRaw + DISTGF61) + g * WIDTH + me;

  for (u32 i = 0; i < NW; ++i) u61[i] = in61[i * G_W];
  fft_RADIX(u61);
  for (u32 i = 0; i < NW; ++i) out61[i * G_W] = u61[i];
}

#else

#if !PFA_RADIX

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig) {
  local GF61 lds61[LDS_BYTES / sizeof(GF61)];
  local GF31 *lds31 = (local GF31 *) lds61;
  GF31 u31[NW];
  GF61 u61[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 m31_log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } m31_combo, m61_combo;
#define frac_bits           m31_combo.a[0]
#define m31_weight_shift    m31_combo.a[1]
#define m31_combo_counter   m31_combo.b
#define m61_weight_shift    m61_combo.a[1]
#define m61_combo_counter   m61_combo.b

  const u64 m31_combo_step = make_u64(m31_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m31_combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * m31_bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  m31_combo_counter = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m31_weight_shift = m31_weight_shift % 31;
  const u64 m61_combo_step = make_u64(m61_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m61_combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * m61_bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  m61_combo_counter = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m61_weight_shift = m61_weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Convert and weight input
    u31[i] = U2(shl(make_Z31(in[p].x), m31_weight_shift0), shl(make_Z31(in[p].y), m31_weight_shift1));      // Form a GF31 from each pair of input words
    u61[i] = U2(shl(make_Z61(in[p].x), m61_weight_shift0), shl(make_Z61(in[p].y), m61_weight_shift1));      // Form a GF61 from each pair of input words

// Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  fft_WIDTH(lds31, u31, smallTrig31, 1, me);
  writeCarryFusedLine(u31, out31, g, me);

  fft_WIDTH(lds61, u61, smallTrig61, 1, me);
  writeCarryFusedLine(u61, out61, g, me);
}

#else

// Native Good-Thomas pack: gather canonical transposed register digits,
// apply the unchanged Aevum IBDWT weights, then run the stock width NTT.
inline u32 pfaLogicalIndex(u32 row, u32 binary_index) {
  const u32 delta = (row + PFA_RADIX - binary_index % PFA_RADIX) % PFA_RADIX;
  const u32 t = (delta * PFA_L_INV) % PFA_RADIX;
  return binary_index + PFA_BINARY_LENGTH * t;
}

inline Word pfaLoadCanonicalWord(CP(Word2) in, u32 logical) {
  const u32 pair = logical >> 1;
  const u32 x = pair / BIG_HEIGHT;
  const u32 g = pair - x * BIG_HEIGHT;
  const Word2 value = in[g * WIDTH + x];
  return (logical & 1u) ? value.y : value.x;
}

// Compute both CRT-plane shifts from one shared fixed-point fracBits product.
// Reduce logical before multiplying: (logical*step) mod p equals
// ((logical mod p)*step) mod p and avoids expensive 64-bit division.
inline uint2 pfaWeightShifts3161(u32 logical, u32 step31, u32 step61) {
  const u64 frac = comboFracBits(logical);
  union { uint2 a; u64 b; } c31, c61;
  c31.b = frac + make_u64(((logical % 31u) * step31) % 31u, 0xFFFFFFFFu);
  c61.b = frac + make_u64(((logical % 61u) * step61) % 61u, 0xFFFFFFFFu);
  return U2(c31.a[1] % 31u, c61.a[1] % 61u);
}

KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig) {
  local GF61 lds61[LDS_BYTES / sizeof(GF61)];
  local GF31 *lds31 = (local GF31 *) lds61;
  GF31 u31[NW];
  GF61 u61[NW];

  const u32 g = get_group_id(0);       // odd-row * SMALL_HEIGHT + y
  const u32 me = get_local_id(0);
  const u32 row = g / SMALL_HEIGHT;
  const u32 y = g - row * SMALL_HEIGHT;

  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  const u32 m31_log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_step = (m31_bigword_weight_shift + 30u) % 31u;
  const u32 m61_log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_step = (m61_bigword_weight_shift + 60u) % 61u;

  const u32 first_binary_pair = me * SMALL_HEIGHT + y;
  u32 n0 = pfaLogicalIndex(row, first_binary_pair * 2u);
  u32 n1 = pfaLogicalIndex(row, first_binary_pair * 2u + 1u);
#pragma unroll
  for (u32 i = 0; i < NW; ++i) {
    const Word word0 = pfaLoadCanonicalWord(in, n0);
    const Word word1 = pfaLoadCanonicalWord(in, n1);
    const uint2 shifts0 = pfaWeightShifts3161(n0, m31_step, m61_step);
    const uint2 shifts1 = pfaWeightShifts3161(n1, m31_step, m61_step);
    u31[i] = U2(shl(make_Z31(word0), shifts0.x), shl(make_Z31(word1), shifts1.x));
    u61[i] = U2(shl(make_Z61(word0), shifts0.y), shl(make_Z61(word1), shifts1.y));
    n0 += PFA_LOGICAL_STEP; if (n0 >= NWORDS) n0 -= NWORDS;
    n1 += PFA_LOGICAL_STEP; if (n1 >= NWORDS) n1 -= NWORDS;
  }

  fft_WIDTH(lds31, u31, smallTrig31, 1, me);
  writeCarryFusedLine(u31, out31, g, me);
  fft_WIDTH(lds61, u61, smallTrig61, 1, me);
  writeCarryFusedLine(u61, out61, g, me);
}

#endif  // PFA_RADIX


#endif  // AEVUM_APPLE_SPLIT_FFTP

/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/*******************************************************************************/


#elif FFT_TYPE == FFT323161

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local GF61 lds61[LDS_BYTES / sizeof(GF61)];
  local F2 *ldsF2 = (local F2 *) lds61;
  local GF31 *lds31 = (local GF31 *) lds61;
  F2 uF2[NW];
  GF31 u31[NW];
  GF61 u61[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  u32 me_frac_bits = fracBits(me * BIG_HEIGHT * 2);
  u32 line_frac_bits = fracBits(g * 2);
  u32 base_frac_bits = me_frac_bits + line_frac_bits;
  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y), base_frac_bits > line_frac_bits);

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 m31_log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } m31_combo, m61_combo;
#define frac_bits           m31_combo.a[0]
#define m31_weight_shift    m31_combo.a[1]
#define m31_combo_counter   m31_combo.b
#define m61_weight_shift    m61_combo.a[1]
#define m61_combo_counter   m61_combo.b

  const u64 m31_combo_step = make_u64(m31_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m31_combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * m31_bigword_weight_shift_minus1, 0)) % (31ULL << 32);
  m31_combo_counter = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m31_weight_shift = m31_weight_shift % 31;
  const u64 m61_combo_step = make_u64(m61_bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 m61_combo_bigstep = (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) + make_u64((G_W * BIG_HEIGHT * 2 - 1) * m61_bigword_weight_shift_minus1, 0)) % (61ULL << 32);
  m61_combo_counter = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);
  m61_weight_shift = m61_weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP32 weights and the second GF31 and GF61 weight shift
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)), frac_bits > base_frac_bits);
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Convert and weight input
    uF2[i] = U2(in[p].x * w1, in[p].y * w2);
    u31[i] = U2(shl(make_Z31(in[p].x), m31_weight_shift0), shl(make_Z31(in[p].y), m31_weight_shift1));      // Form a GF31 from each pair of input words
    u61[i] = U2(shl(make_Z61(in[p].x), m61_weight_shift0), shl(make_Z61(in[p].y), m61_weight_shift1));      // Form a GF61 from each pair of input words

// Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  fft_WIDTH(ldsF2, uF2, smallTrigF2, 1, me);
  writeCarryFusedLine(uF2, outF2, g, me);

  fft_WIDTH(lds31, u31, smallTrig31, 1, me);
  writeCarryFusedLine(u31, out31, g, me);

  fft_WIDTH(lds61, u61, smallTrig61, 1, me);
  writeCarryFusedLine(u61, out61, g, me);
}


#else
error - missing FFTp kernel implementation
#endif
