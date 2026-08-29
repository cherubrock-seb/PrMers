// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "trig.cl"
#include "carryutil.cl"
#include "weight.cl"

#if FFT_TYPE == FFT64

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives with real and imaginary values swapped and weighted.

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 line = gy * CARRY_LEN;
  u32 word_index = (gx * G_W * H + me * H + line) * 2;
  u32 frac_bits = fracBits(word_index) + FRAC_BPW_HI;

  T base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    T w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    T w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;   // Same as frac_bits + (2*i) * FRAC_BPW_HI + FRAC_BPW_HI <= FRAC_BPW_HI;
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), U2(w1, w2), carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives with real and imaginary values swapped and weighted.

KERNEL(G_W) carry(P(Word2) out, CP(F2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.
  u32 line = gy * CARRY_LEN;
  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  u32 frac_bits = fracBits(word_index);

  // Base_frac_bits and frac_bits are inexact values.  We only want to trigger an optional double when it is clear to do so.
  // Fudge base_frac_bits to make it harder to trigger a double when the two inexact values are equal.
  base_frac_bits++;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (line + i) + me;
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), frac_bits > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    frac_bits += FRAC_BPW_HI;
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), U2(w1, w2), carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
    // Generate frac_bits for next pair
    frac_bits += FRAC_BPW_HI;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

KERNEL(G_W) carry(P(Word2) out, CP(GF31) in, u32 posROE, P(CarryABM) carryOut, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 30th root GF31.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), weight_shift0, weight_shift1, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  float fltRoundMax = (float) roundMax / (float) M31;      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

KERNEL(G_W) carry(P(Word2) out, CP(GF61) in, u32 posROE, P(CarryABM) carryOut, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), weight_shift0, weight_shift1, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  float fltRoundMax = (float) roundMax / (float) (M61 >> 32);      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  T base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the FP64 and second GF31 weight shift
    T w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    T w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), SWAP_XY(in31[p]), w1, w2, weight_shift0, weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (line + i) + me;

    // Generate the FP32 and second GF31 weight shift
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), frac_bits > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in31[p]), w1, w2, weight_shift0, weight_shift1,
                                carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  const u32 log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo_counter = comboFracBits(word_index) + make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (line + i) + me;

    // Generate the FP32 and second GF61 weight shift
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), frac_bits > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in61[p]), w1, w2, weight_shift0, weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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
  m31_combo_counter = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  const u64 m61_combo_step = make_u64(m61_bigword_weight_shift_minus1, FRAC_BPW_HI);
  m61_combo_counter = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = (m31_weight_shift + log2_NWORDS + 1) % 31;
  m61_weight_shift = (m61_weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in31[p]), SWAP_XY(in61[p]), m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  float fltRoundMax = (float) roundMax / (float) 0x1FFFFFFF;      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
#if PFA_RESIDENT && PFA_RADIX
  P(Word) outResident = (P(Word)) out;
#endif

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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

  const u64 m31_combo_step = ((u64) m31_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  m31_combo_counter = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  m61_combo_counter = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (PFA_RADIX ? 0 : (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4)) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = (m31_weight_shift + log2_NWORDS + 1) % 31;
  m61_weight_shift = (m61_weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the FP32 and second GF31 and GF61 weight shift
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), frac_bits > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), frac_bits + FRAC_BPW_HI > FRAC_BPW_HI);
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result in canonical logical order.  Only the destination
    // layout changes in resident mode; carry propagation itself is untouched.
    Word2 result = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in31[p]), SWAP_XY(in61[p]), w1, w2, m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                                     LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
#if PFA_RESIDENT && PFA_RADIX
    const u32 logical0 = word_index + 2u * (u32)i;
    outResident[pfaResidentScalarIndex(logical0)] = result.x;
    outResident[pfaResidentScalarIndex(logical0 + 1u)] = result.y;
#else
    out[p] = result;
#endif

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
// EXP4: tiled/coalesced PFA-resident carry for pfa9full:4:512:9:512:202.
// One 512-thread workgroup covers all eight legacy 64-thread gx groups for a
// fixed gy.  Arithmetic and carry chains remain canonical.  The result is
// permuted through LDS so global resident writes are ordered by q rather than
// stride-9 across a warp.
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryResidentTiled is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
KERNEL(WIDTH) carryResidentTiled(P(Word2) out, CP(T2) in, u32 posROE,
                                 P(CarryABM) carryOut,
                                 BigTabFP32 THREAD_WEIGHTS,
                                 P(uint) bufROE) {
  const u32 A = get_local_id(0);          // canonical width coordinate 0..511
  const u32 gy = get_group_id(0);         // carry segment 0..575
  const u32 gx = A / G_W;                 // 0..7
  const u32 me = A - gx * G_W;            // 0..63
  const u32 H = BIG_HEIGHT;                // 4608
  const u32 line = gy * CARRY_LEN;         // aligned to 8

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(Word) outResident = (P(Word)) out;

  // Double-buffered LDS: one barrier per i is enough.  While iteration i is
  // read back in q-order, iteration i+1 writes the other tile.
  local Word2 tile0[WIDTH];
  local Word2 tile1[WIDTH];

  CarryABM carry = (LL & (A == 0) & (gy == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  const u32 word_index = (A * H + line) * 2u;

  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2u);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  const u32 m31_log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = M61_LOG2_ROOT_TWO;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  union { uint2 a; u64 b; } c31, c61;
  const u64 c31_step = ((u64) m31_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 c61_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  c31.b = comboFracBits(word_index) + make_u64(word_index * m31_bigword_weight_shift_minus1, 0xFFFFFFFF);
  c61.b = comboFracBits(word_index) + make_u64(word_index * m61_bigword_weight_shift_minus1, 0xFFFFFFFF);

  const u32 log2_NWORDS = 9u + 9u + 1u; // PFA: WIDTH=512, SMALL_HEIGHT=512, no middle factor
  c31.a[1] = (c31.a[1] + log2_NWORDS + 1u) % 31u;
  c61.a[1] = (c61.a[1] + log2_NWORDS + 1u) % 61u;

  // For s=line+i and L/2=512*512:
  //   q   = (9*A + floor(s/512)) mod 512
  //   y   = s mod 512
  //   rowX= 2*s mod 9, rowY=(2*s+1) mod 9.
  // line is a multiple of 8 and 512 is a multiple of 8, therefore section
  // never changes inside this eight-pair carry segment.
  const u32 section = gy >> 6;             // line / 512
  const u32 y0 = (gy & 63u) << 3;          // line % 512
  const u32 q = (9u * A + section) & 511u;
  u32 rowX = (7u * (gy % 9u)) % 9u;        // 16*gy mod 9
  u32 rowY = rowX + 1u;
  if (rowY >= 9u) rowY -= 9u;

#pragma unroll
  for (i32 i = 0; i < CARRY_LEN; ++i) {
    const u32 p = A + WIDTH * (line + (u32)i);

    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), c31.a[0] > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), c31.a[0] + FRAC_BPW_HI > FRAC_BPW_HI);

    const u32 m31_weight_shift0 = c31.a[1];
    c31.b += c31_step;
    c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    const u32 m31_weight_shift1 = c31.a[1];

    const u32 m61_weight_shift0 = c61.a[1];
    c61.b += c61_step;
    c61.a[1] = adjust_m61_weight_shift(c61.a[1]);
    const u32 m61_weight_shift1 = c61.a[1];

    const bool biglit0 = c31.a[0] <= FRAC_BPW_HI;
    const bool biglit1 = c31.a[0] >= -FRAC_BPW_HI;

    const Word2 result = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in31[p]), SWAP_XY(in61[p]),
                                            w1, w2,
                                            m31_weight_shift0, m31_weight_shift1,
                                            m61_weight_shift0, m61_weight_shift1,
                                            LL != 0 || i != 0, carry,
                                            biglit0, biglit1,
                                            &carry, &roundMax, &carryMax);

    if ((i & 1) == 0) tile0[q] = result;
    else              tile1[q] = result;
    barrier(CLK_LOCAL_MEM_FENCE);

    const Word2 ordered = ((i & 1) == 0) ? tile0[A] : tile1[A];
    const u32 y = y0 + (u32)i;
    const u32 pairX = (rowX * SMALL_HEIGHT + y) * WIDTH + A;
    const u32 pairY = (rowY * SMALL_HEIGHT + y) * WIDTH + A;
    outResident[2u * pairX] = ordered.x;
    outResident[2u * pairY + 1u] = ordered.y;

    // Advance exact weight/frac state for the next canonical pair.
    c31.b += c31_step;
    c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    c61.b += c61_step;
    c61.a[1] = adjust_m61_weight_shift(c61.a[1]);

    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }

  carryOut[WIDTH * gy + A] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}
#endif


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
// EXP6: block four stock carry segments together.  Stock carryB relies on the
// CARRY_LEN invariant that an incoming segment carry dies inside that segment;
// therefore feeding carryOut(segment j) directly into segment j+1 is exactly
// equivalent for the three internal boundaries.  Only one external boundary
// remains per 32 canonical pairs.
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryResidentBlock4 is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
#define PFA_CARRY_BLOCK 4u
KERNEL(WIDTH) carryResidentBlock4(P(Word2) out, CP(T2) in, u32 posROE,
                                   P(CarryABM) carryOut,
                                   BigTabFP32 THREAD_WEIGHTS,
                                   P(uint) bufROE) {
  const u32 A = get_local_id(0);          // canonical width coordinate 0..511
  const u32 blockGy = get_group_id(0);    // 0..143
  const u32 gx = A / G_W;
  const u32 me = A - gx * G_W;
  const u32 H = BIG_HEIGHT;               // 4608
  const u32 line = blockGy * (PFA_CARRY_BLOCK * CARRY_LEN); // multiple of 32

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(Word) outResident = (P(Word)) out;

  local Word2 tile0[WIDTH];
  local Word2 tile1[WIDTH];

  // Resident LEAD_WIDTH PFA squares never use LL, but retain stock expression.
  CarryABM carry = (LL & (A == 0) & (blockGy == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  const u32 word_index = (A * H + line) * 2u;

  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2u);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * M31_LOG2_ROOT_TWO % 31u;
  const u32 m31_step = (m31_bigword_weight_shift + 30u) % 31u;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * M61_LOG2_ROOT_TWO % 61u;
  const u32 m61_step = (m61_bigword_weight_shift + 60u) % 61u;

  union { uint2 a; u64 b; } c31, c61;
  const u64 c31_step = ((u64)m31_step << 32) + FRAC_BPW_HI;
  const u64 c61_step = ((u64)m61_step << 32) + FRAC_BPW_HI;
  c31.b = comboFracBits(word_index) + make_u64(word_index * m31_step, 0xFFFFFFFFu);
  c61.b = comboFracBits(word_index) + make_u64(word_index * m61_step, 0xFFFFFFFFu);

  // PFA log2 binary part: WIDTH=512, SMALL_HEIGHT=512, plus real-pair factor.
  const u32 log2_NWORDS = 9u + 9u + 1u;
  c31.a[1] = (c31.a[1] + log2_NWORDS + 1u) % 31u;
  c61.a[1] = (c61.a[1] + log2_NWORDS + 1u) % 61u;

  // 32 divides 512, so section is constant throughout the whole block.
  const u32 section = blockGy >> 4;          // line / 512
  const u32 y0 = (blockGy & 15u) << 5;       // line % 512
  const u32 q = (9u * A + section) & 511u;
  // rowX=(2*line) mod 9=(64*blockGy) mod 9=blockGy mod 9.
  u32 rowX = blockGy % 9u;
  u32 rowY = rowX + 1u;
  if (rowY >= 9u) rowY -= 9u;

#pragma unroll
  for (i32 i = 0; i < (i32)(PFA_CARRY_BLOCK * CARRY_LEN); ++i) {
    const u32 p = A + WIDTH * (line + (u32)i);

    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), c31.a[0] > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), c31.a[0] + FRAC_BPW_HI > FRAC_BPW_HI);

    const u32 m31_weight_shift0 = c31.a[1];
    c31.b += c31_step;
    c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    const u32 m31_weight_shift1 = c31.a[1];

    const u32 m61_weight_shift0 = c61.a[1];
    c61.b += c61_step;
    c61.a[1] = adjust_m61_weight_shift(c61.a[1]);
    const u32 m61_weight_shift1 = c61.a[1];

    const bool biglit0 = c31.a[0] <= FRAC_BPW_HI;
    const bool biglit1 = c31.a[0] >= -FRAC_BPW_HI;

    const Word2 result = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in31[p]), SWAP_XY(in61[p]),
                                            w1, w2,
                                            m31_weight_shift0, m31_weight_shift1,
                                            m61_weight_shift0, m61_weight_shift1,
                                            LL != 0 || i != 0, carry,
                                            biglit0, biglit1,
                                            &carry, &roundMax, &carryMax);

    if ((i & 1) == 0) tile0[q] = result;
    else              tile1[q] = result;
    barrier(CLK_LOCAL_MEM_FENCE);

    const Word2 ordered = ((i & 1) == 0) ? tile0[A] : tile1[A];
    const u32 y = y0 + (u32)i;
    const u32 pairX = (rowX * SMALL_HEIGHT + y) * WIDTH + A;
    const u32 pairY = (rowY * SMALL_HEIGHT + y) * WIDTH + A;
    outResident[2u * pairX] = ordered.x;
    outResident[2u * pairY + 1u] = ordered.y;

    c31.b += c31_step;
    c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    c61.b += c61_step;
    c61.a[1] = adjust_m61_weight_shift(c61.a[1]);

    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }

  carryOut[WIDTH * blockGy + A] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}
#undef PFA_CARRY_BLOCK
#endif


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryResidentQBlock4 is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
#define PFA_CARRY_BLOCK 4u
#define PFA_Q_INV9 57u
KERNEL(G_W) carryResidentQBlock4(P(Word2) out, CP(T2) in, u32 posROE,
                                  P(CarryABM) carryOut,
                                  BigTabFP32 THREAD_WEIGHTS,
                                  P(uint) bufROE) {
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  const u32 qx = g % NW;
  const u32 blockGy = g / NW;
  const u32 q = qx * G_W + me;
  const u32 line = blockGy * (PFA_CARRY_BLOCK * CARRY_LEN);
  const u32 section = blockGy >> 4;       // line / 512, constant for 32 pairs
  const u32 A = (PFA_Q_INV9 * ((q - section) & (WIDTH - 1u))) & (WIDTH - 1u);
  const u32 gx = A / G_W;
  const u32 canonicalMe = A - gx * G_W;
  const u32 H = BIG_HEIGHT;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(Word) outResident = (P(Word)) out;

  CarryABM carry = (LL & (A == 0) & (blockGy == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  const u32 word_index = (A * H + line) * 2u;

  F base = fancyMul(THREAD_WEIGHTS[canonicalMe].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(canonicalMe * H * 2u);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * M31_LOG2_ROOT_TWO % 31u;
  const u32 m31_step = (m31_bigword_weight_shift + 30u) % 31u;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * M61_LOG2_ROOT_TWO % 61u;
  const u32 m61_step = (m61_bigword_weight_shift + 60u) % 61u;

  union { uint2 a; u64 b; } c31, c61;
  const u64 c31_step = ((u64)m31_step << 32) + FRAC_BPW_HI;
  const u64 c61_step = ((u64)m61_step << 32) + FRAC_BPW_HI;
  c31.b = comboFracBits(word_index) + make_u64(word_index * m31_step, 0xFFFFFFFFu);
  c61.b = comboFracBits(word_index) + make_u64(word_index * m61_step, 0xFFFFFFFFu);
  const u32 log2_NWORDS = 9u + 9u + 1u;
  c31.a[1] = (c31.a[1] + log2_NWORDS + 1u) % 31u;
  c61.a[1] = (c61.a[1] + log2_NWORDS + 1u) % 61u;

  const u32 y0 = (blockGy & 15u) << 5;
  u32 rowX = blockGy % 9u;
  u32 rowY = rowX + 1u; if (rowY >= 9u) rowY -= 9u;

#pragma unroll
  for (i32 i = 0; i < (i32)(PFA_CARRY_BLOCK * CARRY_LEN); ++i) {
    const u32 y = y0 + (u32)i;
    const u32 pairX = (rowX * SMALL_HEIGHT + y) * WIDTH + q;
    const u32 pairY = (rowY * SMALL_HEIGHT + y) * WIDTH + q;

    // EXP7 fftW resident representation is already logical-even/logical-odd.
    const F2 vF2 = U2(inF2[pairX].x, inF2[pairY].y);
    const GF31 v31 = U2(in31[pairX].x, in31[pairY].y);
    const GF61 v61 = U2(in61[pairX].x, in61[pairY].y);

    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), c31.a[0] > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), c31.a[0] + FRAC_BPW_HI > FRAC_BPW_HI);

    const u32 m31_weight_shift0 = c31.a[1];
    c31.b += c31_step; c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    const u32 m31_weight_shift1 = c31.a[1];
    const u32 m61_weight_shift0 = c61.a[1];
    c61.b += c61_step; c61.a[1] = adjust_m61_weight_shift(c61.a[1]);
    const u32 m61_weight_shift1 = c61.a[1];

    const bool biglit0 = c31.a[0] <= FRAC_BPW_HI;
    const bool biglit1 = c31.a[0] >= -FRAC_BPW_HI;

    const Word2 result = weightAndCarryPair(vF2, v31, v61, w1, w2,
                                            m31_weight_shift0, m31_weight_shift1,
                                            m61_weight_shift0, m61_weight_shift1,
                                            LL != 0 || i != 0, carry,
                                            biglit0, biglit1,
                                            &carry, &roundMax, &carryMax);

    // q is contiguous across the workgroup, so both scalar streams are
    // regular stride-2 stores rather than a 9-stride global permutation.
    outResident[2u * pairX] = result.x;
    outResident[2u * pairY + 1u] = result.y;

    c31.b += c31_step; c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    c61.b += c61_step; c61.a[1] = adjust_m61_weight_shift(c61.a[1]);
    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }

  // Preserve canonical A/block order for the tiny carry shuttle so the exact
  // cyclic predecessor rule remains unchanged.
  carryOut[WIDTH * blockGy + A] = carry;
#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}
#undef PFA_Q_INV9
#undef PFA_CARRY_BLOCK
#endif


#if PFA_RESIDENT && PFA_RADIX && FFT_TYPE == FFT323161
// EXP8: keep V7's direct Good-Thomas fftW representation, but restore the
// arithmetic schedule to canonical A order.  Each 512-thread group first
// coalesces one native q-row from all three CRT planes into LDS, then canonical
// thread A reads q=(9*A+section) mod 512.  Results are transposed A->q through
// a Word2 LDS tile before coalesced resident stores.  This avoids V7's
// per-warp permutation of gx/me/weights while retaining V7's cheap fftW.
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 9 || CARRY_LEN != 8 || NW != 8
#error carryResidentATiledBlock4 is specialized for WIDTH=512, SMALL_HEIGHT=512, MIDDLE=9, CARRY_LEN=8, NW=8
#endif
#define PFA_CARRY_BLOCK 4u
KERNEL(WIDTH) carryResidentATiledBlock4(P(Word2) out, CP(T2) in, u32 posROE,
                                         P(CarryABM) carryOut,
                                         BigTabFP32 THREAD_WEIGHTS,
                                         P(uint) bufROE) {
  const u32 A = get_local_id(0);          // canonical width coordinate
  const u32 blockGy = get_group_id(0);    // 0..143
  const u32 gx = A / G_W;
  const u32 me = A - gx * G_W;
  const u32 H = BIG_HEIGHT;
  const u32 line = blockGy * (PFA_CARRY_BLOCK * CARRY_LEN);
  const u32 section = blockGy >> 4;       // line / 512, constant in block
  const u32 q = (9u * A + section) & (WIDTH - 1u);

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(Word) outResident = (P(Word)) out;

  // 16 KiB input tile + 8 KiB output tile = 24 KiB LDS/workgroup.
  // The load index is qLoad=A, therefore all global source accesses are
  // contiguous.  Canonical arithmetic later indexes these tiles by q(A).
  local F2 inTileF2[WIDTH];
  local GF31 inTile31[WIDTH];
  local GF61 inTile61[WIDTH];
  local Word2 outTile[WIDTH];

  CarryABM carry = (LL & (A == 0) & (blockGy == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  const u32 word_index = (A * H + line) * 2u;

  // Exactly the V6 canonical-A weight/counter schedule.
  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  u32 me_frac_bits = fracBits(me * H * 2u);
  u32 step_frac_bits = weightStepFracBits(gx);
  u32 base_frac_bits = me_frac_bits + step_frac_bits;
  base = optionalDouble(base, base_frac_bits > step_frac_bits);

  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * M31_LOG2_ROOT_TWO % 31u;
  const u32 m31_step = (m31_bigword_weight_shift + 30u) % 31u;
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * M61_LOG2_ROOT_TWO % 61u;
  const u32 m61_step = (m61_bigword_weight_shift + 60u) % 61u;

  union { uint2 a; u64 b; } c31, c61;
  const u64 c31_step = ((u64)m31_step << 32) + FRAC_BPW_HI;
  const u64 c61_step = ((u64)m61_step << 32) + FRAC_BPW_HI;
  c31.b = comboFracBits(word_index) + make_u64(word_index * m31_step, 0xFFFFFFFFu);
  c61.b = comboFracBits(word_index) + make_u64(word_index * m61_step, 0xFFFFFFFFu);
  const u32 log2_NWORDS = 9u + 9u + 1u;
  c31.a[1] = (c31.a[1] + log2_NWORDS + 1u) % 31u;
  c61.a[1] = (c61.a[1] + log2_NWORDS + 1u) % 61u;

  const u32 y0 = (blockGy & 15u) << 5;
  u32 rowX = blockGy % 9u;
  u32 rowY = rowX + 1u; if (rowY >= 9u) rowY -= 9u;

#pragma unroll
  for (i32 i = 0; i < (i32)(PFA_CARRY_BLOCK * CARRY_LEN); ++i) {
    const u32 y = y0 + (u32)i;

    // Stage resident native rows in q order.  Here local id A is intentionally
    // used as qLoad, yielding unit-stride source traffic for every plane.
    const u32 loadPairX = (rowX * SMALL_HEIGHT + y) * WIDTH + A;
    const u32 loadPairY = (rowY * SMALL_HEIGHT + y) * WIDTH + A;
    inTileF2[A] = U2(inF2[loadPairX].x, inF2[loadPairY].y);
    inTile31[A] = U2(in31[loadPairX].x, in31[loadPairY].y);
    inTile61[A] = U2(in61[loadPairX].x, in61[loadPairY].y);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Canonical thread A now gets its exact pair via the Good-Thomas q map.
    const F2 vF2 = inTileF2[q];
    const GF31 v31 = inTile31[q];
    const GF61 v61 = inTile61[q];

    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + line + i].x), c31.a[0] > base_frac_bits);
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP), c31.a[0] + FRAC_BPW_HI > FRAC_BPW_HI);

    const u32 m31_weight_shift0 = c31.a[1];
    c31.b += c31_step; c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    const u32 m31_weight_shift1 = c31.a[1];
    const u32 m61_weight_shift0 = c61.a[1];
    c61.b += c61_step; c61.a[1] = adjust_m61_weight_shift(c61.a[1]);
    const u32 m61_weight_shift1 = c61.a[1];

    const bool biglit0 = c31.a[0] <= FRAC_BPW_HI;
    const bool biglit1 = c31.a[0] >= -FRAC_BPW_HI;

    const Word2 result = weightAndCarryPair(vF2, v31, v61, w1, w2,
                                            m31_weight_shift0, m31_weight_shift1,
                                            m61_weight_shift0, m61_weight_shift1,
                                            LL != 0 || i != 0, carry,
                                            biglit0, biglit1,
                                            &carry, &roundMax, &carryMax);

    // Canonical A -> resident q transpose in LDS. q is a permutation, so
    // there are no write collisions. The barrier also protects reuse of the
    // input tiles on the next iteration.
    outTile[q] = result;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Local id is now qOut for the global store, restoring coalesced resident
    // streams.  The components belong to adjacent logical words, which occupy
    // x of rowX and y of rowY respectively.
    const Word2 ordered = outTile[A];
    const u32 outPairX = (rowX * SMALL_HEIGHT + y) * WIDTH + A;
    const u32 outPairY = (rowY * SMALL_HEIGHT + y) * WIDTH + A;
    outResident[2u * outPairX] = ordered.x;
    outResident[2u * outPairY + 1u] = ordered.y;

    c31.b += c31_step; c31.a[1] = adjust_m31_weight_shift(c31.a[1]);
    c61.b += c61_step; c61.a[1] = adjust_m61_weight_shift(c61.a[1]);
    rowX += 2u; if (rowX >= 9u) rowX -= 9u;
    rowY += 2u; if (rowY >= 9u) rowY -= 9u;
  }

  carryOut[WIDTH * blockGy + A] = carry;
#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}
#undef PFA_CARRY_BLOCK
#endif


#else
error - missing Carry kernel implementation
#endif
