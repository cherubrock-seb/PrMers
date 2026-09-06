// Copyright (C) Mihai Preda

#pragma once

#include "fft4.cl"

#if FFT_FP64

//#define X2_apply_delay(a, b) { T2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { T2 t = a; a.x = fma(b.x, M_SQRT1_2, a.x); a.y = fma(b.y, M_SQRT1_2, a.y); b.x = fma(-M_SQRT1_2, b.x, t.x); b.y = fma(-M_SQRT1_2, b.y, t.y); }

void OVERLOAD fft4CoreSpecial(T2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);                                        // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void OVERLOAD fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

//#define X2_apply_delay(a, b) { F2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { F2 t = a; a.x = fma(b.x, (float) M_SQRT1_2, a.x); a.y = fma(b.y, (float) M_SQRT1_2, a.y); b.x = fma((float) -M_SQRT1_2, b.x, t.x); b.y = fma((float) -M_SQRT1_2, b.y, t.y); }

void OVERLOAD fft4CoreSpecial(F2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);                                        // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void OVERLOAD fft8Core(F2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  // Keep the radix-8 type-4 plan self-contained.  Some OpenCL front-ends
  // lose the overload imported from fft4.cl specifically with variant 202.
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(F2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft8Core(GF31 *u) {
  X2(u[0], u[4]);
  X2_mul_t8(u[1], u[5]);
  X2_mul_t4(u[2], u[6]);
  X2_mul_3t8(u[3], u[7]);
  // Inline the exact GF31 fft4Core body twice.  This is algebraically
  // identical and avoids an OpenCL overload-visibility failure in the
  // power-of-two type-4 variant-202 compilation unit.
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  X2(u[4], u[6]);
  X2_mul_t4(u[5], u[7]);
  X2(u[4], u[5]);
  X2(u[6], u[7]);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(GF31 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft4CoreSpecial1(GF61 *u) {         // Starts with u[0,1,2,3] in range of 0..2*M61+epsilon.
  X2q(&u[0], &u[2]);                              // X2(u[0], u[2]);  No reductions mod M61.  u[0,2] range is 0..4+, -2-..2+
  X2q_mul_t4(&u[1], &u[3]);                       // X2(u[1], u[3]);  u[3] = mul_t4(u[3]);    u[1,3] range is 0..4+, -2-..2+
  u[1] = optsubqu(u[1], 2, 2);                    // Partially reduce.  If u[1] > 2*M61, sub 2*M61.  u[1] now has range 0..2+
  u[3] = optsubqs(u[3], 0, 2);                    // Partially reduce.  If u[3] > 0*M61, sub 2*M61.  u[3] now has range -2-..0+
  X2q(&u[0], &u[1]);                              // X2(u[0], u[1]);  u[0,1] range is 0..6+, -2-..4+
  X2q(&u[2], &u[3]);                              // X2(u[2], u[3]);  u[2,3] range is -4-..2+, -2-..4+
  u[0] = modM61q(u[0], 0);
  u[1] = modM61q(u[1], 3);
  u[2] = modM61q(u[2], 5);
  u[3] = modM61q(u[3], 3);
}

void OVERLOAD fft4CoreSpecial2(GF61 *u) {         // Bottom half of an fft8.  Starts with u[0,1,2,3] in range of -1*M61-epsilon..1*M61+epsilon
  X2q(&u[0], &u[2]);                              // X2(u[0], u[2]);  No reductions mod M61.  u[0,2] range is -2-..2+, -2-..2+
  u[1] = mul_t8q(u[1], 3);                        // Perform delayed mul_t8.  u[1] range is 0..1+
  u[3] = mul_t8q(u[3], 3);                        // Perform delayed mul_t8.  u[3] range is 0..1+
  X2q_mul_t4(&u[1], &u[3]);                       // X2(u[1], u[3]);  u[3] = mul_t4(u[3]);    u[1,3] range is 0..2+, -1-..1+
  X2q(&u[0], &u[1]);                              // X2(u[0], u[1]);  u[0,1] range is -2-..4+, -4-..2+
  X2q(&u[2], &u[3]);                              // X2(u[2], u[3]);  u[2,3] range is -3-..3+, -3-..3+
  u[0] = modM61q(u[0], 3);
  u[1] = modM61q(u[1], 5);
  u[2] = modM61q(u[2], 4);
  u[3] = modM61q(u[3], 4);
}

void OVERLOAD fft8Core(GF61 *u) {                 // Starts with all u[i] values in range of 0..M61+epsilon (shorthand notation is 0..1+)
  X2q(&u[0], &u[4]);                              // X2(u[0], u[4]);  No reductions mod M61.  u[0,4] range is 0..2+, -1-..1+
  X2q(&u[1], &u[5]);                              // X2(u[1], u[5]);  Delay mul_t8 on u[5].   u[1,5] range is 0..2+, -1-..1+
  X2q_mul_t4(&u[2], &u[6]);                       // X2(u[2], u[6]);  u[6] = mul_t4(u[6]);    u[2,6] range is 0..2+, -1-..1+
  X2q_mul_t4(&u[3], &u[7]);                       // X2(u[3], u[7]);  u[7] = mul_t4(u[7]);    u[3,7] range is 0..2+, -1-..1+   Delay mul_t8 on u[7].
  fft4CoreSpecial1(u);
  fft4CoreSpecial2(u + 4);
}

void OVERLOAD fft8(GF61 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


// AEVUM_GWOLT_1K_RADIX8_TAIL
//***********************************************************************************************************************
//     In a primarily radix 8 FFT, support other some other options such as radix-4 and radix-16
//***********************************************************************************************************************

#if FFT_FP64

// Perform the last three levels of a radix-16 butterfly.  The initial radix-2 has already been performed and shufl'ed.
// This is used by SIZE=1K, RADIX=8 fft.  There are two versions, one for the first eight radix-16 values and one for the second eight radix-16 values.
void OVERLOAD fft8_16a(T2 *u) {
  fft8(u);
}
void OVERLOAD fft8_16b(T2 *u) {
  const double C1 = 0.92387953251128674, // cos(tau/16)
               S1 = 0.38268343236508978, // sin(tau/16)
               S1_over_C1 = 0.4142135623730950488017,
               C1_over_S1 = 2.4142135623730950488017;

  X2t4(u[0], u[4]);
  X2t4(u[1], u[5]);
  X2t4(u[2], u[6]);
  X2t4(u[3], u[7]);

  u[1] = partial_cmul(u[1], S1_over_C1);  // delays a mul by C1
  u[2] = mul_t8_delayed(u[2]);            // delays a mul by M_SQRT1_2
  u[3] = partial_cmul(u[3], C1_over_S1);  // delays a mul by S1
  X2ad(u[0], u[2], M_SQRT1_2);
  X2ad_mul_t4(u[1], u[3], S1_over_C1);    // mul by S1/C1, now both are delaying a mul by C1
  X2ad(u[0], u[1], C1);                   // apply delayed mul by C1
  X2ad(u[2], u[3], C1);                   // apply delayed mul by C1

  u[5] = partial_cmul(u[5], C1_over_S1);  // delays a mul by S1
  u[6] = mul_t8_delayed(u[6]);            // delays a mul by i*M_SQRT1_2 (a negation cheaper than mul_3t8_delayed)
  u[7] = partial_cmul(u[7], S1_over_C1);  // delays a mul by -C1
  X2t4ad(u[4], u[6], M_SQRT1_2);
  X2ad_mul_t4(u[5], u[7], -C1_over_S1);   // mul by -C1/S1, now both are delaying a mul by S1
  X2ad(u[4], u[5], S1);                   // apply delayed mul by S1
  X2ad(u[6], u[7], S1);                   // apply delayed mul by S1

  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

#if FFT_FP32

// Perform the last three levels of a radix-16 butterfly.  The initial radix-2 has already been performed and shufl'ed.
// This is used by SIZE=1K, RADIX=8 fft.  There are two versions, one for the first eight radix-16 values and one for the second eight radix-16 values.
void OVERLOAD fft8_16a(F2 *u) {
  fft8(u);
}
void OVERLOAD fft8_16b(F2 *u) {
  const float C1 = 0.92387953251128674, // cos(tau/16)
              S1 = 0.38268343236508978, // sin(tau/16)
              S1_over_C1 = 0.4142135623730950488017,
              C1_over_S1 = 2.4142135623730950488017;

  X2t4(u[0], u[4]);
  X2t4(u[1], u[5]);
  X2t4(u[2], u[6]);
  X2t4(u[3], u[7]);

  u[1] = partial_cmul(u[1], S1_over_C1);  // delays a mul by C1
  u[2] = mul_t8_delayed(u[2]);            // delays a mul by M_SQRT1_2
  u[3] = partial_cmul(u[3], C1_over_S1);  // delays a mul by S1
  X2ad(u[0], u[2], M_SQRT1_2);
  X2ad_mul_t4(u[1], u[3], S1_over_C1);    // mul by S1/C1, now both are delaying a mul by C1
  X2ad(u[0], u[1], C1);                   // apply delayed mul by C1
  X2ad(u[2], u[3], C1);                   // apply delayed mul by C1

  u[5] = partial_cmul(u[5], C1_over_S1);  // delays a mul by S1
  u[6] = mul_t8_delayed(u[6]);            // delays a mul by i*M_SQRT1_2 (a negation cheaper than mul_3t8_delayed)
  u[7] = partial_cmul(u[7], S1_over_C1);  // delays a mul by -C1
  X2t4ad(u[4], u[6], M_SQRT1_2);
  X2ad_mul_t4(u[5], u[7], -C1_over_S1);   // mul by -C1/S1, now both are delaying a mul by S1
  X2ad(u[4], u[5], S1);                   // apply delayed mul by S1
  X2ad(u[6], u[7], S1);                   // apply delayed mul by S1

  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

#if NTT_GF31

// Perform the last three levels of a radix-16 butterfly.  The initial radix-2 has already been performed and shufl'ed.
// This is used by SIZE=1K, RADIX=8 fft.  There are two versions, one for the first eight radix-16 values and one for the second eight radix-16 values.
void OVERLOAD fft8_16a(GF31 *u) {
  fft8(u);
}
void OVERLOAD fft8_16b(GF31 *u) {
  const Z31 C1 = 1556715293;
  const Z31 S1 = 978592373;
  const Z31 negC1 = M31 - C1;
  const Z31 negS1 = M31 - S1;

  X2t4(u[0], u[4]);
  X2t4(u[1], u[5]);
  X2t4(u[2], u[6]);
  X2t4(u[3], u[7]);

  u[1] = cmul_const(u[1], U2(C1, S1));
  u[2] = mul_t8(u[2]);
  u[3] = cmul_const(u[3], U2(S1, C1));
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);

  u[5] = cmul_const(u[5], U2(S1, C1));
  u[6] = mul_3t8(u[6]);
  u[7] = cmul_const(u[7], U2(negC1, negS1));
  X2(u[4], u[6]);
  X2_mul_t4(u[5], u[7]);
  X2(u[4], u[5]);
  X2(u[6], u[7]);

  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

#if NTT_GF61

// Perform the last three levels of a radix-16 butterfly.  The initial radix-2 has already been performed and shufl'ed.
// This is used by SIZE=1K, RADIX=8 fft.  There are two versions, one for the first eight radix-16 values and one for the second eight radix-16 values.

// Aevum-local GF61 radix-4 core for the gpuowl 1K radix-8 tail.
// Kept local to fft8.cl to avoid OpenCL overload visibility issues.
void aevum_fft4Core16_GF61(GF61 *u) {
  X2q(&u[0], &u[2]);
  X2q_mul_t4(&u[1], &u[3]);
  X2q(&u[0], &u[1]);
  X2q(&u[2], &u[3]);

  u[0] = modM61q(u[0], 0);
  for (u32 i = 1; i <= 3; ++i)
    u[i] = modM61q(u[i], 3);
}

void OVERLOAD fft8_16a(GF61 *u) {
  // shufl_and_fft2 performed "quick" adds, u[0-7] are in range 0..2+
  X2q(&u[0], &u[4]);               // X2(u[0], u[4]);  No reductions mod M61.  u[0,4] range is 0..4+, -2-..2+
  X2q(&u[1], &u[5]);               // X2(u[1], u[5]);  Delay mul_t8 on u[5].   u[1,5] range is 0..4+, -2-..2+
  X2q_mul_t4(&u[2], &u[6]);        // X2(u[2], u[6]);  u[6] = mul_t4(u[6]);    u[2,6] range is 0..4+, -2-..2+
  X2q_mul_t4(&u[3], &u[7]);        // X2(u[3], u[7]);  u[7] = mul_t4(u[7]);    u[3,7] range is 0..4+, -2-..2+   Delay mul_t8 on u[7].

  // Must normalize values.  The delayed mul_t8s can help with that (half of the complex number needs normalizing before mul_t8).
  for (u32 i = 0; i <= 3; ++i) u[i] = modM61q(u[i], 0);
  u[4] = modM61q(u[4], 3);
  u[5].x = optional_add((i64)u[5].x, 2*M61);   // u[5] now 0-..2+, -2..2+
  u[5] = mul_t8q(u[5], 5);                     // Perform delayed mul_t8 (count of 5 based on u[5].y - u[5].x range of -4-..2+
  u[6] = modM61q(u[6], 3);
  u[7].x = optional_add((i64)u[7].x, 2*M61);   // u[7] now 0-..2+, -2..2+
  u[7] = mul_t8q(u[7], 5);                     // Perform delayed mul_t8.

  aevum_fft4Core16_GF61(u);
  aevum_fft4Core16_GF61(u + 4);

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}
void OVERLOAD fft8_16b(GF61 *u) {
  // shufl_and_fft2 performed "quick" subtracts, u[0-7] are in range -1-..1+
  X2qt4(&u[0], &u[4]);            // -2..2+
  X2qt4(&u[1], &u[5]);
  X2qt4(&u[2], &u[6]);
  X2qt4(&u[3], &u[7]);

  // Must normalize values.  Some mul_t8s can help with that (only half of the complex number needs normalizing before mul_t8).
  u[0] = modM61q(u[0], 3);
  u[1] = modM61q(u[1], 3);
  u[2].x = optional_add((i64)u[2].x, 2*M61);   // u[2] now 0-..2+, -2..2+
  u[3] = modM61q(u[3], 3);
  u[4] = modM61q(u[4], 3);
  u[5] = modM61q(u[5], 3);
  u[6].y = optional_add((i64)u[6].y, 2*M61);   // u[6] now -2-..2+, 0-..2+
  u[7] = modM61q(u[7], 3);

  u[1] = mul_t16(u[1]);
  u[2] = mul_t8q(u[2], 5);                     // Perform mul_t8 (count of 5 based on u[2].y - u[2].x range of -4-..2+, -(u[2].x + u[2].y) range of -4-..2+
  u[3] = mul_3t16(u[3]);
  aevum_fft4Core16_GF61(u);

  u[5] = mul_3t16(u[5]);
  u[6] = mul_3t8q(u[6], 3);                    // Perform mul_3t8 (count of 5 based on u[6].y - u[6].x range of -2-..4+, u[6].y + u[6].x range of -2-..4+
  u[7] = mul_9t16(u[7]);
  aevum_fft4Core16_GF61(u + 4);

  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif
