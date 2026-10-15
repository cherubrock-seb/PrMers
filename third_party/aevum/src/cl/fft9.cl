// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.7 "9-Point DFT".
// 12 FMA + 8 MUL, 72 ADD
void fft9(T2 *u) {
  double
      C0 = 0x1.8836fa2cf5039p-1, //   0.766044443118978013 (2*c(u) - c(2*u) - c(4*u))/3
      C1 = 0x1.e11f642522d1cp-1, //   0.939692620785908428 (c(u) + c(2*u) - 2*c(4*u))/3
      C2 = 0x1.63a1a7e0b738ap-3, //   0.173648177666930359 -(c(u) - 2*c(2*u) + c(4*u))/3
      C3 = 0x1.bb67ae8584caap-1, //   0.866025403784438597 s(3*u)
      C4 = 0x1.491b7523c161dp-1, //   0.642787609686539363 s(u)
      C5 = 0x1.5e3a8748a0bf5p-2, //   0.342020143325668713 s(4*u)
      C6 = 0x1.f838b8c811c17p-1; //   0.984807753012208020 s(2*u)

  X2(u[1], u[8]);
  X2(u[2], u[7]);
  X2(u[3], u[6]);
  X2(u[4], u[5]);

  T2 m4 = (u[4] - u[2]) * C1;
  T2 s0 = fmaT2(C0, u[2] - u[1], m4);

  X2(u[1], u[4]);

  T2 t5 = u[1] + u[2];

  T2 m8  = mul_t4(u[8] + u[7]);
  T2 m10 = mul_t4(u[8] - u[5]);

  X2(u[5], u[7]);

  T2 m9  = mul_t4(u[5]) * C5;
  T2 t10 = u[8] + u[7];

  T2 s2 = fmaT2(C4, m8, m9);
  u[5]  = fmaT2(C6, m10, m9);

  u[2] = fmaT2(-0.5, u[3], u[0]);
  u[0] += u[3];

  u[3] = fmaT2(-0.5, t5, u[0]);
  u[0] += t5;

  u[7] = mul_t4(u[6]) * C3;
  u[8] = u[7] + s2;
  u[6] = mul_t4(t10)  * C3;

  u[1] = u[2] - s0;

  u[4] = fmaT2(C2, u[4], m4);

  X2(u[2], u[4]);

  u[4] += s0;

  X2(u[5], u[7]);
  u[5] -= s2;

  X2(u[4], u[5]);
  X2(u[3], u[6]);
  X2(u[2], u[7]);
  X2(u[1], u[8]);
}

#endif

#ifndef AEVUM_FP32_PFA_COSBASE
#define AEVUM_FP32_PFA_COSBASE 0
#endif
#if AEVUM_FP32_PFA_COSBASE != 0 && AEVUM_FP32_PFA_COSBASE != 1
#error AEVUM_FP32_PFA_COSBASE must be 0 or 1
#endif

#if FFT_FP32

// FP32 form of the same optimized 9-point DFT.  This is used by the
// experimental FFT323161 + PFA9 path so all three residue planes share the
// exact Good-Thomas digit map.
void fft9(F2 *u) {
#if AEVUM_FP32_PFA_COSBASE
  // PRPLL 1a563b6: shared cosine base removes one term per output row.
const float COS2 = 0.1736481776669303488517f;    // cos(2*tau/9): subtracted from the cos coefficients of rows 1, 2, 4, it zeroes one term per row
  const float C1_C2 = 0.5923962654520476863507f;   // cos(tau/9) - cos(2*tau/9)
  const float C4_C2 = -1.1133407984528387329058f;  // cos(4*tau/9) - cos(2*tau/9)
  const float SIN3 = 0.8660254037844386467637f;    // sin(3*tau/9) = sqrt(3)/2: the scale of row 3, whose sin terms are all +-1
  const float SIN4 = 0.3420201433256687330441f;    // sin(4*tau/9): factored out of the sin rows 1, 2, 4, it makes one term per row a plain add
  const float S1_S4 = 1.8793852415718167681082f;   // sin(tau/9) / sin(4*tau/9)
  const float S2_S4 = 2.8793852415718167681082f;   // sin(2*tau/9) / sin(4*tau/9)
  const float S3_S4 = 2.5320888862379560704048f;   // sin(3*tau/9) / sin(4*tau/9)

  X2_mul_t4(u[1], u[8]);
  X2_mul_t4(u[2], u[7]);
  X2_mul_t4(u[3], u[6]);
  X2_mul_t4(u[4], u[5]);

  F2 sum = u[1] + u[2] + u[4];
  F2 t = u[0] + u[3];
  F2 cbase = fmaT2(-0.5f, u[3], u[0]);                   // u[0] - u[3]/2: rows 1, 2, 4 all have cos(3*tau/9) = -1/2 on u[3]
  cbase = fmaT2(COS2, sum, cbase);                      // + COS2 * sum: each of rows 1, 2, 4 then has one term fewer
  F2 tmp47a = fmaT2(-0.5f, sum, t);			// row 3: u[0] + u[3] - (u[1] + u[2] + u[4]) / 2
  u[0] = t + sum;

  F2 tmp29a = fmaT2(C1_C2, u[1], cbase);
  F2 tmp38a = fmaT2(C1_C2, u[4], cbase);
  F2 tmp56a = fmaT2(C1_C2, u[2], cbase);

  tmp29a = fmaT2(C4_C2, u[4], tmp29a);
  tmp38a = fmaT2(C4_C2, u[2], tmp38a);
  tmp56a = fmaT2(C4_C2, u[1], tmp56a);

  F2 tmp29b = fmaT2(S1_S4, u[8], u[5]);
  F2 tmp38b = fmaT2(S1_S4, -u[5], u[7]);
  F2 tmp56b = fmaT2(S1_S4, -u[7], u[8]);

  tmp29b = fmaT2(S3_S4, u[6], tmp29b);
  tmp38b = fmaT2(S3_S4, -u[6], tmp38b);
  tmp56b = fmaT2(S3_S4, u[6], tmp56b);

  tmp29b = fmaT2(S2_S4, u[7], tmp29b);
  tmp38b = fmaT2(S2_S4, u[8], tmp38b);
  tmp56b = fmaT2(S2_S4, -u[5], tmp56b);

  F2 tmp47b = u[8] - u[7] + u[5];                               // row 3: sin(3*tau/9) * (dif1 - dif2 + dif4)

  fma_addsub(u[1], u[8], SIN4, tmp29a, tmp29b);
  fma_addsub(u[2], u[7], SIN4, tmp38a, tmp38b);
  fma_addsub(u[4], u[5], SIN4, tmp56a, tmp56b);
  fma_addsub(u[3], u[6], SIN3, tmp47a, tmp47b);

#else
const float
      C0 = 0.766044443118978013f,
      C1 = 0.939692620785908428f,
      C2 = 0.173648177666930359f,
      C3 = 0.866025403784438597f,
      C4 = 0.642787609686539363f,
      C5 = 0.342020143325668713f,
      C6 = 0.984807753012208020f;

  X2(u[1], u[8]);
  X2(u[2], u[7]);
  X2(u[3], u[6]);
  X2(u[4], u[5]);

  F2 m4 = (u[4] - u[2]) * C1;
  F2 s0 = fmaT2(C0, u[2] - u[1], m4);

  X2(u[1], u[4]);
  F2 t5 = u[1] + u[2];
  F2 m8  = mul_t4(u[8] + u[7]);
  F2 m10 = mul_t4(u[8] - u[5]);

  X2(u[5], u[7]);
  F2 m9  = mul_t4(u[5]) * C5;
  F2 t10 = u[8] + u[7];
  F2 s2 = fmaT2(C4, m8, m9);
  u[5]  = fmaT2(C6, m10, m9);

  u[2] = fmaT2(-0.5f, u[3], u[0]);
  u[0] += u[3];
  u[3] = fmaT2(-0.5f, t5, u[0]);
  u[0] += t5;

  u[7] = mul_t4(u[6]) * C3;
  u[8] = u[7] + s2;
  u[6] = mul_t4(t10) * C3;
  u[1] = u[2] - s0;
  u[4] = fmaT2(C2, u[4], m4);

  X2(u[2], u[4]);
  u[4] += s0;
  X2(u[5], u[7]);
  u[5] -= s2;
  X2(u[4], u[5]);
  X2(u[3], u[6]);
  X2(u[2], u[7]);
  X2(u[1], u[8]);

#endif
}

#endif

