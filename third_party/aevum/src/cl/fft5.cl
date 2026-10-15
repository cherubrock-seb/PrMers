// Copyright (C) Mihai Preda and George Woltman

#pragma once

#if FFT_FP64

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.4 "5-Point DFT".
// 12 FMA + 24 ADD (or 10 FMA + 28 ADD)
void fft5by(T2 *u, u32 base, u32 step, u32 m) {
  const double
      S1 = -0.048943483704846427, // sin(tau/5) - 1
      S2 = 1.5388417685876268,    // sin(t/5) + sin(2t/5); S2 - 1 == 0.53884176858762667
      S3 = 0.36327126400268045,   // sin(t/5) - sin(2t/5)
      // C1 = 0.11803398874989485,// cos(t/5) - cos(2t/5) - 1
      C2 = 0.55901699437494745;   // (cos(t/5) - cos(2t/5)) / 2

#define A(k) u[(base + k * step) % m]

  X2(A(2), A(3));
  X2(A(1), A(4));
  X2(A(1), A(2));

  A(3) = mul_t4(A(3));
  A(4) = mul_t4(A(4));
  T2 t = A(4) - A(3);
  t = fmaT2(S1, t, t);
  A(3) = fmaT2( S2, A(3), t);
  A(4) = fmaT2(-S3, A(4), t);
  SWAP(A(3), A(4));

  t = A(0);
  A(0) = A(0) + A(1);
  A(1) = fmaT2(-0.25, A(1), t);

#if 0
  A(2) = C2 * A(2);
  X2(A(1), A(2));
#else
  t = A(1);
  A(1) = fmaT2( C2, A(2), t);
  A(2) = fmaT2(-C2, A(2), t);
#endif

  X2(A(1), A(4));
  X2(A(2), A(3));
#undef A
}

void fft5(T2 *u) { return fft5by(u, 0, 1, 5); }

#endif

// FP32 odd-radix implementation from PRPLL 1a563b6, used by explicit PFA5/15.
#define A(i) u[(base + i * step) % M]

#if FFT_FP32

// 18 FMA + 12 ADD.  The method of fft11.cl (prime95's gwnum/zr11.mac): pair inputs k and 5-k, then each output pair j, 5-j is
// (u0 + sum_k cos(j*k*tau/5) sum_k) +- sum_k sin(j*k*tau/5) dif_k, with w = e^(2*pi*i/5).  cbase = u0 + cos(1*tau/5) * sum zeroes one
// term per cos row (the cosine nearest the middle of the range: the smallest coefficients), and the sin rows are scaled by sin(2*tau/5),
// the most accurately represented sine.
void fft5by(F2 *u, u32 base, u32 step, u32 M) {
  const float COS1 = 0.3090169943749474241023f;    // cos(1*tau/5): subtracted from every cos coefficient, it zeroes one term per cos row
  const float C2_C1 = -1.1180339887498948482046f;  // cos(2*tau/5) - cos(1*tau/5)
  const float SIN2 = 0.5877852522924731291687f;    // sin(2*tau/5): factored out of the sin rows, it makes one term per row a plain add
  const float S1_S2 = 1.6180339887498948482046f;   // sin(1*tau/5) / sin(2*tau/5)

  X2_mul_t4(A(1), A(4));
  X2_mul_t4(A(2), A(3));

  F2 sum = A(1) + A(2);
  F2 cbase = fmaT2(COS1, sum, A(0));                    // A(0) + COS1 * sum: each cos row below then has one term fewer
  A(0) = A(0) + sum;

  F2 tmp34a = fmaT2(C2_C1, A(1), cbase);
  F2 tmp25a = fmaT2(C2_C1, A(2), cbase);

  F2 tmp25b = fmaT2(S1_S2, A(4), A(3));
  F2 tmp34b = fmaT2(S1_S2, -A(3), A(4));

  fma_addsub(A(1), A(4), SIN2, tmp25a, tmp25b);
  fma_addsub(A(2), A(3), SIN2, tmp34a, tmp34b);
}

void fft5(F2 *u) { return fft5by(u, 0, 1, 5); }

#endif

#undef A
