// Copyright (C) Mihai Preda and George Woltman

#if FFT_FP64

// The fft3by() and fft5by() below use a different "output map" relative to fft3.cl and fft5.cl
// This way fft15() does not need a "fix order" step at the end.
// See "An In-Place, In-Order Prime Factor Algorithm" by Burrus & Eschenbacher (1981)

// 6 FMA + 6 ADD
void fft3_15(T2 *u, u32 base) {
#define A(k) u[(base + k * 5) % 15]
  double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2

  X2(A(1), A(2));
  A(2) = mul_t4(A(2));

  T2 t = fmaT2(-0.5, A(1), A(0));

  A(0) = A(0) + A(1);

  // here we prefer 2 FMAs vs. 1 MUL + 2 ADDs.
  A(1) = fmaT2(-SIN1, A(2), t);
  A(2) = fmaT2( SIN1, A(2), t);
#undef A
}

// 12 FMA + 24 ADD (or 10 FMA + 28 ADD)
void fft5_15(T2 *u, u32 base) {
  const double
      S1 = -0.048943483704846427, // sin(tau/5) - 1
      S2 = 1.5388417685876268,    // sin(t/5) + sin(2t/5); S2 - 1 == 0.53884176858762667
      S3 = 0.36327126400268045,   // sin(t/5) - sin(2t/5)
      // C1 = 0.11803398874989485,// cos(t/5) - cos(2t/5) - 1
      C2 = 0.55901699437494745;   // (cos(t/5) - cos(2t/5)) / 2

#define A(k) u[(base + k * 3) % 15]

  X2(A(2), A(3));
  X2(A(1), A(4));
  X2(A(1), A(2));

  A(3) = mul_t4(A(3));
  A(4) = mul_t4(A(4));
  T2 t = A(4) - A(3);
  t = fmaT2(S1, t, t);
  A(3) = fmaT2( S2, A(3), t);
  A(4) = fmaT2(-S3, A(4), t);

  t = A(0);
  A(0) = A(0) + A(1);
  A(1) = fmaT2(-0.25, A(1), t);

  t = A(1);
  A(1) = fmaT2( C2, A(2), t);
  A(2) = fmaT2(-C2, A(2), t);

  X2(A(1), A(3));
  X2(A(2), A(4));

  t = A(1);
  A(1) = A(4);
  A(4) = A(2);
  A(2) = t;
#undef A
}

// FFT 15 using PFA ("Prime Factor Algorithm"): 66 FMA + 102 ADD
void fft15(T2 *u) {
  fft3_15(u,  0);
  fft3_15(u,  3);
  fft3_15(u,  6);
  fft3_15(u,  9);
  fft3_15(u, 12);
  fft5_15(u,  0);
  fft5_15(u,  5);
  fft5_15(u, 10);
}

#endif

// FP32 odd-radix implementation from PRPLL 1a563b6, used by explicit PFA5/15.
#if FFT_FP32

#include "fft3.cl"
#include "fft5.cl"

// FFT 15 using PFA ("Prime Factor Algorithm") in place and in order, see "An In-Place, In-Order Prime Factor Algorithm" by Burrus &
// Eschenbacher (1981): five radix-3 DFTs on u[b + 5k] then three radix-5 DFTs on u[b + 3k], with no twiddles and no final reordering.
// Their output maps differ from the standard DFT's -- radix-3 output k is X(2k mod 3), radix-5 output k is X(3k mod 5) -- but that is only
// a permutation of the outputs of fft3by / fft5by, free in registers.  30 FMA + 30 ADD + 3 * (18 FMA + 12 ADD) = 84 FMA + 66 ADD.
void fft15(F2 *u) {
  #pragma unroll
  for (u32 b = 0; b < 15; b += 3) {
    fft3by(u, b, 5, 15);
    SWAP(u[(b + 5) % 15], u[(b + 10) % 15]);                 // outputs 1 and 2 exchanged
  }
  #pragma unroll
  for (u32 b = 0; b < 15; b += 5) {
    fft5by(u, b, 3, 15);
    F2 t = u[(b + 3) % 15];                                    // outputs 1, 2, 3, 4 become X(3), X(1), X(4), X(2)
    u[(b + 3) % 15] = u[(b + 9) % 15];
    u[(b + 9) % 15] = u[(b + 12) % 15];
    u[(b + 12) % 15] = u[(b + 6) % 15];
    u[(b + 6) % 15] = t;
  }
}

#endif
