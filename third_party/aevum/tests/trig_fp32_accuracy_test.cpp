// Accuracy of the FP32 slowTrig (trig.cl reducedCosSin under FFT_FP32), emulated on the host in float with fmaf.
//
// The host passes trigCoefsFP32(FFTShape::size() / 4) to the kernel as float literals (Gpu.cpp: "%#.9gf" of float(v), which
// round-trips the float exactly) and the kernel evaluates, for k <= ND / 8 (ND = 2 * n):
//   x = k * TRIG_SCALE;  z = x * x;
//   r1 = Horner(S[last..1], z);  r2 = Horner(C[last..1], z);
//   cos = fma(r2, z, 1);  sin = fma(x, S[0], r1 * x)
// which must approximate cos/sin(2 pi k / ND).  Checks, for every odd mid the FP32 FFTs can have, up to large sizes:
//   - x = k * TRIG_SCALE is exact in float (a rounded x shifts the angle);
//   - sin and cos RMS error <= 0.35 ulp (with at least 1000 samples) and max error <= 1.5 ulp (upstream gpuowl 2ba5134:
//     0.46-0.60 -> 0.29 ulp sin RMS).
#include "../src/Trig.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Stats {
  double sumSq = 0, maxErr = 0;
  u64 n = 0;
  void add(double e) { sumSq += e * e; maxErr = std::fmax(maxErr, std::fabs(e)); ++n; }
  double rms() const { return std::sqrt(sumSq / n); }
};

// Error of a float result in ulps of the exact value.
double ulpErr(float got, long double want) {
  if (want == 0) { return got == 0 ? 0 : 1e9; }
  int e;
  std::frexp((double) want, &e);              // |want| in [2^(e-1), 2^e)
  long double ulp = std::ldexp(1.0L, e - 24);  // float has 24 significant bits
  return (double) ((got - want) / ulp);
}

int failures = 0;

void check(u32 mid, u32 twos) {
  u32 const n = mid * twos;   // = size() / 4 = ND / 2
  u32 const ND = 2 * n;
  TrigCoefsFP32 const coefs = trigCoefsFP32(n);
  std::vector<float> S, C;
  for (double v : coefs.sinCoefs) { S.push_back(float(v)); }
  for (double v : coefs.cosCoefs) { C.push_back(float(v)); }
  float const scale = float(coefs.scale);
  size_t const last = S.size() - 1;

  Stats sinS, cosS;
  u64 inexact = 0;
  long double const tau = 2 * 3.141592653589793238462643383279502884L;
  for (u32 k = 0; k <= ND / 8; ++k) {
    float const x = float(k) * scale;
    if ((long double) x != (long double) k * (long double) coefs.scale) { ++inexact; }
    float const z = x * x;
    float r1 = S[last], r2 = C[last];
    for (size_t i = last; i-- > 1;) {
      r1 = std::fmaf(r1, z, S[i]);
      r2 = std::fmaf(r2, z, C[i]);
    }
    r1 = r1 * x;
    float const c = std::fmaf(r2, z, 1.0f);
    float const s = std::fmaf(x, S[0], r1);
    long double const a = tau * k / ND;
    cosS.add(ulpErr(c, std::cos(a)));
    if (k) { sinS.add(ulpErr(s, std::sin(a))); }
  }
  // RMS needs enough samples to mean anything (the smallest sizes have a handful of k).
  bool const rmsBad = cosS.n >= 1000 && (sinS.rms() > 0.35 || cosS.rms() > 0.35);
  bool const bad = inexact || rmsBad || sinS.maxErr > 1.5 || cosS.maxErr > 1.5;
  std::printf("%s n=%u*%u: inexact x %llu, sin rms %.3f max %.3f ulp, cos rms %.3f max %.3f ulp\n", bad ? "FAIL" : "ok  ", mid, twos,
              (unsigned long long) inexact, sinS.rms(), sinS.maxErr, cosS.rms(), cosS.maxErr);
  failures += bad;
}

}  // namespace

int main() {
  // FP32 FFTs (types 2, 4, 50) with power-of-two, PFA (middle 3/7/9) and odd-height shapes; n = size() / 4.
  for (u32 mid : {1u, 3u, 5u, 7u, 9u, 15u}) {
    for (u32 twos : {4u, 8u, 1u << 12, 1u << 16, 1u << 20, 1u << 22}) {
      check(mid, twos);
    }
  }
  if (failures) {
    std::printf("FAIL: %d FP32 slowTrig cases out of bounds\n", failures);
    return 1;
  }
  std::printf("FP32 slowTrig accuracy ok\n");
  return 0;
}
