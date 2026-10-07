#pragma once
// Affine arithmetic on y^2 = x^3 + 4x modulo N, used to build the torsion-16
// twisted Edwards family: get() computes [n]P0 for P0 = (s1, t1).
//
// N need not be prime.  When a denominator is not invertible modulo N, the
// gcd with N may be a proper factor of N; get() reports it through `factor`
// (when non-null) instead of silently returning the point at infinity.

#include <cstdint>
#include <gmpxx.h>
#ifdef _MSC_VER
#  include <intrin.h>
#endif

namespace ecm_local {

struct EC_mod4 {
    struct Pt { mpz_class x, y; bool inf=false; };

    static inline void norm(mpz_class& z, const mpz_class& N) {
        mpz_mod(z.get_mpz_t(), z.get_mpz_t(), N.get_mpz_t());
        if (z < 0) z += N;
    }

    // Invert den mod N.  On failure the result is "infinity"; if the gcd is a
    // proper factor of N it is stored in *factor.
    static bool invert(const mpz_class& den, const mpz_class& N, mpz_class& inv, mpz_class* factor) {
        if (mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t())) return true;
        if (factor) {
            mpz_class g;
            mpz_gcd(g.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) *factor = g;
        }
        return false;
    }

    static Pt dbl(const Pt& P, const mpz_class& N, mpz_class* factor = nullptr) {
        if (P.inf) return P;
        mpz_class num = 3 * P.x * P.x + 4;
        mpz_class den = 2 * P.y, inv;
        norm(num, N); norm(den, N);
        if (mpz_sgn(den.get_mpz_t()) == 0) return Pt{{}, {}, true};
        if (!invert(den, N, inv, factor)) return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        mpz_class x3 = (lambda * lambda - 2 * P.x) % N; if (x3 < 0) x3 += N;
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }

    static Pt add(const Pt& P, const Pt& Q, const mpz_class& N, mpz_class* factor = nullptr) {
        if (P.inf) {return Q;}
        if (Q.inf) {return P;}
        if (P.x == Q.x) {
            mpz_class ysum = (P.y + Q.y) % N; if (ysum < 0) ysum += N;
            if (ysum == 0) return Pt{{}, {}, true};
            return dbl(P, N, factor);
        }
        mpz_class num = Q.y - P.y; norm(num, N);
        mpz_class den = Q.x - P.x; norm(den, N);
        mpz_class inv;
        if (!invert(den, N, inv, factor)) return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        mpz_class x3 = (lambda * lambda - P.x - Q.x) % N; if (x3 < 0) x3 += N;
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }

    static inline int msb_index_u64(uint64_t n) {
        if (!n) return -1;
    #if defined(_MSC_VER) && !defined(__clang__)
        unsigned long idx;
    #if defined(_M_X64) || defined(_M_ARM64)
        _BitScanReverse64(&idx, n);
        return (int)idx;
    #else
        // 32-bit MSVC fallback
        unsigned long hi = (unsigned long)(n >> 32);
        if (hi) { _BitScanReverse(&idx, hi); return (int)idx + 32; }
        _BitScanReverse(&idx, (unsigned long)(n & 0xFFFFFFFFu));
        return (int)idx;
    #endif
    #else
        // GCC/Clang
        return 63 - __builtin_clzll(n);
    #endif
    }

    // s,t = [n](s1,t1) mod N, or (0,0) at infinity.  If a proper factor of N
    // turns up while inverting, it is stored in *factor (when non-null) and
    // the computation stops.
    static void get(uint64_t n, int s1, int t1, const mpz_class& N, mpz_class& s, mpz_class& t,
                    mpz_class* factor = nullptr) {
        Pt P0, P;
        P0.x = s1; if (s1 < 0) P0.x += N; P0.x %= N;
        P0.y = t1; if (t1 < 0) P0.y += N; P0.y %= N;
        P    = P0;

        int msb = msb_index_u64(n);
        for (int b = msb - 1; b >= 0; --b) {
            P = dbl(P, N, factor);
            if (factor && *factor > 1) { s = 0; t = 0; return; }
            if (((n >> b) & 1ULL) != 0) P = add(P, P0, N, factor);
            if (factor && *factor > 1) { s = 0; t = 0; return; }
            if (P.inf) { s = 0; t = 0; return; }
        }
        s = P.x; t = P.y;
    }
};

} // namespace ecm_local
