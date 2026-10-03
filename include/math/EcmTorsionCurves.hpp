#pragma once
// Montgomery-form ECM curves with a guaranteed rational torsion subgroup.
//
// Both families are built as twisted Edwards curves a*x^2 + y^2 = 1 + d*x^2*y^2
// with a known non-torsion point (x1, y1), and then mapped to the Montgomery
// curve B*v^2 = u^3 + A*u^2 + u through the standard birational map
//
//     A = 2(a+d)/(a-d),   B = 4/(a-d),   u = (1+y)/(1-y)
//
// (Bernstein, Birkner, Joye, Lange, Peters, "Twisted Edwards curves",
// AFRICACRYPT 2008, Theorem 3.2).  The x-only ladder only needs
// A24 = (A+2)/4 = a/(a-d) and the affine u of the starting point; B only selects
// the quadratic twist, and that twist is the one that carries both the image of
// (x1, y1) and the image of the Edwards torsion, so the torsion survives.
//
//  * torsion 16 (Z/2 x Z/8), a = 1: the construction used by the
//    twisted-Edwards ECM path (RunEcmTwistedEdwards.cpp).  (s, t) = [k](4, 8) on
//    y^2 = x^3 + 4x - 16, alpha = (t+8)/(s-4), r = (8+2*alpha)/(8-alpha^2),
//    d = (8r^2-8r+1)/(2r-1)^4, y1 = (2r-1)^2/(4r-3).  With x8 = 2r-1 this is
//    d = (2*x8^2-1)/x8^4, i.e. Bernstein, Birkner, Lange, Peters, "ECM using
//    Edwards curves", Math. Comp. 82 (2013), Theorems 6.6 and 6.9: (x8, x8) has
//    order 8 and d is a square, so the torsion group is Z/2 x Z/8.
//
//  * torsion 8 (Z/8), a = -1: Bernstein, Birkner, Lange, "Starfish on strike",
//    LATINCRYPT 2010, Theorems 4.1 and 4.4.  (r, s) = [k](4, -16) on
//    S^2 = R^3 + 48R, u = 2r/s, v = (2r^3 - s^2)/s^2, d = 16u^4/(4u^4-1)^2,
//    non-torsion point (x1, y1) = (2u^2, (4u^4-1)/v).  The point of order 8 is
//    (x8, y8) = ((2u^2-1)/(2u), (2u^2+1)/(2u)).
//
// Every computation is done modulo N.  A denominator that is not invertible
// modulo N either exposes a proper factor (BuildStatus::Factor, gcd in
// `factor`) or means this k is degenerate modulo N (BuildStatus::Degenerate);
// the caller should then try another k.

#include <gmpxx.h>
#include <cstdint>

namespace ecm_torsion {

enum class BuildStatus { Ok, Factor, Degenerate };

struct MontgomeryCurve {
    mpz_class A;        // Montgomery coefficient A
    mpz_class A24;      // (A + 2) / 4, as used by the x-only ladder
    mpz_class x0;       // affine u-coordinate of the starting point
    mpz_class te_a;     // twisted Edwards a (1 or N-1) the curve was built from
    mpz_class te_d;     // twisted Edwards d
    mpz_class torsion_x;// affine u-coordinate of a point of order 8 (for tests)
};

namespace detail {

struct ModCtx {
    const mpz_class& N;
    mpz_class factor = 0;
    BuildStatus status = BuildStatus::Ok;

    explicit ModCtx(const mpz_class& n) : N(n) {}

    mpz_class red(const mpz_class& a) const {
        mpz_class r;
        mpz_mod(r.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
        return r;
    }
    mpz_class add(const mpz_class& a, const mpz_class& b) const { return red(a + b); }
    mpz_class sub(const mpz_class& a, const mpz_class& b) const { return red(a - b); }
    mpz_class mul(const mpz_class& a, const mpz_class& b) const { return red(a * b); }
    mpz_class sqr(const mpz_class& a) const { return red(a * a); }

    // Returns false and records Factor/Degenerate if a is not invertible mod N.
    bool inv(const mpz_class& a, mpz_class& out) {
        mpz_class ar = red(a);
        if (ar != 0 && mpz_invert(out.get_mpz_t(), ar.get_mpz_t(), N.get_mpz_t())) return true;
        mpz_class g;
        mpz_gcd(g.get_mpz_t(), ar.get_mpz_t(), N.get_mpz_t());
        if (g > 1 && g < N) { factor = g; status = BuildStatus::Factor; }
        else { status = BuildStatus::Degenerate; }
        return false;
    }
    bool div(const mpz_class& num, const mpz_class& den, mpz_class& out) {
        mpz_class i;
        if (!inv(den, i)) return false;
        out = mul(num, i);
        return true;
    }
};

// Affine short Weierstrass y^2 = x^3 + a4*x + a6 (a6 is implicit).
struct WPt { mpz_class x, y; bool inf = false; };

inline bool w_add(ModCtx& m, const mpz_class& a4, const WPt& P, const WPt& Q, WPt& R) {
    if (P.inf) { R = Q; return true; }
    if (Q.inf) { R = P; return true; }
    mpz_class lam;
    if (m.red(P.x - Q.x) == 0) {
        if (m.red(P.y + Q.y) == 0) { R = WPt{0, 0, true}; return true; }
        if (!m.div(m.add(m.mul(3, m.sqr(P.x)), a4), m.mul(2, P.y), lam)) return false;
    } else {
        if (!m.div(m.sub(Q.y, P.y), m.sub(Q.x, P.x), lam)) return false;
    }
    mpz_class x3 = m.sub(m.sub(m.sqr(lam), P.x), Q.x);
    mpz_class y3 = m.sub(m.mul(lam, m.sub(P.x, x3)), P.y);
    R = WPt{x3, y3, false};
    return true;
}

// [k]P by left-to-right double-and-add.  A point at infinity modulo N (rather
// than modulo one prime factor) is reported as Degenerate.
inline bool w_mul(ModCtx& m, const mpz_class& a4, uint64_t k, const WPt& P, WPt& R) {
    if (k == 0) { m.status = BuildStatus::Degenerate; return false; }
    int top = 63;
    while (!((k >> top) & 1ULL)) --top;
    WPt acc = P;
    for (int i = top - 1; i >= 0; --i) {
        if (!w_add(m, a4, acc, acc, acc)) return false;
        if ((k >> i) & 1ULL) { if (!w_add(m, a4, acc, P, acc)) return false; }
    }
    if (acc.inf) { m.status = BuildStatus::Degenerate; return false; }
    R = acc;
    return true;
}

// Twisted Edwards (a, d) with point ordinate y1 -> Montgomery A, A24, x0.
inline bool te_to_montgomery(ModCtx& m, const mpz_class& a, const mpz_class& d,
                             const mpz_class& y1, MontgomeryCurve& out) {
    mpz_class A24, x0, dinv;
    if (!m.inv(d, dinv)) return false;                       // d = 0: singular
    if (!m.div(a, m.sub(a, d), A24)) return false;          // (A+2)/4 = a/(a-d)
    if (!m.div(m.add(1, y1), m.sub(1, y1), x0)) return false; // u = (1+y)/(1-y)
    out.A24 = A24;
    out.A = m.sub(m.mul(4, A24), 2);
    out.x0 = x0;
    out.te_a = m.red(a);
    out.te_d = m.red(d);
    return true;
}

} // namespace detail

// Z/2 x Z/8 torsion.  k should be odd (the twisted-Edwards path uses odd k).
inline BuildStatus build_montgomery_torsion16(const mpz_class& N, uint64_t k,
                                              MontgomeryCurve& out, mpz_class& factor) {
    detail::ModCtx m(N);
    auto fail = [&]() { factor = m.factor; return m.status; };
    detail::WPt G{4, 8, false}, P;
    if (!detail::w_mul(m, 4, k, G, P)) return fail();
    const mpz_class& s = P.x;
    const mpz_class& t = P.y;
    mpz_class alpha, r, d, y1;
    if (!m.div(m.add(t, 8), m.sub(s, 4), alpha)) return fail();
    if (!m.div(m.add(8, m.mul(2, alpha)), m.sub(8, m.sqr(alpha)), r)) return fail();
    mpz_class x8 = m.sub(m.mul(2, r), 1);                    // 2r - 1
    mpz_class x8sq = m.sqr(x8);
    if (!m.div(m.sub(m.mul(2, x8sq), 1), m.sqr(x8sq), d)) return fail(); // (2x8^2-1)/x8^4
    if (!m.div(x8sq, m.sub(m.mul(4, r), 3), y1)) return fail();
    if (!detail::te_to_montgomery(m, 1, d, y1, out)) return fail();
    // Order-8 Edwards point (x8, x8) maps to u = (1+x8)/(1-x8).
    if (!m.div(m.add(1, x8), m.sub(1, x8), out.torsion_x)) return fail();
    factor = 0;
    return BuildStatus::Ok;
}

// Z/8 torsion.  k = 1 is rejected (it yields s = -4r, a torsion point).
inline BuildStatus build_montgomery_torsion8(const mpz_class& N, uint64_t k,
                                             MontgomeryCurve& out, mpz_class& factor) {
    detail::ModCtx m(N);
    auto fail = [&]() { factor = m.factor; return m.status; };
    if (k < 2) { factor = 0; return BuildStatus::Degenerate; }
    detail::WPt G{4, N - 16, false}, P;
    if (!detail::w_mul(m, 48, k, G, P)) return fail();
    const mpz_class& r = P.x;
    const mpz_class& s = P.y;
    mpz_class u, v, d, y1;
    if (!m.div(m.mul(2, r), s, u)) return fail();
    mpz_class s2 = m.sqr(s);
    if (!m.div(m.sub(m.mul(2, m.mul(m.sqr(r), r)), s2), s2, v)) return fail();
    mpz_class u2 = m.sqr(u);
    mpz_class u4 = m.sqr(u2);
    mpz_class e = m.sub(m.mul(4, u4), 1);                    // 4u^4 - 1
    if (!m.div(m.mul(16, u4), m.sqr(e), d)) return fail();
    if (!m.div(e, v, y1)) return fail();
    // Reject the excluded points s = +-4r (then x1 = 2u^2 is a torsion abscissa).
    if (m.red(s - 4 * r) == 0 || m.red(s + 4 * r) == 0) { factor = 0; return BuildStatus::Degenerate; }
    if (!detail::te_to_montgomery(m, N - 1, d, y1, out)) return fail();
    // Order-8 point y8 = (2u^2+1)/(2u) maps to u8 = (1+y8)/(1-y8)
    //   = (2u^2+2u+1)/(-(2u^2-2u+1)).
    mpz_class num = m.add(m.add(m.mul(2, u2), m.mul(2, u)), 1);
    mpz_class den = m.sub(0, m.add(m.sub(m.mul(2, u2), m.mul(2, u)), 1));
    if (!m.div(num, den, out.torsion_x)) return fail();
    factor = 0;
    return BuildStatus::Ok;
}

} // namespace ecm_torsion
