// Host-side check of the Montgomery torsion-8 / torsion-16 ECM curve families
// (include/math/EcmTorsionCurves.hpp).  No GPU is involved.
//
// For each family the curve is built modulo a prime p (the same code path the
// ECM driver runs modulo N) and checked for:
//   * exact group order on small primes: 8 | #E (resp. 16 | #E), counting the
//     quadratic twist that actually contains the starting point x0;
//   * on 30..40-bit primes, a structural proof: the reported torsion abscissa
//     has exact order 8 on the same twist as x0 (and, for torsion 16, the
//     2-torsion is full), so Z/8 (resp. Z/2 x Z/8) embeds in the group;
//   * x0 is not a torsion point of order dividing 16;
//   * determinism, and a proper factor reported for a composite N on which a
//     construction denominator vanishes modulo one prime only.

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gmpxx.h>

#include "math/EcmTorsionCurves.hpp"

using ecm_torsion::BuildStatus;
using ecm_torsion::MontgomeryCurve;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << "\n"; }
}

bool is_prime(const mpz_class& n) { return mpz_probab_prime_p(n.get_mpz_t(), 30) != 0; }

mpz_class random_prime(std::mt19937_64& rng, unsigned bits) {
    for (;;) {
        mpz_class c = mpz_class((unsigned long)(rng() >> (64 - bits))) | (mpz_class(1) << (bits - 1)) | 1;
        if (is_prime(c)) return c;
    }
}

int legendre(const mpz_class& a, const mpz_class& p) {
    mpz_class r = a % p; if (r < 0) r += p;
    return mpz_legendre(r.get_mpz_t(), p.get_mpz_t());
}

mpz_class md(const mpz_class& a, const mpz_class& p) { mpz_class r = a % p; if (r < 0) r += p; return r; }

mpz_class rhs(const mpz_class& A, const mpz_class& x, const mpz_class& p) {
    return md(x * x * x + A * x * x + x, p);
}

// x-only Montgomery ladder: returns Z of [k](x : 1); Z == 0 means [k]P = O.
mpz_class ladder_z(uint64_t k, const mpz_class& x, const mpz_class& A24, const mpz_class& p) {
    mpz_class X1 = 1, Z1 = 0, X2 = x, Z2 = 1;
    int top = 63;
    while (top > 0 && !((k >> top) & 1ULL)) --top;
    for (int i = top; i >= 0; --i) {
        bool bit = (k >> i) & 1ULL;
        mpz_class& Xa = bit ? X2 : X1; mpz_class& Za = bit ? Z2 : Z1;   // doubled
        mpz_class t1 = md((X1 - Z1) * (X2 + Z2), p);
        mpz_class t2 = md((X1 + Z1) * (X2 - Z2), p);
        mpz_class Xs = md((t1 + t2) * (t1 + t2), p);
        mpz_class Zs = md(x * (t1 - t2) * (t1 - t2), p);
        mpz_class U = md((Xa + Za) * (Xa + Za), p);
        mpz_class V = md((Xa - Za) * (Xa - Za), p);
        mpz_class E = md(U - V, p);
        mpz_class Xd = md(U * V, p);
        mpz_class Zd = md(E * (V + A24 * E), p);
        if (bit) { X1 = Xs; Z1 = Zs; X2 = Xd; Z2 = Zd; }
        else     { X2 = Xs; Z2 = Zs; X1 = Xd; Z1 = Zd; }
    }
    return md(Z1, p);
}

// #E(F_p) for the quadratic twist containing x0, by direct character sum.
uint64_t group_order_containing(const mpz_class& A, const mpz_class& x0, uint64_t p) {
    const unsigned __int128 P = p;
    uint64_t a = mpz_class(A % mpz_class((unsigned long)p)).get_ui();
    auto mulmod = [&](uint64_t x, uint64_t y) { return (uint64_t)((unsigned __int128)x * y % P); };
    auto powmod = [&](uint64_t b, uint64_t e) { uint64_t r = 1; while (e) { if (e & 1) r = mulmod(r, b); b = mulmod(b, b); e >>= 1; } return r; };
    int64_t sum = 0;
    for (uint64_t x = 0; x < p; ++x) {
        uint64_t f = (mulmod(mulmod(x, x), (x + a) % p) + x) % p;
        if (f == 0) continue;
        sum += (powmod(f, (p - 1) / 2) == 1) ? 1 : -1;
    }
    uint64_t n = (uint64_t)((int64_t)p + 1 + sum);
    mpz_class pz((unsigned long)p);
    int l = legendre(rhs(A, x0, pz), pz);
    return (l >= 0) ? n : 2 * p + 2 - n;
}

using Builder = BuildStatus (*)(const mpz_class&, uint64_t, MontgomeryCurve&, mpz_class&);

struct Family { const char* name; Builder build; unsigned torsion; };

uint64_t pick_k(std::mt19937_64& rng, unsigned torsion) {
    uint64_t k = rng();
    if (torsion == 16) k |= 1ULL;          // the driver uses odd k for torsion 16
    if (k < 2) k = 3;
    return k;
}

void check_family(const Family& fam, std::mt19937_64& rng) {
    // Exact counts on small primes.
    unsigned exact_ok = 0, exact_n = 0, skipped = 0;
    while (exact_n < 60) {
        mpz_class p = random_prime(rng, 13 + (unsigned)(rng() % 4));   // 13..16 bits
        MontgomeryCurve c; mpz_class f;
        uint64_t k = pick_k(rng, fam.torsion);
        if (fam.build(p, k, c, f) != BuildStatus::Ok) { ++skipped; continue; }
        if (legendre(rhs(c.A, c.x0, p), p) == 0) { ++skipped; continue; }
        uint64_t n = group_order_containing(c.A, c.x0, p.get_ui());
        ++exact_n;
        if (n % fam.torsion == 0) ++exact_ok;
        else std::cerr << "  " << fam.name << " p=" << p << " k=" << k << " #E=" << n << "\n";
    }
    std::cout << fam.name << ": exact count " << fam.torsion << " | #E in " << exact_ok << "/" << exact_n
              << " small primes (" << skipped << " degenerate draws skipped)\n";
    expect(exact_ok == exact_n, std::string(fam.name) + " exact torsion divisibility");

    // Structural check on 30..40-bit primes.
    unsigned struct_ok = 0, struct_n = 0, x0_torsion = 0;
    skipped = 0;
    while (struct_n < 400) {
        mpz_class p = random_prime(rng, 30 + (unsigned)(rng() % 11));
        MontgomeryCurve c; mpz_class f;
        uint64_t k = pick_k(rng, fam.torsion);
        if (fam.build(p, k, c, f) != BuildStatus::Ok) { ++skipped; continue; }
        ++struct_n;
        bool ok = true;
        // A24 consistent with A.
        ok &= md(4 * c.A24 - 2 - c.A, p) == 0;
        // Nonsingular: A^2 != 4.
        ok &= md(c.A * c.A - 4, p) != 0;
        // Torsion point has exact order 8 ...
        ok &= ladder_z(8, c.torsion_x, c.A24, p) == 0;
        ok &= ladder_z(4, c.torsion_x, c.A24, p) != 0;
        // ... on the same twist as x0.
        ok &= legendre(rhs(c.A, c.torsion_x, p) * rhs(c.A, c.x0, p), p) == 1;
        if (fam.torsion == 16) ok &= legendre(c.A * c.A - 4, p) == 1;  // full 2-torsion
        if (ok) ++struct_ok;
        else std::cerr << "  " << fam.name << " structural failure p=" << p << " k=" << k << "\n";
        if (ladder_z(16, c.x0, c.A24, p) == 0) ++x0_torsion;
    }
    std::cout << fam.name << ": structural Z/" << (fam.torsion == 16 ? "2 x Z/8" : "8") << " embedding on "
              << struct_ok << "/" << struct_n << " primes of 30-40 bits (" << skipped << " degenerate draws skipped); "
              << "x0 killed by 16 on " << x0_torsion << "\n";
    expect(struct_ok == struct_n, std::string(fam.name) + " structural torsion");
    // x0 of order dividing 16 modulo a random large prime happens with
    // probability about 1/p^(1/2) at most; allow a couple of coincidences.
    expect(x0_torsion <= 2, std::string(fam.name) + " x0 should not be a torsion point");

    // Determinism: the same (N, k) gives the same curve.
    {
        mpz_class N = (mpz_class(1) << 127) - 1;
        uint64_t k = 0x9E3779B97F4A7C15ULL;
        MontgomeryCurve c1, c2; mpz_class f1, f2;
        BuildStatus s1 = fam.build(N, k, c1, f1), s2 = fam.build(N, k, c2, f2);
        expect(s1 == BuildStatus::Ok && s2 == BuildStatus::Ok, std::string(fam.name) + " builds mod M127");
        expect(c1.A == c2.A && c1.x0 == c2.x0 && c1.A24 == c2.A24, std::string(fam.name) + " deterministic");
        expect(md(4 * c1.A24 - 2 - c1.A, N) == 0, std::string(fam.name) + " A24 = (A+2)/4 mod M127");
    }

    // Factor path: find (q, k) degenerate mod a small prime q, then N = q * P
    // must report q as the factor.
    {
        bool tested = false;
        const mpz_class P = (mpz_class(1) << 89) - 1;           // prime M89
        for (unsigned long q = 101; q < 400 && !tested; q += 2) {
            mpz_class qz(q);
            if (!is_prime(qz)) continue;
            for (uint64_t k = 3; k < 200 && !tested; k += (fam.torsion == 16 ? 2 : 1)) {
                MontgomeryCurve c; mpz_class f;
                if (fam.build(qz, k, c, f) != BuildStatus::Degenerate) continue;
                MontgomeryCurve cP; mpz_class fP;
                if (fam.build(P, k, cP, fP) != BuildStatus::Ok) continue;
                BuildStatus s = fam.build(qz * P, k, c, f);
                // The degenerate step happens mod q; it may be preceded by a
                // different one, so accept any proper factor.
                expect(s == BuildStatus::Factor && f > 1 && f < qz * P && (qz * P) % f == 0,
                       std::string(fam.name) + " reports a proper factor");
                if (s == BuildStatus::Factor) {
                    std::cout << fam.name << ": composite N = " << q << " * M89, k=" << k
                              << " -> factor " << f << "\n";
                }
                tested = true;
            }
        }
        expect(tested, std::string(fam.name) + " found a factor-path test case");
    }
}

} // namespace

int main() {
    std::mt19937_64 rng(0x5EC0DE5EEDULL);
    const Family fams[] = {
        {"torsion16", &ecm_torsion::build_montgomery_torsion16, 16},
        {"torsion8",  &ecm_torsion::build_montgomery_torsion8,  8},
    };
    for (const auto& f : fams) check_family(f, rng);
    if (failures) { std::cerr << failures << " check(s) failed\n"; return 1; }
    std::cout << "ecm torsion curves: all checks passed\n";
    return 0;
}
