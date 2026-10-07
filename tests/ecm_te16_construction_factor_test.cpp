// Host-side check that the torsion-16 twisted Edwards construction
// (include/math/EcMod4.hpp) reports a factor of N found while inverting,
// instead of silently returning the point at infinity.  No GPU is involved.

#include <cstdint>
#include <iostream>

#include <gmpxx.h>

#include "math/EcMod4.hpp"

using ecm_local::EC_mod4;

int main() {
    int failures = 0;
    const mpz_class p1 = 10159;
    const mpz_class p2 = 1000000007;
    const mpz_class N = p1 * p2;

    int with_factor = 0;
    int agree = 0;
    for (uint64_t m = 3; m < 400001; m += 2) {
        mpz_class s_old, t_old, s_new, t_new, f;
        EC_mod4::get(m, 4, 8, N, s_old, t_old);          // no factor reporting
        EC_mod4::get(m, 4, 8, N, s_new, t_new, &f);      // with factor reporting
        if (f > 1) {
            ++with_factor;
            if (!(f == p1 || f == p2)) {
                std::cerr << "m=" << m << ": reported " << f << " is not a proper factor of N\n";
                ++failures;
            }
            if (s_new != 0 || t_new != 0) {
                std::cerr << "m=" << m << ": expected (0,0) after a factor was found\n";
                ++failures;
            }
            if (s_old != 0 || t_old != 0) {
                std::cerr << "m=" << m << ": legacy call did not return infinity\n";
                ++failures;
            }
        } else {
            // No factor appeared: both calls must give the same point.
            if (s_old == s_new && t_old == t_new) ++agree;
            else { std::cerr << "m=" << m << ": results differ without a factor\n"; ++failures; }
        }
    }
    std::cout << "m values with a factor reported: " << with_factor << "\n";
    if (with_factor == 0) {
        std::cerr << "no factor was ever reported; the search range is too small\n";
        ++failures;
    }
    if (agree == 0) { std::cerr << "no ordinary case exercised\n"; ++failures; }
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "PrMers ECM torsion-16 construction factor test passed\n";
    return 0;
}
