// Host test: util::mersenneReduce returns the canonical x mod (2^E - 1).
#include <cstdint>
#include <iostream>
#include <random>

#include <gmpxx.h>

#include "util/GmpUtils.hpp"

namespace {

int failures = 0;

void check(const mpz_class& x, uint32_t E, const char* what) {
    const mpz_class Mp = (mpz_class(1) << E) - 1;
    const mpz_class got = util::mersenneReduce(x, E);
    if (got != x % Mp) {
        std::cerr << what << ": mersenneReduce(" << x.get_str(16) << ", " << E
                  << ") = " << got.get_str(16) << ", want " << mpz_class(x % Mp).get_str(16) << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    for (uint32_t E : {13u, 31u, 89u, 127u}) {
        const mpz_class Mp = (mpz_class(1) << E) - 1;
        check(0, E, "zero");
        check(1, E, "one");
        check(Mp - 1, E, "Mp - 1");
        check(Mp, E, "Mp");            // used to come back as Mp
        check(Mp + 1, E, "Mp + 1");
        check(mpz_class(1) << E, E, "2^E");
        check(Mp * 2, E, "2 Mp");      // used to come back as 2 Mp
        check(Mp * Mp, E, "Mp^2");
        check((Mp - 1) * (Mp - 1), E, "(Mp-1)^2");
        check((mpz_class(1) << (2 * E)) - 1, E, "2^(2E) - 1");
        check(mpz_class(1) << (3 * E + 5), E, "3E+5 bits");
    }

    std::mt19937_64 rng(99);
    gmp_randclass rnd(gmp_randinit_default);
    rnd.seed(12345);
    for (int t = 0; t < 2000; ++t) {
        const uint32_t E = 13 + static_cast<uint32_t>(rng() % 200);
        check(rnd.get_z_bits(2 * E + static_cast<mp_bitcnt_t>(rng() % 8)), E, "random");
    }

    // isZeroResidue: 0 and 2^E - 1 are zero modulo 2^E - 1, nothing else is.
    for (uint32_t E : {13u, 32u, 64u, 89u}) {
        const size_t n = (E + 31) / 32;
        std::vector<uint32_t> zero(n, 0u), ones(n, 0u), one(n, 0u), twoE(n, 0u);
        for (uint32_t b = 0; b < E; ++b) ones[b / 32] |= uint32_t(1) << (b % 32);
        one[0] = 1;
        if (!util::isZeroResidue(zero, E)) { std::cerr << "zero residue not detected, E=" << E << "\n"; ++failures; }
        if (!util::isZeroResidue(ones, E)) { std::cerr << "2^E-1 not detected, E=" << E << "\n"; ++failures; }
        if (util::isZeroResidue(one, E)) { std::cerr << "1 taken for zero, E=" << E << "\n"; ++failures; }
        std::vector<uint32_t> almost = ones;
        almost[0] &= ~1u;
        if (util::isZeroResidue(almost, E)) { std::cerr << "2^E-2 taken for zero, E=" << E << "\n"; ++failures; }
    }

    if (failures) {
        std::cerr << "mersenneReduce regression: FAIL\n";
        return 1;
    }
    std::cout << "mersenneReduce regression: PASS\n";
    return 0;
}
