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

    if (failures) {
        std::cerr << "mersenneReduce regression: FAIL\n";
        return 1;
    }
    std::cout << "mersenneReduce regression: PASS\n";
    return 0;
}
