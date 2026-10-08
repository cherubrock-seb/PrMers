// Host test for include/math/EcmRandom.hpp (ECM curve randomness).
//
// The reference values were computed independently (Python, 64-bit masked
// splitmix64), so a driver that narrows the words to `unsigned long` -- 32 bits on
// Windows (LLP64) -- cannot pass on any platform.
#include <gmpxx.h>

#include "math/EcmRandom.hpp"

#include <cstdint>
#include <iostream>
#include <string>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++failures; std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; } } while (0)

struct Ref { std::uint64_t seed; unsigned bits; const char* hex; };

static const Ref refs[] = {
    {1ULL, 63, "910a2dec89025cc1"},
    {1ULL, 192, "910a2dec89025cc1beeb8da1658eec67f893a2eefb32555e"},
    {1ULL, 256, "910a2dec89025cc1beeb8da1658eec67f893a2eefb32555e71c18690ee42c90b"},
    {111ULL, 63, "f9364c1f89270349"},
    {111ULL, 256, "f9364c1f89270349830e76017ba2d95d28ef050f7bcd3d42a4ab8925801602d2"},
    {0xDEADBEEFCAFEF00DULL, 63, "901d4f652fb472cb"},
    {0xDEADBEEFCAFEF00DULL, 192, "901d4f652fb472cba7ce246440f7452719b40bbbb9380d34"},
    {0xFFFFFFFFFFFFFFFFULL, 63, "e4d971771b652c20"},
    {0xFFFFFFFFFFFFFFFFULL, 256, "e4d971771b652c20e99ff867dbf682c9382ff84cb27281e96d1db36ccba982d2"},
    {0ULL, 63, "e220a8397b1dcdaf"},
    {0ULL, 192, "e220a8397b1dcdaf6e789e6aa1b965f406c45d188009454f"},
};

// What the drivers did before: every word narrowed to a 32-bit unsigned long, as on LLP64.
static mpz_class legacy_llp64(std::uint64_t seed0, unsigned bits) {
    mpz_class z = 0;
    std::uint64_t s = seed0;
    for (unsigned i = 0; i < bits; i += 64) {
        z <<= 64;
        z += static_cast<unsigned long>(static_cast<std::uint32_t>(ecm_rng::splitmix64_step(s)));
    }
    return z;
}

int main() {
    for (const Ref& r : refs) {
        const mpz_class got = ecm_rng::random_mpz_bits(r.seed, r.bits);
        const mpz_class want(std::string(r.hex), 16);
        if (got != want) {
            std::cerr << "seed " << r.seed << " bits " << r.bits << ": got " << got.get_str(16)
                      << " want " << r.hex << "\n";
        }
        CHECK(got == want);
        // The 32-bit narrowing is a different (and much smaller) number.
        CHECK(legacy_llp64(r.seed, r.bits) != want);
        CHECK(mpz_sizeinbase(legacy_llp64(r.seed, r.bits).get_mpz_t(), 2) <= 32 + 64 * ((r.bits - 1) / 64));
    }

    // Word count: ceil(bits / 64), including the boundaries.
    CHECK(ecm_rng::random_words(1, 0).empty());
    CHECK(ecm_rng::random_words(1, 1).size() == 1);
    CHECK(ecm_rng::random_words(1, 63).size() == 1);
    CHECK(ecm_rng::random_words(1, 64).size() == 1);
    CHECK(ecm_rng::random_words(1, 65).size() == 2);
    CHECK(ecm_rng::random_words(1, 256).size() == 4);
    CHECK(ecm_rng::random_words(1, 257).size() == 5);
    CHECK(ecm_rng::random_mpz_bits(1, 0) == 0);

    // Words with the top 32 bits set survive intact (the part a 32-bit unsigned long loses).
    bool saw_high = false;
    for (std::uint64_t seed = 0; seed < 64; ++seed) {
        for (std::uint64_t w : ecm_rng::random_words(seed, 256)) {
            if ((w >> 32) != 0) saw_high = true;
            CHECK(ecm_rng::random_mpz_bits(seed, 64) == mpz_class(std::to_string(ecm_rng::random_words(seed, 64)[0])));
        }
    }
    CHECK(saw_high);

    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "ecm random words test passed\n";
    return 0;
}
