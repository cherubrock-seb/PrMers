// Host test for the Gaussian-Mersenne admissible-factor divisibility check.
//
// On LLP64 targets (Windows/MSVC) `unsigned long` is 32 bits, so passing a
// 64-bit q to mpz_divisible_ui_p truncates it once q >= 2^32.  The helper is
// instantiated with a 32-bit UL here to reproduce that on an LP64 host.
#include "core/GmU64Divisor.hpp"

#include <cstdint>
#include <iostream>

namespace gu = core::gm_u64;

#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; return 1; } } while (0)

// The pre-fix test, with `unsigned long` replaced by the given width.
template <typename UL>
static bool truncating_check(const mpz_class& n, std::uint64_t q) {
    return mpz_divisible_ui_p(n.get_mpz_t(), static_cast<UL>(q)) &&
           mpz_cmp_ui(n.get_mpz_t(), static_cast<UL>(q)) != 0;
}

int main() {
    const std::uint64_t q = (1ULL << 32) + 15;           // truncates to 15
    const std::uint64_t big = (1ULL << 40) + 15;         // also truncates to 15

    // n is a multiple of the truncated value 15 but not of q.
    const mpz_class decoy = mpz_class(15) * 7919;
    CHECK(truncating_check<std::uint32_t>(decoy, q));    // the old bug: false positive
    CHECK(truncating_check<std::uint32_t>(decoy, big));
    CHECK(!gu::is_proper_divisor<std::uint32_t>(decoy, q));
    CHECK(!gu::is_proper_divisor<std::uint32_t>(decoy, big));
    CHECK(!gu::is_proper_divisor<std::uint64_t>(decoy, q));
    CHECK(!gu::is_proper_divisor(decoy, q));

    // n is a true multiple of q: must be reported with either width.
    const mpz_class mult = mpz_class(q) * 12345;
    const mpz_class bigmult = (mpz_class(1) << 70) * big;
    CHECK(gu::is_proper_divisor<std::uint32_t>(mult, q));
    CHECK(gu::is_proper_divisor<std::uint64_t>(mult, q));
    CHECK(gu::is_proper_divisor(mult, q));
    CHECK(gu::is_proper_divisor<std::uint32_t>(bigmult, big));
    CHECK(gu::is_proper_divisor(bigmult, big));
    // The truncated value 15 must not be what decides it.
    CHECK(!gu::is_proper_divisor<std::uint32_t>(mpz_class(q) * 12345 + 1, q));

    // q == n is not a proper divisor, at either width.
    CHECK(!gu::is_proper_divisor<std::uint32_t>(mpz_class(q), q));
    CHECK(!gu::is_proper_divisor(mpz_class(q), q));
    CHECK(!gu::is_proper_divisor<std::uint32_t>(mpz_class(7), 7));

    // Small q still takes the native path and agrees with plain arithmetic.
    CHECK(gu::is_proper_divisor<std::uint32_t>(mpz_class(7) * 11, 7));
    CHECK(!gu::is_proper_divisor<std::uint32_t>(mpz_class(7) * 11, 13));
    CHECK(gu::is_proper_divisor<std::uint32_t>(mpz_class(4294967291UL) * 3, 4294967291ULL));

    std::cout << "GM u64 divisor test passed\n";
    return 0;
}
