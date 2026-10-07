#pragma once

// Divisibility test of a GMP integer by a 64-bit value.
//
// mpz_divisible_ui_p / mpz_cmp_ui take an `unsigned long`, which is only 32
// bits wide on LLP64 targets (Windows/MSVC).  Passing a 64-bit q straight
// through would silently truncate it once q >= 2^32, so the admissible-factor
// sieve could report (or miss) a wrong divisor.  When `unsigned long` cannot
// hold q the value is imported into an mpz_class instead.
//
// `UL` is the `unsigned long` type to assume; it exists so the host test can
// exercise the 32-bit behaviour on LP64 hosts.

#include <gmpxx.h>

#include <cstdint>
#include <limits>

namespace core::gm_u64 {

// True when q divides n and q != n (q is a proper small factor of n).
template <typename UL = unsigned long>
inline bool is_proper_divisor(const mpz_class& n, std::uint64_t q) {
    if (q <= static_cast<std::uint64_t>(std::numeric_limits<UL>::max())) {
        return mpz_divisible_ui_p(n.get_mpz_t(), static_cast<UL>(q)) &&
               mpz_cmp_ui(n.get_mpz_t(), static_cast<UL>(q)) != 0;
    }
    mpz_class qz;
    mpz_import(qz.get_mpz_t(), 1, -1, sizeof(q), 0, 0, &q);
    return mpz_divisible_p(n.get_mpz_t(), qz.get_mpz_t()) &&
           mpz_cmp(n.get_mpz_t(), qz.get_mpz_t()) != 0;
}

} // namespace core::gm_u64
