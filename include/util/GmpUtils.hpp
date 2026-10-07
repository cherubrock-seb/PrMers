#pragma once
#include <cstdint>
#include <vector>
#include <gmpxx.h>
#include <gmp.h>

// GMP-based modular arithmetic helpers
namespace util {
    mpz_class convertToGMP(const std::vector<uint32_t>& words);
    std::vector<uint32_t> convertFromGMP(const mpz_class& gmp_val);
    mpz_class vectToMpz(const std::vector<uint64_t>& v,
                        const std::vector<int>& widths,
                        const mpz_class& Mp);

    // True when the E-bit residue in `words` is 0 modulo 2^E - 1: all words
    // zero, or the E low bits all ones (2^E - 1 itself).
    bool isZeroResidue(const std::vector<uint32_t>& words, uint32_t E);

    mpz_class mersenneReduce(const mpz_class& x, uint32_t E);
    mpz_class mersennePowMod(const mpz_class& base, uint64_t exp, uint32_t E);
}
