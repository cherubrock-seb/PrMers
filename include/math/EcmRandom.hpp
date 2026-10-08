#pragma once

// Deterministic random words for the ECM drivers (curve sigma, fallback points).
//
// A curve seed must give the same curve on every platform.  The words are 64-bit
// values; they must never be narrowed to `unsigned long`, which is 32 bits on
// Windows (LLP64) and would make a 63-bit sigma a 32-bit one there.

#include <cstdint>
#include <vector>

namespace ecm_rng {

inline std::uint64_t splitmix64_step(std::uint64_t& x) {
    x += 0x9E3779B97f4A7C15ULL;
    std::uint64_t z = x;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

// ceil(bits / 64) words from `seed0`, most significant first.  The first word is
// the first output of the generator, so a value built from them is
// ((...(w0 << 64) + w1) << 64) + ...) for however many words are needed.
inline std::vector<std::uint64_t> random_words(std::uint64_t seed0, unsigned bits) {
    std::vector<std::uint64_t> words;
    std::uint64_t s = seed0;
    for (unsigned i = 0; i < bits; i += 64) words.push_back(splitmix64_step(s));
    return words;
}

#ifdef __GMP_PLUSPLUS__
// The words as one integer.  Each word is imported as a whole 64-bit value
// (mpz_import, not mpz_add_ui / an `unsigned long` conversion).
inline mpz_class random_mpz_bits(std::uint64_t seed0, unsigned bits) {
    mpz_class z = 0;
    for (std::uint64_t w : random_words(seed0, bits)) {
        mpz_class word;
        mpz_import(word.get_mpz_t(), 1, 1, sizeof(w), 0, 0, &w);
        z <<= 64;
        z += word;
    }
    return z;
}
#endif

} // namespace ecm_rng
