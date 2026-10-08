// GMP-free half of the ECM random-word test, so it can be built for Windows
// (MinGW, LLP64: unsigned long is 32 bits) and run under Wine.  It prints the
// words ecm_rng::random_words gives and what the old `(unsigned long)` narrowing
// would have kept; the shell test checks the words are identical to the native
// (LP64) build's and that the narrowing is lossy where unsigned long is 32 bits.
#include "math/EcmRandom.hpp"

#include <cstdint>
#include <cstdio>

int main() {
    std::printf("sizeof_unsigned_long=%zu\n", sizeof(unsigned long));
    const std::uint64_t seeds[] = {1ULL, 111ULL, 0xDEADBEEFCAFEF00DULL, 0xFFFFFFFFFFFFFFFFULL, 0ULL};
    const unsigned bit_counts[] = {63u, 192u, 256u};
    for (std::uint64_t seed : seeds) {
        for (unsigned bits : bit_counts) {
            std::printf("seed=%llu bits=%u words=", static_cast<unsigned long long>(seed), bits);
            for (std::uint64_t w : ecm_rng::random_words(seed, bits))
                std::printf("%016llx ", static_cast<unsigned long long>(w));
            std::printf("\n");
        }
    }
    // The old code kept (unsigned long)word: lossless only where unsigned long is 64 bits.
    std::uint64_t s = 1ULL;
    const std::uint64_t word = ecm_rng::splitmix64_step(s);
    const unsigned long narrowed = static_cast<unsigned long>(word);
    std::printf("narrowing_lossless=%d\n", static_cast<std::uint64_t>(narrowed) == word ? 1 : 0);
    return 0;
}
