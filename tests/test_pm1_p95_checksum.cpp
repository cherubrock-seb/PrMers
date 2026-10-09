// The Prime95 save-file checksum is a 32-bit sum of the file's fields, where
// Prime95 adds the low 32 bits of (high word + low word) for every 64-bit
// field.  B1 is stored twice (B_done and C_done), so its contribution is
// 2 * (hi + lo) mod 2^32 -- not 2*B1, which loses the high word once
// B1 >= 2^32.
#include "core/AlgoUtils.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;

static void check(uint64_t B1, const std::vector<uint8_t>& data) {
    const uint32_t base = core::algo::checksum_prime95_s1(0, data);
    const uint32_t got = core::algo::checksum_prime95_s1(B1, data);
    const uint32_t b1_term = static_cast<uint32_t>(2u * ((B1 >> 32) + B1));
    const uint32_t want = base + b1_term;
    if (got != want) {
        std::printf("B1=%llu: checksum %u, expected %u\n", static_cast<unsigned long long>(B1), got, want);
        ++failures;
    }
}

int main() {
    std::vector<uint8_t> data(64);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i * 37 + 11);
    const uint64_t cases[] = {1, 1000, 4294967295ULL, 4294967296ULL, 4294967296ULL + 5,
                              (5ULL << 32) + 12345, 0xFFFFFFFFFFFFFFFFULL};
    for (uint64_t B1 : cases) check(B1, data);
    if (failures) return EXIT_FAILURE;
    std::puts("prime95 checksum test passed");
    return EXIT_SUCCESS;
}
