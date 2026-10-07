// Host test: JsonBuilder::compactBits must keep the value modulo 2^E - 1 when
// the top digit is not normalised (the carry out of the top wraps to bit 0).
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <gmpxx.h>

#include "io/JsonBuilder.hpp"

namespace {

mpz_class digitsValue(const std::vector<uint64_t>& x, const std::vector<int>& w) {
    mpz_class acc = 0;
    unsigned shift = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        mpz_class d;
        mpz_import(d.get_mpz_t(), 1, -1, sizeof(uint64_t), 0, 0, &x[i]);
        acc += d << shift;
        shift += static_cast<unsigned>(w[i]);
    }
    return acc;
}

int failures = 0;

void check(const std::vector<uint64_t>& x, const std::vector<int>& w, uint32_t E,
           const char* what) {
    const mpz_class Mp = (mpz_class(1) << E) - 1;
    const auto words = io::JsonBuilder::compactBits(x, w, E);
    mpz_class got;
    mpz_import(got.get_mpz_t(), words.size(), -1, sizeof(uint32_t), 0, 0, words.data());
    if (got >= (mpz_class(1) << E) || got % Mp != digitsValue(x, w) % Mp) {
        std::cerr << what << ": compactBits value differs mod 2^E - 1\n";
        ++failures;
    }
}

} // namespace

int main() {
    const uint32_t E = 89;
    const std::vector<int> w{18, 18, 18, 18, 17};
    const uint64_t m17 = (uint64_t(1) << 17) - 1;

    // Normalised input.
    check({1, 2, 3, 4, 5}, w, E, "normalised");
    // The top digit is above its width: its carry wraps to bit 0, not bit 32.
    check({1, 2, 3, 4, m17 + 3}, w, E, "top digit over width");
    check({0, 0, 0, 0, (uint64_t(1) << 17) + 1}, w, E, "top digit carry only");
    check({(uint64_t(1) << 18) - 1, (uint64_t(1) << 18) - 1, (uint64_t(1) << 18) - 1,
           (uint64_t(1) << 18) - 1, m17 + 5}, w, E, "carry ripples to the top");

    std::mt19937_64 rng(7);
    for (int t = 0; t < 5000 && failures < 5; ++t) {
        std::vector<uint64_t> x(w.size());
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = rng() & ((uint64_t(1) << (w[i] + 2)) - 1);
        check(x, w, E, "random");
    }

    // An exponent that is a multiple of 32 (no spare bits in the top word).
    {
        const uint32_t E2 = 96;
        const std::vector<int> w2{24, 24, 24, 24};
        const uint64_t m24 = (uint64_t(1) << 24) - 1;
        check({m24, m24, m24, m24}, w2, E2, "E=96 all ones");
        check({m24, m24, m24, (uint64_t(1) << 24) + 5}, w2, E2, "E=96 top over width");
        for (int t = 0; t < 5000 && failures < 5; ++t) {
            std::vector<uint64_t> x(w2.size());
            for (auto& d : x) d = rng() & ((uint64_t(1) << 26) - 1);
            check(x, w2, E2, "E=96 random");
        }
    }

    if (failures) {
        std::cerr << "compactBits wraparound regression: FAIL\n";
        return 1;
    }
    std::cout << "compactBits wraparound regression: PASS\n";
    return 0;
}
