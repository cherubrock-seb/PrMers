// Host test for math::Carry::handleFinalCarry (no OpenCL device needed).
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <gmpxx.h>

#include "math/Carry.hpp"

// Carry.cpp is linked for handleFinalCarry only; the carry kernels that call
// this are never run here.
std::size_t prmers::ocl::Context::getWorkersCarry() const noexcept { return 0; }

namespace {

mpz_class valueMod(const std::vector<uint64_t>& x, const std::vector<int>& w,
                   const mpz_class& Mp) {
    mpz_class acc = 0;
    unsigned shift = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        mpz_class d;
        mpz_import(d.get_mpz_t(), 1, -1, sizeof(uint64_t), 0, 0, &x[i]);
        acc += d << shift;
        shift += static_cast<unsigned>(w[i]);
    }
    return acc % Mp;
}

int failures = 0;

void check(std::vector<uint64_t> x, const std::vector<int>& w,
           const mpz_class& Mp, const char* what) {
    const mpz_class want = valueMod(x, w, Mp);
    math::Carry::handleFinalCarry(x, w);
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] >> w[i]) {
            std::cerr << what << ": digit " << i << " not reduced: 0x"
                      << std::hex << x[i] << std::dec << "\n";
            ++failures;
            return;
        }
    }
    if (valueMod(x, w, Mp) != want) {
        std::cerr << what << ": value changed\n";
        ++failures;
    }
}

} // namespace

int main() {
    // 2^89 - 1 split into five digits.
    const std::vector<int> w{18, 18, 18, 18, 17};
    const mpz_class Mp = (mpz_class(1) << 89) - 1;
    const uint64_t m18 = (uint64_t(1) << 18) - 1;
    const uint64_t m17 = (uint64_t(1) << 17) - 1;

    // Low digit all ones: after adding 1 and normalising, digit 0 is 0 and
    // taking the 1 back off used to wrap it to 2^64-1.
    check({m18, 5, 6, 7, 8}, w, Mp, "low digit all ones");
    check({m18, 0, 0, 0, 0}, w, Mp, "only low digit all ones");
    check({m18, m18, 0, 0, 0}, w, Mp, "two low digits all ones");
    check({m18, m18, m18, m18, m17}, w, Mp, "all digits all ones");
    check({m18, 0, 0, 0, m17 - 1}, w, Mp, "borrow across zero digits");
    // Values congruent to 0 and to -1.
    check({0, 0, 0, 0, 0}, w, Mp, "zero");
    check({m18 - 1, m18, m18, m18, m17}, w, Mp, "2^p - 2");

    // Unnormalised digits, as left by the transform; force digit 0 to all
    // ones for a third of them.
    std::mt19937_64 rng(12345);
    for (int t = 0; t < 20000 && failures < 5; ++t) {
        std::vector<uint64_t> x(w.size());
        for (auto& d : x) d = rng() >> (rng() % 40);
        if (t % 3 == 0) x[0] = m18;
        check(x, w, Mp, "random");
    }

    if (failures) {
        std::cerr << "handleFinalCarry digit 0 regression: FAIL\n";
        return 1;
    }
    std::cout << "handleFinalCarry digit 0 regression: PASS\n";
    return 0;
}
