// include/math/Precompute.hpp
#pragma once
#include <cstdint>
#include <vector>

namespace math {
uint32_t transformsize(uint32_t exponent);

void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digitWeight,
                   std::vector<uint64_t>& digitInvWeight,
                   std::vector<int>&      digitWidth,
                   std::vector<uint64_t>& twiddles,
                   std::vector<uint64_t>& invTwiddles);

class Precompute {
public:
    explicit Precompute(uint32_t exponent);
    uint32_t                    getN()           const;
    const std::vector<uint64_t>& digitWeight()    const;
    const std::vector<uint64_t>& digitInvWeight() const;
    const std::vector<int>&      getDigitWidth()  const;
    const std::vector<uint64_t>& twiddles()       const;
    const std::vector<uint64_t>& invTwiddles()    const;

private:
    uint32_t            n_;
    std::vector<uint64_t> digitWeight_;
    std::vector<uint64_t> digitInvWeight_;
    std::vector<int>     digitWidth_;
    std::vector<uint64_t> twiddles_;
    std::vector<uint64_t> invTwiddles_;
};

} // namespace math
