// include/math/Precompute.hpp
#pragma once
#include <cstdint>
#include <vector>

namespace math {
uint32_t transformsize(uint64_t exponent);

void precalc_for_p(uint64_t p,
                   std::vector<uint64_t>& digitWeight,
                   std::vector<uint64_t>& digitInvWeight,
                   std::vector<int>&      digitWidth,
                   std::vector<uint64_t>& twiddles,
                   std::vector<uint64_t>& invTwiddles);

class Precompute {
public:
    explicit Precompute(uint64_t exponent);
    uint32_t                    getN()           const;
    const std::vector<uint64_t>& digitWeight()    const;
    const std::vector<uint64_t>& digitInvWeight() const;
    const std::vector<int>&      getDigitWidth()  const;
    const std::vector<uint64_t>& twiddles()       const;
    const std::vector<uint64_t>& invTwiddles()    const;
    uint64_t getDigitWidthValue1() const;
    uint64_t getDigitWidthValue2() const;
    const std::vector<bool>& getDigitWidthMask() const;
    const std::vector<uint64_t>& twiddlesRadix4() const;
    const std::vector<uint64_t>& invTwiddlesRadix4() const;
    const std::vector<uint64_t>& twiddlesRadix5() const;
    const std::vector<uint64_t>& invTwiddlesRadix5() const;

private:
    uint32_t            n_;
    std::vector<uint64_t> digitWeight_;
    std::vector<uint64_t> digitInvWeight_;
    std::vector<int>     digitWidth_;
    std::vector<uint64_t> twiddles_;
    std::vector<uint64_t> invTwiddles_;
    uint64_t                  digitWidthValue1_;
    uint64_t                  digitWidthValue2_;
    std::vector<bool>         digitWidthMask_;
    std::vector<uint64_t> w4_, iw4_, w5_, iw5_;
};

} // namespace math
