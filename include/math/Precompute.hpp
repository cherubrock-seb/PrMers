// src/math/Precompute.hpp
#pragma once
#include <vector>
#include <cstdint>

namespace math {

class Precompute {
public:
    explicit Precompute(uint64_t exponent);
    uint32_t getN() const;
    const std::vector<uint64_t>& digitWeight() const;
    const std::vector<uint64_t>& digitInvWeight() const;
    const std::vector<int>&      getDigitWidth() const;
    const std::vector<bool>&     getDigitWidthMask() const;
    const std::vector<uint64_t>& twiddles() const;
    const std::vector<uint64_t>& invTwiddles() const;
    const std::vector<uint64_t>& twiddlesRadix4() const;
    const std::vector<uint64_t>& invTwiddlesRadix4() const;
    const std::vector<uint64_t>& twiddlesRadix5() const;
    const std::vector<uint64_t>& invTwiddlesRadix5() const;
    uint64_t getDigitWidthValue1() const;
    uint64_t getDigitWidthValue2() const;

private:
    uint32_t n_;
    std::vector<uint64_t> digitWeight_;
    std::vector<uint64_t> digitInvWeight_;
    std::vector<int>      digitWidth_;
    std::vector<bool>     digitWidthMask_;
    std::vector<uint64_t> twiddles_;
    std::vector<uint64_t> invTwiddles_;
    std::vector<uint64_t> w4_;
    std::vector<uint64_t> iw4_;
    std::vector<uint64_t> w5_;
    std::vector<uint64_t> iw5_;
    uint64_t digitWidthValue1_{0};
    uint64_t digitWidthValue2_{0};
};

} // namespace math
