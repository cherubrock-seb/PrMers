#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace math {

// Utilities for Mersenne cofactor tests
class Cofactor {
public:
    // Check that factors actually divide the Mersenne number
    static bool validateFactors(uint32_t exponent, const std::vector<std::string>& factors);
};

} // namespace math
