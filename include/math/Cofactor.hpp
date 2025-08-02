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
    
    // Check if the cofactor is PRP based on the final computed residue
    // KF = product(known_factors)
    // N = (2^p - 1) / KF
    // residue == base^(KF-1) (mod N)
    static bool isCofactorPRP(uint32_t exponent,
                              const std::vector<std::string>& factors,
                              const std::vector<uint32_t>& finalResidue,
                              uint32_t base = 3);

};

} // namespace math
