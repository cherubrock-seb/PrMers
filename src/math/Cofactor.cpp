#include "math/Cofactor.hpp"
#include "util/GmpUtils.hpp"
#include <gmpxx.h>
#include <iostream>

namespace math {

// Check that factors actually divide the Mersenne number
bool Cofactor::validateFactors(uint32_t exponent, const std::vector<std::string>& factors) {
    mpz_class mersenne = (mpz_class{1} << exponent) - 1; // Compute 2^p - 1
    
    for (const auto& factorStr : factors) {
      if (factorStr.empty()) {
        std::cout << "Factor validation failed: empty factor string" << std::endl;
        return false;
      }
      
      mpz_class factor{factorStr};
      if (factor <= 1) {
        std::cout << "Factor validation failed: factor " << factorStr << " <= 1" << std::endl;
        return false;
      }
      
      if (mersenne % factor != 0) {
        std::cout << "Factor validation failed: " << factorStr << " does not divide 2^" << exponent << "-1" << std::endl;
        return false;
      }      
    }
    
    return true;
}

} // namespace math
