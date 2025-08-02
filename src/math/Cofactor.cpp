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

// Check Type 5 residue:
// KF = product(known_factors)
// N = (2^p - 1) / KF
// residue == base^(KF-1) (mod N)
bool Cofactor::isCofactorPRP(uint32_t exponent,
                             const std::vector<std::string>& factors,
                             const std::vector<uint32_t>& finalResidue,
                             uint32_t base) {  
  try {
    mpz_class finalRes = util::convertToGMP(finalResidue);
    mpz_class baseGmp{base};
    
    mpz_class knownFactorsProduct{1};
    for (const auto& factorStr : factors) {
      mpz_class factor{factorStr};
      knownFactorsProduct *= factor;
    }
    
    mpz_class mersenne = (mpz_class{1} << exponent) - 1;
    mpz_class cofactor = mersenne / knownFactorsProduct;
    
    mpz_class kfMinus1 = knownFactorsProduct - 1;
    mpz_class expected;
    mpz_powm(expected.get_mpz_t(), baseGmp.get_mpz_t(), kfMinus1.get_mpz_t(), cofactor.get_mpz_t());
    
    mpz_class actual = finalRes % cofactor;
    bool isPrime = (actual == expected);
    
    return isPrime;
  } catch (const std::exception& e) {
    std::cout << "Cofactor PRP error:" << e.what() << std::endl;
    return false;
  }
}

} // namespace math
