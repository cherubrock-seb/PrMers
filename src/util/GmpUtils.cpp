#include "util/GmpUtils.hpp"

namespace util {

mpz_class convertToGMP(const std::vector<uint32_t>& words) {
  mpz_class result;
  // Use GMP's optimized mpz_import function
  mpz_import(result.get_mpz_t(), words.size(), -1 /*order: LSWord first*/, sizeof(uint32_t), 0 /*endian: native*/, 0 /*nails*/, words.data());
  return result;
}

// Optimized modular reduction for Mersenne numbers: x mod (2^E - 1)
// Uses the identity: X mod (2^E - 1) ≡ (Xlo + Xhi) mod (2^E - 1)
mpz_class mersenneReduce(const mpz_class& x, uint32_t E) {
  // For small numbers, use regular mod
  if (mpz_sizeinbase(x.get_mpz_t(), 2) <= E + 1) {
    return x;
  }
  
  // Create Mersenne modulus: 2^E - 1
  mpz_class mersenne_mod = 1;
  mersenne_mod <<= E;
  mersenne_mod -= 1;
  
  // Split x into high and low parts
  // xlo = x & (2^E - 1)  (low E bits)
  mpz_class xlo = x & mersenne_mod;
  
  // xhi = x >> E  (remaining high bits)
  mpz_class xhi = x >> E;
  
  // Add high and low parts
  mpz_class result = xlo + xhi;
  
  // If result >= 2^E - 1, subtract the modulus
  if (result >= mersenne_mod) {
    result -= mersenne_mod;
  }
  
  return result;
}

// Optimized modular exponentiation for Mersenne numbers: base^exp mod (2^E - 1)
// Uses fast Mersenne reduction at each step instead of general division
mpz_class mersennePowMod(const mpz_class& base, uint64_t exp, uint32_t E) {
  if (exp == 0) {
    return mpz_class(1);
  }
  
  if (exp == 1) {
    return mersenneReduce(base, E);
  }
  
  // Initialize result to 1
  mpz_class result = 1;
  
  // Copy base and reduce it
  mpz_class square = mersenneReduce(base, E);
  
  // Binary exponentiation with fast Mersenne reduction
  while (exp > 0) {
    if (exp & 1) {
      // result = result * square mod (2^E - 1)
      mpz_class temp = result * square;
      result = mersenneReduce(temp, E);
    }
    
    exp >>= 1;
    if (exp > 0) {
      // square = square * square mod (2^E - 1)
      mpz_class temp = square * square;
      square = mersenneReduce(temp, E);
    }
  }
  
  return result;
}

std::vector<uint32_t> convertFromGMP(const mpz_class& gmp_val) {
  size_t wordCount = (mpz_sizeinbase(gmp_val.get_mpz_t(), 2) + 31) / 32;
  std::vector<uint32_t> data(wordCount, 0);
  
  // Use GMP's optimized mpz_export function
  size_t actualWords = 0;
  mpz_export(data.data(), &actualWords, -1 /*order: LSWord first*/, sizeof(uint32_t), 0 /*endian: native*/, 0 /*nails*/, gmp_val.get_mpz_t());
  
  // Note: actualWords may be less than wordCount if the number has leading zeros
  // The vector is already zero-initialized, so this is correct
  return data;
}

}
