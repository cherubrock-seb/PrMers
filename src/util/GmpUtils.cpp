#include "util/GmpUtils.hpp"
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <climits>

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

static inline void add_ui64(mpz_class& x, uint64_t w) {
#if ULONG_MAX == 0xFFFFFFFFFFFFFFFFULL
    mpz_add_ui(x.get_mpz_t(), x.get_mpz_t(), static_cast<unsigned long>(w));
#else
    mpz_class tmp;
    mpz_import(tmp.get_mpz_t(), 1, -1, sizeof(uint64_t), 0, 0, &w);
    x += tmp;
#endif
}

mpz_class vectToMpz(const std::vector<uint64_t>& v,
                    const std::vector<int>& widths,
                    const mpz_class& Mp)
{
    const size_t n = v.size();
    const unsigned T = std::max(1u, std::thread::hardware_concurrency());
    std::vector<mpz_class> partial(T);

    std::vector<unsigned> total_width(T, 0);
    std::atomic<size_t> global_count{0};

    std::vector<std::thread> threads(T);
    size_t chunk = (n + T - 1) / T;

    for (unsigned t = 0; t < T; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, n);
        threads[t] = std::thread([&, start, end, t]() {
            mpz_class acc = 0;
            for (ptrdiff_t i = ptrdiff_t(end) - 1; i >= ptrdiff_t(start); --i) {
                acc <<= static_cast<mp_bitcnt_t>(widths[static_cast<size_t>(i)]);
                add_ui64(acc, v[static_cast<size_t>(i)]);
                if (acc >= Mp) acc -= Mp;

                total_width[t] += static_cast<unsigned>(widths[static_cast<size_t>(i)]);

                size_t count = ++global_count;
                if (count % 10000 == 0 || count == n) {
                    double progress = 100.0 * count / n;
                    printf("\rProgress: %.2f%%", progress);
                    fflush(stdout);
                }
            }
            partial[t] = acc;
        });
    }

    for (auto& th : threads) th.join();
    printf("\n");

    mpz_class result = 0;
    for (unsigned t = T; t-- > 0;) {
        result <<= total_width[t];
        result += partial[t];
        if (result >= Mp) result -= Mp;
    }


    return result;
}


}
