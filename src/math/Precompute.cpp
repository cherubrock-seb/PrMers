/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
// src/math/Precompute.cpp
#include "math/Precompute.hpp"
#include "math/Mod64.hpp"
#include <iostream>
#include <cstdint>
#include <cmath>

namespace math {

uint32_t transformsize(uint32_t exponent) {
    int log_n = 0;
    uint32_t w = 0;
    do {
        ++log_n;
        w = exponent >> log_n;
    } while ((w + 1) * 2 + log_n >= 63);

    uint32_t n2 = uint32_t(1) << log_n;

    if (n2 >= 8) {
        uint32_t n5 = (n2 >> 3) * 5u;          // n2 * 5 / 8
        uint32_t w5 = exponent / n5;
        long double cost5 = std::log2((long double)n5) + 2.0L * (w5 + 1);
        if (cost5 < 64.0L)
            return n5;
    }
    return n2;
}
static void prepare_radix_twiddles(uint32_t n,
                                   std::vector<uint64_t>& w4,
                                   std::vector<uint64_t>& iw4,
                                   std::vector<uint64_t>& w5,
                                   std::vector<uint64_t>& iw5)
{
    uint64_t root = powModP(7ULL, (MOD_P - 1) / n);
    uint64_t invroot = invModP(root);
    uint32_t m  = n / 5;
    if(m==0){
        m=1;
    }
    w5.resize(4 * m);
    iw5.resize(4 * m);
    if(n%5==0){
        for (uint32_t j = 0; j < m; ++j) {
            uint64_t w1  = powModP(root, j);
            uint64_t iw1 = powModP(invroot, j);
            uint64_t w2  = mulModP(w1,  w1);
            uint64_t iw2 = mulModP(iw1, iw1);
            uint64_t w3  = mulModP(w2,  w1);
            uint64_t iw3 = mulModP(iw2, iw1);
            uint64_t w4v = mulModP(w3,  w1);
            uint64_t iw4v= mulModP(iw3, iw1);
            w5 [4*j]     = w1;  w5 [4*j+1] = w2;  w5 [4*j+2] = w3;  w5 [4*j+3] = w4v;
            iw5[4*j]     = iw1; iw5[4*j+1] = iw2; iw5[4*j+2] = iw3; iw5[4*j+3] = iw4v;
        }
    }
    /*std::cout << "w5:\n";
    for (uint32_t j = 0; j < m; ++j) {
        std::cout << "j = " << j << " : ";
        std::cout << w5[4 * j] << ", " << w5[4 * j + 1] << ", "
                << w5[4 * j + 2] << ", " << w5[4 * j + 3] << '\n';
    }*/
    uint32_t n5 = n;

    
    if(n%5==0){
        n5 = n/5;
    }
    m = n5;
    w4.resize(3 * n5);
    iw4.resize(3 * n5);
    for (size_t m = n5 / 2, s = 1; m >= 1; m /= 2, s *= 2){
        root = powModP(7ULL, (MOD_P - 1) / (2*m));
        invroot = invModP(root);
        for (size_t j = 0; j < m; j++)
		{
            uint64_t r1  = powModP(root, j);
            uint64_t ir1 = powModP(invroot, j);
            uint64_t r2  = mulModP(r1,  r1);
            uint64_t ir2 = mulModP(ir1, ir1);
            uint64_t r3  = mulModP(r2,  r1);
            uint64_t ir3 = mulModP(ir2, ir1);
            w4 [3 * (m + j) + 0]     = r1;  w4 [3 * (m + j) + 1] = r2;  w4 [3 * (m + j) + 2] = r3;
            iw4[3 * (m + j) + 0]     = ir1; iw4[3 * (m + j) + 1] = ir2; iw4[3 * (m + j) + 2] = ir3;
        }

    }
}



void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digitWeight,
                   std::vector<uint64_t>& digitInvWeight,
                   std::vector<int>&      digitWidth,
                   std::vector<uint64_t>& twiddles,
                   std::vector<uint64_t>& invTwiddles,
                   uint64_t& digitWidthValue1,
                   uint64_t& digitWidthValue2,
                   std::vector<bool>& digitWidthMask

                   )
{
    uint32_t n = transformsize(p);
    std::cout << "Transform Size = " << n << std::endl;
    
    if (n < 4) n = 4;

    #ifdef _MSC_VER
        uint64_t high = 0, low = MOD_P - 1ULL;
        uint64_t tmp1 = _udiv128(high, low, 192ULL, &low);
        uint64_t tmp2 = tmp1 / n;
        uint64_t exponent = tmp2 * 5ULL;
    #else
        __uint128_t bigPminus1 = (__uint128_t)MOD_P - 1ULL;
        __uint128_t tmp = bigPminus1 / 192ULL;
        tmp /= n;
        uint64_t exponent = (uint64_t)(tmp * 5ULL);
    #endif

    uint64_t nr2 = powModP(7ULL, exponent);
    uint64_t inv_n = invModP(n);
    digitWeight[0]    = 1ULL;
    digitInvWeight[0] = inv_n;

    uint32_t prev = 0;
    for (uint32_t j = 1; j <= n; ++j) {
        uint64_t qj = uint64_t(p) * j;
        //uint64_t qq = qj - 1;
        uint32_t ceil_qj_n = (qj == 0) ? 0 : uint32_t((qj - 1) / n + 1);
        digitWidth[j - 1]  = int(ceil_qj_n - prev);
        prev               = ceil_qj_n;

        if (j < n) {
            uint32_t r = uint32_t(qj % n);
            uint64_t nr2r = r
                ? powModP(nr2, (uint64_t)(n - r))
                : 1ULL;
            digitWeight[j]    = nr2r;
            digitInvWeight[j] = mulModP(invModP(nr2r), inv_n);
        }
    }
    uint64_t w1 = static_cast<uint64_t>(digitWidth[0]);
    uint64_t w2 = 0;
    for (int w : digitWidth) {
        if (uint64_t(w) != w1) {
            w2 = uint64_t(w);
            break;
        }
    }
    digitWidthValue1 = w1;
    digitWidthValue2 = w2;

    digitWidthMask.resize(n);
    for (size_t i = 0; i < n; ++i) {
        digitWidthMask[i] = (uint64_t(digitWidth[i]) == w2);
    }


    if(n%5 == 0){
        n = n/5;
    }


}


Precompute::Precompute(uint32_t exponent)
  : n_{ transformsize(exponent) }
, digitWeight_()
, digitInvWeight_()
, digitWidth_()
, twiddles_()
, invTwiddles_()
, w4_()
, iw4_()
, w5_()
, iw5_()
{
    if (n_ < 4) n_ = 4;
    digitWeight_.resize(n_);
    digitInvWeight_.resize(n_);
    digitWidth_   .resize(n_);
    digitWidthMask_   .resize(n_);
    twiddles_     .resize(3 * n_);
    invTwiddles_  .resize(3 * n_);
    precalc_for_p(exponent,
                  digitWeight_,
                  digitInvWeight_,
                  digitWidth_,
                  twiddles_,
                  invTwiddles_,
                  digitWidthValue1_,
                  digitWidthValue2_,
                  digitWidthMask_
                  );
    prepare_radix_twiddles(n_, w4_, iw4_, w5_, iw5_);
    
}

uint32_t Precompute::getN() const { return n_; }
const std::vector<uint64_t>& Precompute::digitWeight() const { return digitWeight_; }
const std::vector<uint64_t>& Precompute::digitInvWeight() const { return digitInvWeight_; }
const std::vector<int>& Precompute::getDigitWidth() const { return digitWidth_; }
const std::vector<uint64_t>& Precompute::twiddles() const { return twiddles_; }
const std::vector<uint64_t>& Precompute::invTwiddles() const { return invTwiddles_; }
uint64_t Precompute::getDigitWidthValue1() const {return digitWidthValue1_;}
uint64_t Precompute::getDigitWidthValue2() const {return digitWidthValue2_;}
const std::vector<bool>& Precompute::getDigitWidthMask() const {return digitWidthMask_;}
const std::vector<uint64_t>&          Precompute::twiddlesRadix4()      const { return w4_; }
const std::vector<uint64_t>&          Precompute::invTwiddlesRadix4()   const { return iw4_; }
const std::vector<uint64_t>&          Precompute::twiddlesRadix5()      const { return w5_; }
const std::vector<uint64_t>&          Precompute::invTwiddlesRadix5()   const { return iw5_; }
} // namespace math
