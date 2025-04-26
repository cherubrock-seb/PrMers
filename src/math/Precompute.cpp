// src/math/Precompute.cpp
#include "math/Precompute.hpp"
#include "math/Mod64.hpp"
#include <iostream>
namespace math {

uint32_t transformsize(uint32_t exponent) {
    int    log_n = 0;
    uint32_t w   = 0;
    do {
        ++log_n;
        w = exponent >> log_n;
    } while ((w + 1) * 2 + log_n >= 63);
    return uint32_t(1) << log_n;
}

void precalc_for_p(uint32_t p,
                   std::vector<uint64_t>& digitWeight,
                   std::vector<uint64_t>& digitInvWeight,
                   std::vector<int>&      digitWidth,
                   std::vector<uint64_t>& twiddles,
                   std::vector<uint64_t>& invTwiddles)
{
    uint32_t n = transformsize(p);
    std::cout << "Transform Size = " << n << std::endl;
    
    if (n < 4) n = 4;

    __uint128_t bigPminus1 = ( (__uint128_t)MOD_P - 1 );
    __uint128_t tmp        = bigPminus1 / 192;
    tmp                   /= n;
    uint64_t exponent      = (uint64_t)(tmp * 5ULL);
    uint64_t nr2           = powModP(7ULL, exponent);

    uint64_t inv_n = invModP(n);
    digitWeight[0]    = 1ULL;
    digitInvWeight[0] = inv_n;

    uint32_t prev = 0;
    for (uint32_t j = 1; j <= n; ++j) {
        uint64_t qj = uint64_t(p) * j;
        uint64_t qq = qj - 1;
        uint32_t ceil_qj_n = uint32_t(qq / n + 1);
        digitWidth[j - 1]  = int(ceil_qj_n - prev);
        prev               = ceil_qj_n;

        if (j < n) {
            uint32_t r = uint32_t(qj % n);
            // 2) use 'nr2' here, _not_ the primitive-root:
            uint64_t nr2r = r
                ? powModP(nr2, (uint64_t)(n - r))
                : 1ULL;
            digitWeight[j]    = nr2r;
            // exactly as before
            digitInvWeight[j] = mulModP(invModP(nr2r), inv_n);
        }
    }


    twiddles.assign(3 * n, 0);
    invTwiddles.assign(3 * n, 0);

    uint64_t root    = powModP(7ULL, (MOD_P - 1) / n);
    uint64_t invroot = invModP(root);

    for (uint32_t m = n >> 1, s = 1; m >= 1; m >>= 1, s <<= 1) {
        uint64_t r_s   = powModP(root,    s);
        uint64_t ir_s  = powModP(invroot, s);
        uint64_t w     = 1, invw = 1;

        for (uint32_t j = 0; j < m; ++j) {
            uint32_t idx = 3 * (m + j);
            twiddles   [idx + 0] = w;
            uint64_t w2         = mulModP(w, w);
            twiddles   [idx + 1] = w2;
            twiddles   [idx + 2] = mulModP(w2, w);

            invTwiddles[idx + 0] = invw;
            uint64_t iw2         = mulModP(invw, invw);
            invTwiddles[idx + 1] = iw2;
            invTwiddles[idx + 2] = mulModP(iw2, invw);

            w    = mulModP(w,    r_s);
            invw = mulModP(invw, ir_s);
        }
    }
}


Precompute::Precompute(uint32_t exponent)
  : n_{ transformsize(exponent) }
, digitWeight_()
, digitInvWeight_()
, digitWidth_()
, twiddles_()
, invTwiddles_()
{
    if (n_ < 4) n_ = 4;
    digitWeight_.resize(n_);
    digitInvWeight_.resize(n_);
    digitWidth_   .resize(n_);
    twiddles_     .resize(3 * n_);
    invTwiddles_  .resize(3 * n_);
    precalc_for_p(exponent,
                  digitWeight_,
                  digitInvWeight_,
                  digitWidth_,
                  twiddles_,
                  invTwiddles_);
}

uint32_t Precompute::getN() const { return n_; }
const std::vector<uint64_t>& Precompute::digitWeight() const { return digitWeight_; }
const std::vector<uint64_t>& Precompute::digitInvWeight() const { return digitInvWeight_; }
const std::vector<int>& Precompute::getDigitWidth() const { return digitWidth_; }
const std::vector<uint64_t>& Precompute::twiddles() const { return twiddles_; }
const std::vector<uint64_t>& Precompute::invTwiddles() const { return invTwiddles_; }

} // namespace math
