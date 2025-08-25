// src/math/Precompute.cpp
#include "math/Precompute.hpp"
#include "math/Mod64.hpp"
#include <cstdint>
#include <cmath>
#include <iostream>

namespace math {

static uint32_t transformsize(uint64_t exponent) {
    uint64_t log_n = 0;
    uint64_t w = 0;
    do {
        ++log_n;
        w = exponent >> log_n;
    } while ((w + 1) * 2 + log_n >= 63);
    uint64_t n2 = uint64_t(1) << log_n;
    if (n2 >= 128) {
        uint64_t n5 = (n2 >> 3) * 5u;
        if (n5 >= 80) {
            uint64_t w5 = exponent / n5;
            long double cost5 = std::log2((long double)n5) + 2.0L * (w5 + 1);
            if (cost5 < 64.0L) return n5;
        }
    }
    if (exponent > 1207959503) n2 = (n2 / 4) * 5;
    return (uint32_t)n2;
}

struct Pair { uint64_t x, y; };

static inline uint64_t add_m31(uint64_t a, uint64_t b) {
    const uint64_t p = (1ULL << 31) - 1;
    uint64_t t = a + b;
    if (t >= p) t -= p;
    return t;
}
static inline uint64_t sub_m31(uint64_t a, uint64_t b) {
    const uint64_t p = (1ULL << 31) - 1;
    return (a >= b) ? (a - b) : (a + p - b);
}
static inline uint64_t mul_m31(uint64_t a, uint64_t b) {
    const uint64_t p = (1ULL << 31) - 1;
    __uint128_t t = (__uint128_t)a * b;
    uint64_t s = (uint64_t)t & p;
    s += (uint64_t)(t >> 31);
    if (s >= p) s -= p;
    return s;
}
static inline uint64_t pow_m31(uint64_t a, uint64_t e) {
    const uint64_t p = (1ULL << 31) - 1;
    uint64_t r = 1, x = a % p;
    while (e) {
        if (e & 1) r = mul_m31(r, x);
        x = mul_m31(x, x);
        e >>= 1;
    }
    return r;
}
static inline uint64_t inv_m31(uint64_t a) {
    const uint64_t p = (1ULL << 31) - 1;
    return pow_m31(a, p - 2);
}

static inline uint64_t add_m61(uint64_t a, uint64_t b) {
    const uint64_t p = (1ULL << 61) - 1;
    uint64_t t = a + b;
    if (t >= p) t -= p;
    return t;
}
static inline uint64_t sub_m61(uint64_t a, uint64_t b) {
    const uint64_t p = (1ULL << 61) - 1;
    return (a >= b) ? (a - b) : (a + p - b);
}
static inline uint64_t mul_m61(uint64_t a, uint64_t b) {
    const uint64_t p = (1ULL << 61) - 1;
    __uint128_t t = (__uint128_t)a * b;
    uint64_t s = ((uint64_t)t & p) + (uint64_t)(t >> 61);
    if (s >= p) s -= p;
    return s;
}
static inline uint64_t pow_m61(uint64_t a, uint64_t e) {
    const uint64_t p = (1ULL << 61) - 1;
    uint64_t r = 1, x = a % p;
    while (e) {
        if (e & 1) r = mul_m61(r, x);
        x = mul_m61(x, x);
        e >>= 1;
    }
    return r;
}
static inline uint64_t inv_m61(uint64_t a) {
    const uint64_t p = (1ULL << 61) - 1;
    return pow_m61(a, p - 2);
}

static inline Pair gf_add31(Pair a, Pair b){ return { add_m31(a.x,b.x), add_m31(a.y,b.y) }; }
static inline Pair gf_sub31(Pair a, Pair b){ return { sub_m31(a.x,b.x), sub_m31(a.y,b.y) }; }
static inline Pair gf_mul31(Pair a, Pair b){
    uint64_t ac = mul_m31(a.x,b.x);
    uint64_t bd = mul_m31(a.y,b.y);
    uint64_t bc = mul_m31(a.y,b.x);
    uint64_t ad = mul_m31(a.x,b.y);
    return { sub_m31(ac, bd), add_m31(bc, ad) };
}
static inline Pair gf_pow31(Pair a, uint64_t e){
    Pair r{1,0}, x=a;
    while(e){
        if(e&1) r = gf_mul31(r,x);
        x = gf_mul31(x,x);
        e >>= 1;
    }
    return r;
}
static inline Pair gf_inv31(Pair a){
    uint64_t aa = mul_m31(a.x,a.x);
    uint64_t bb = mul_m31(a.y,a.y);
    uint64_t den = add_m31(aa, bb);
    uint64_t inv = inv_m31(den);
    return { mul_m31(a.x,inv), sub_m31(0, mul_m31(a.y,inv)) };
}

static inline Pair gf_add61(Pair a, Pair b){ return { add_m61(a.x,b.x), add_m61(a.y,b.y) }; }
static inline Pair gf_sub61(Pair a, Pair b){ return { sub_m61(a.x,b.x), sub_m61(a.y,b.y) }; }
static inline Pair gf_mul61(Pair a, Pair b){
    uint64_t ac = mul_m61(a.x,b.x);
    uint64_t bd = mul_m61(a.y,b.y);
    uint64_t bc = mul_m61(a.y,b.x);
    uint64_t ad = mul_m61(a.x,b.y);
    return { sub_m61(ac, bd), add_m61(bc, ad) };
}
static inline Pair gf_pow61(Pair a, uint64_t e){
    Pair r{1,0}, x=a;
    while(e){
        if(e&1) r = gf_mul61(r,x);
        x = gf_mul61(x,x);
        e >>= 1;
    }
    return r;
}
static inline Pair gf_inv61(Pair a){
    uint64_t aa = mul_m61(a.x,a.x);
    uint64_t bb = mul_m61(a.y,a.y);
    uint64_t den = add_m61(aa, bb);
    uint64_t inv = inv_m61(den);
    return { mul_m61(a.x,inv), sub_m61(0, mul_m61(a.y,inv)) };
}

static void prepare_radix_twiddles(uint32_t n,
                                   std::vector<uint64_t>& w4,
                                   std::vector<uint64_t>& iw4,
                                   std::vector<uint64_t>& w5,
                                   std::vector<uint64_t>& iw5)
{
    //auto mode = getModMode();
    auto mode = 0;
    if (mode == 0) {
        uint64_t root = powModP(7ULL, (MOD_P - 1) / n);
        uint64_t invroot = invModP(root);
        uint32_t m = n / 5;
        if (m == 0) m = 1;
        w5.resize(4 * m);
        iw5.resize(4 * m);
        if (n % 5 == 0) {
            for (uint32_t j = 0; j < m; ++j) {
                uint64_t w1  = powModP(root, j);
                uint64_t iw1 = powModP(invroot, j);
                uint64_t w2  = mulModP(w1,  w1);
                uint64_t iw2 = mulModP(iw1, iw1);
                uint64_t w3  = mulModP(w2,  w1);
                uint64_t iw3 = mulModP(iw2, iw1);
                uint64_t w4v = mulModP(w2,  w2);
                uint64_t iw4v= mulModP(iw2, iw2);
                w5 [4*j+0] = w1; w5 [4*j+1] = w2; w5 [4*j+2] = w3; w5 [4*j+3] = w4v;
                iw5[4*j+0] = iw1; iw5[4*j+1] = iw2; iw5[4*j+2] = iw3; iw5[4*j+3] = iw4v;
            }
        }
        uint32_t n5 = n;
        if (n % 5 == 0) n5 = n / 5;
        w4.resize(3 * n5);
        iw4.resize(3 * n5);
        for (size_t m2 = n5 / 2; m2 >= 1; m2 /= 2) {
            uint64_t r = powModP(7ULL, (MOD_P - 1) / (2 * m2));
            uint64_t ir = invModP(r);
            for (size_t j = 0; j < m2; ++j) {
                uint64_t r1  = powModP(r, j);
                uint64_t ir1 = powModP(ir, j);
                uint64_t r2  = mulModP(r1, r1);
                uint64_t ir2 = mulModP(ir1, ir1);
                uint64_t r3  = mulModP(r2, r1);
                uint64_t ir3 = mulModP(ir2, ir1);
                w4 [3 * (m2 + j) + 0] = r1;  w4 [3 * (m2 + j) + 1] = r2;  w4 [3 * (m2 + j) + 2] = r3;
                iw4[3 * (m2 + j) + 0] = ir1; iw4[3 * (m2 + j) + 1] = ir2; iw4[3 * (m2 + j) + 2] = ir3;
            }
            if (m2 == 1) break;
        }
        return;
    }

    if (mode == 31) {
        const Pair h{7735ULL, 748621ULL};
        const uint64_t order = 1ULL << 32;
        Pair root = gf_pow31(h, order / n);
        Pair invroot = gf_inv31(root);
        uint32_t m = n / 5;
        if (m == 0) m = 1;
        w5.resize(8 * m);
        iw5.resize(8 * m);
        if (n % 5 == 0) {
            for (uint32_t j = 0; j < m; ++j) {
                Pair w1  = gf_pow31(root, j);
                Pair iw1 = gf_pow31(invroot, j);
                Pair w2  = gf_mul31(w1, w1);
                Pair iw2 = gf_mul31(iw1, iw1);
                Pair w3  = gf_mul31(w2, w1);
                Pair iw3 = gf_mul31(iw2, iw1);
                Pair w4v = gf_mul31(w2, w2);
                Pair iw4v= gf_mul31(iw2, iw2);
                w5[8*j+0]=w1.x; w5[8*j+1]=w1.y; w5[8*j+2]=w2.x; w5[8*j+3]=w2.y;
                w5[8*j+4]=w3.x; w5[8*j+5]=w3.y; w5[8*j+6]=w4v.x; w5[8*j+7]=w4v.y;
                iw5[8*j+0]=iw1.x; iw5[8*j+1]=iw1.y; iw5[8*j+2]=iw2.x; iw5[8*j+3]=iw2.y;
                iw5[8*j+4]=iw3.x; iw5[8*j+5]=iw3.y; iw5[8*j+6]=iw4v.x; iw5[8*j+7]=iw4v.y;
            }
        }
        uint32_t n5 = n;
        if (n % 5 == 0) n5 = n / 5;
        w4.resize(6 * n5);
        iw4.resize(6 * n5);
        for (size_t m2 = n5 / 2; m2 >= 1; m2 /= 2) {
            Pair r = gf_pow31(h, order / (2 * m2));
            Pair ir = gf_inv31(r);
            for (size_t j = 0; j < m2; ++j) {
                Pair r1  = gf_pow31(r, j);
                Pair ir1 = gf_pow31(ir, j);
                Pair r2  = gf_mul31(r1, r1);
                Pair ir2 = gf_mul31(ir1, ir1);
                Pair r3  = gf_mul31(r2, r1);
                Pair ir3 = gf_mul31(ir2, ir1);
                size_t idx = 6 * (m2 + j);
                w4 [idx+0]=r1.x;  w4 [idx+1]=r1.y;  w4 [idx+2]=r2.x;  w4 [idx+3]=r2.y;  w4 [idx+4]=r3.x;  w4 [idx+5]=r3.y;
                iw4[idx+0]=ir1.x; iw4[idx+1]=ir1.y; iw4[idx+2]=ir2.x; iw4[idx+3]=ir2.y; iw4[idx+4]=ir3.x; iw4[idx+5]=ir3.y;
            }
            if (m2 == 1) break;
        }
        return;
    }

    if (mode == 61) {
        const Pair h{481139922016222ULL, 814659809902011ULL};
        const uint64_t order = 1ULL << 62;
        Pair root = gf_pow61(h, order / n);
        Pair invroot = gf_inv61(root);
        uint32_t m = n / 5;
        if (m == 0) m = 1;
        w5.resize(8 * m);
        iw5.resize(8 * m);
        if (n % 5 == 0) {
            for (uint32_t j = 0; j < m; ++j) {
                Pair w1  = gf_pow61(root, j);
                Pair iw1 = gf_pow61(invroot, j);
                Pair w2  = gf_mul61(w1, w1);
                Pair iw2 = gf_mul61(iw1, iw1);
                Pair w3  = gf_mul61(w2, w1);
                Pair iw3 = gf_mul61(iw2, iw1);
                Pair w4v = gf_mul61(w2, w2);
                Pair iw4v= gf_mul61(iw2, iw2);
                w5[8*j+0]=w1.x; w5[8*j+1]=w1.y; w5[8*j+2]=w2.x; w5[8*j+3]=w2.y;
                w5[8*j+4]=w3.x; w5[8*j+5]=w3.y; w5[8*j+6]=w4v.x; w5[8*j+7]=w4v.y;
                iw5[8*j+0]=iw1.x; iw5[8*j+1]=iw1.y; iw5[8*j+2]=iw2.x; iw5[8*j+3]=iw2.y;
                iw5[8*j+4]=iw3.x; iw5[8*j+5]=iw3.y; iw5[8*j+6]=iw4v.x; iw5[8*j+7]=iw4v.y;
            }
        }
        uint32_t n5 = n;
        if (n % 5 == 0) n5 = n / 5;
        w4.resize(6 * n5);
        iw4.resize(6 * n5);
        for (size_t m2 = n5 / 2; m2 >= 1; m2 /= 2) {
            Pair r = gf_pow61(h, order / (2 * m2));
            Pair ir = gf_inv61(r);
            for (size_t j = 0; j < m2; ++j) {
                Pair r1  = gf_pow61(r, j);
                Pair ir1 = gf_pow61(ir, j);
                Pair r2  = gf_mul61(r1, r1);
                Pair ir2 = gf_mul61(ir1, ir1);
                Pair r3  = gf_mul61(r2, r1);
                Pair ir3 = gf_mul61(ir2, ir1);
                size_t idx = 6 * (m2 + j);
                w4 [idx+0]=r1.x;  w4 [idx+1]=r1.y;  w4 [idx+2]=r2.x;  w4 [idx+3]=r2.y;  w4 [idx+4]=r3.x;  w4 [idx+5]=r3.y;
                iw4[idx+0]=ir1.x; iw4[idx+1]=ir1.y; iw4[idx+2]=ir2.x; iw4[idx+3]=ir2.y; iw4[idx+4]=ir3.x; iw4[idx+5]=ir3.y;
            }
            if (m2 == 1) break;
        }
        return;
    }
}

static void precalc_for_p(uint64_t p,
                          std::vector<uint64_t>& digitWeight,
                          std::vector<uint64_t>& digitInvWeight,
                          std::vector<int>&      digitWidth,
                          std::vector<uint64_t>& twiddles,
                          std::vector<uint64_t>& invTwiddles,
                          uint64_t& digitWidthValue1,
                          uint64_t& digitWidthValue2,
                          std::vector<bool>& digitWidthMask)
{
    uint32_t n = transformsize(p);
    if (n < 4) n = 4;
    //auto mode = getModMode();
    auto mode = 0;
    if (mode == 0) {
        __uint128_t bigPminus1 = (__uint128_t)MOD_P - 1ULL;
        __uint128_t tmp = bigPminus1 / 192ULL;
        tmp /= n;
        uint64_t exponent = (uint64_t)(tmp * 5ULL);
        uint64_t nr2 = powModP(7ULL, exponent);
        uint64_t inv_n = invModP(n);
        digitWeight[0]    = 1ULL;
        digitInvWeight[0] = inv_n;
        uint64_t prev = 0;
        for (uint64_t j = 1; j <= n; ++j) {
            uint64_t qj = uint64_t(p) * j;
            uint64_t ceil_qj_n = (qj == 0) ? 0 : uint64_t((qj - 1) / n + 1);
            digitWidth[j - 1]  = int(ceil_qj_n - prev);
            prev               = ceil_qj_n;
            if (j < n) {
                uint64_t r = uint64_t(qj % n);
                uint64_t nr2r = r ? powModP(nr2, (uint64_t)(n - r)) : 1ULL;
                digitWeight[j]    = nr2r;
                digitInvWeight[j] = mulModP(invModP(nr2r), inv_n);
            }
        }
    } else if (mode == 31) {
        const uint64_t p31 = (1ULL << 31) - 1;
        __uint128_t bigPminus1 = (__uint128_t)p31 - 1ULL + 2;
        __uint128_t tmp = bigPminus1 / 192ULL;
        tmp /= n;
        uint64_t exponent = (uint64_t)(tmp * 5ULL);
        Pair nr2p = { pow_m31(7ULL, exponent), 0 };
        uint64_t invn = inv_m31(n % p31);
        digitWeight[0] = 1ULL;
        digitWeight[1] = 0ULL;
        digitInvWeight[0] = invn;
        digitInvWeight[1] = 0ULL;
        uint64_t prev = 0;
        for (uint64_t j = 1; j <= n; ++j) {
            uint64_t qj = uint64_t(p) * j;
            uint64_t ceil_qj_n = (qj == 0) ? 0 : uint64_t((qj - 1) / n + 1);
            digitWidth[j - 1]  = int(ceil_qj_n - prev);
            prev               = ceil_qj_n;
            if (j < n) {
                uint64_t r = uint64_t(qj % n);
                Pair nr2r = r ? gf_pow31(nr2p, (uint64_t)(n - r)) : Pair{1,0};
                digitWeight[2*j+0] = nr2r.x;
                digitWeight[2*j+1] = nr2r.y;
                Pair inv_nr2r = gf_inv31(nr2r);
                Pair invw = { mul_m31(inv_nr2r.x, invn), mul_m31(inv_nr2r.y, invn) };
                digitInvWeight[2*j+0] = invw.x;
                digitInvWeight[2*j+1] = invw.y;
            }
        }
    } else {
        const uint64_t p61 = (1ULL << 61) - 1;
        __uint128_t bigPminus1 = (__uint128_t)p61 - 1ULL + 2;
        __uint128_t tmp = bigPminus1 / 192ULL;
        tmp /= n;
        uint64_t exponent = (uint64_t)(tmp * 5ULL);
        Pair nr2p = { pow_m61(7ULL, exponent), 0 };
        uint64_t invn = inv_m61(n % p61);
        digitWeight[0] = 1ULL;
        digitWeight[1] = 0ULL;
        digitInvWeight[0] = invn;
        digitInvWeight[1] = 0ULL;
        uint64_t prev = 0;
        for (uint64_t j = 1; j <= n; ++j) {
            uint64_t qj = uint64_t(p) * j;
            uint64_t ceil_qj_n = (qj == 0) ? 0 : uint64_t((qj - 1) / n + 1);
            digitWidth[j - 1]  = int(ceil_qj_n - prev);
            prev               = ceil_qj_n;
            if (j < n) {
                uint64_t r = uint64_t(qj % n);
                Pair nr2r = r ? gf_pow61(nr2p, (uint64_t)(n - r)) : Pair{1,0};
                digitWeight[2*j+0] = nr2r.x;
                digitWeight[2*j+1] = nr2r.y;
                Pair inv_nr2r = gf_inv61(nr2r);
                Pair invw = { mul_m61(inv_nr2r.x, invn), mul_m61(inv_nr2r.y, invn) };
                digitInvWeight[2*j+0] = invw.x;
                digitInvWeight[2*j+1] = invw.y;
            }
        }
    }

    uint64_t w1 = (uint64_t)digitWidth[0];
    uint64_t w2 = 0;
    for (int w : digitWidth) {
        if ((uint64_t)w != w1) { w2 = (uint64_t)w; break; }
    }
    digitWidthValue1 = w1;
    digitWidthValue2 = w2;
    digitWidthMask.resize(n);
    for (size_t i = 0; i < n; ++i) digitWidthMask[i] = ((uint64_t)digitWidth[i] == w2);
    if (n % 5 == 0) { (void)0; }
    twiddles.assign(twiddles.size(), 0ULL);
    invTwiddles.assign(invTwiddles.size(), 0ULL);
}

Precompute::Precompute(uint64_t exponent)
  : n_{ transformsize(exponent) }
{
    if (n_ < 4) n_ = 4;
    //auto mode = getModMode();
    auto mode = 0;
    if (mode == 0) {
        digitWeight_.resize(n_);
        digitInvWeight_.resize(n_);
        twiddles_.resize(3 * n_);
        invTwiddles_.resize(3 * n_);
    } else {
        digitWeight_.resize(2 * n_);
        digitInvWeight_.resize(2 * n_);
        twiddles_.resize(6 * n_);
        invTwiddles_.resize(6 * n_);
    }
    digitWidth_.resize(n_);
    digitWidthMask_.resize(n_);
    precalc_for_p(exponent,
                  digitWeight_,
                  digitInvWeight_,
                  digitWidth_,
                  twiddles_,
                  invTwiddles_,
                  digitWidthValue1_,
                  digitWidthValue2_,
                  digitWidthMask_);
    prepare_radix_twiddles(n_, w4_, iw4_, w5_, iw5_);
}

uint32_t Precompute::getN() const { return n_; }
const std::vector<uint64_t>& Precompute::digitWeight() const { return digitWeight_; }
const std::vector<uint64_t>& Precompute::digitInvWeight() const { return digitInvWeight_; }
const std::vector<int>&      Precompute::getDigitWidth() const { return digitWidth_; }
const std::vector<bool>&     Precompute::getDigitWidthMask() const { return digitWidthMask_; }
const std::vector<uint64_t>& Precompute::twiddles() const { return twiddles_; }
const std::vector<uint64_t>& Precompute::invTwiddles() const { return invTwiddles_; }
const std::vector<uint64_t>& Precompute::twiddlesRadix4() const { return w4_; }
const std::vector<uint64_t>& Precompute::invTwiddlesRadix4() const { return iw4_; }
const std::vector<uint64_t>& Precompute::twiddlesRadix5() const { return w5_; }
const std::vector<uint64_t>& Precompute::invTwiddlesRadix5() const { return iw5_; }
uint64_t Precompute::getDigitWidthValue1() const { return digitWidthValue1_; }
uint64_t Precompute::getDigitWidthValue2() const { return digitWidthValue2_; }

} // namespace math
