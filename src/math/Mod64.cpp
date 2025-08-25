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
#include "math/Mod64.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace math {

static ModMode g_mode = ModMode::Solinas;

static inline uint64_t red31_u128(uint64_t lo, uint64_t hi) {
    const uint64_t P = 0x7fffffffULL;
    uint64_t x0 = lo & P;
    uint64_t x1 = (lo >> 31) | (hi << 33);
    uint64_t x2 = hi >> 31;
    uint64_t r = x0 + x1 + x2;
    r = (r & P) + (r >> 31);
    return r >= P ? r - P : r;
}

static inline uint64_t red61_u128(uint64_t lo, uint64_t hi) {
    const uint64_t P = 0x1fffffffffffffffULL;
    uint64_t x0 = lo & P;
    uint64_t x1 = (lo >> 61) | (hi << 3);
    uint64_t x2 = hi >> 58;
    uint64_t r = x0 + x1 + x2;
    r = (r & P) + (r >> 61);
    return r >= P ? r - P : r;
}

void Mod64::setMode(ModMode m) { g_mode = m; }
ModMode Mod64::mode() { return g_mode; }

uint64_t Mod64::mulModP(uint64_t a, uint64_t b) {
    uint64_t hi, lo;
#ifdef _MSC_VER
    lo = _umul128(a, b, &hi);
#else
    __uint128_t p = ( (__uint128_t)a * (__uint128_t)b );
    hi = (uint64_t)(p >> 64);
    lo = (uint64_t)p;
#endif
    uint32_t A = (uint32_t)(hi >> 32);
    uint32_t B = (uint32_t)(hi & 0xffffffffULL);
    uint64_t r = lo;
    uint64_t old = r;
    r += ((uint64_t)B << 32);
    if (r < old) r += ((1ULL << 32) - 1ULL);
    uint64_t sub = (uint64_t)A + (uint64_t)B;
    r = (r >= sub ? r - sub : r + MOD_P - sub);
    return (r >= MOD_P ? r - MOD_P : r);
}

uint64_t Mod64::powModP(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    while (exp) {
        if (exp & 1) r = mulModP(r, base);
        base = mulModP(base, base);
        exp >>= 1;
    }
    return r;
}

uint64_t Mod64::invModP(uint64_t x) { return powModP(x, MOD_P - 2); }

uint64_t Mod64::add31(uint64_t a, uint64_t b) {
    const uint64_t P = 0x7fffffffULL;
    uint64_t s = a + b;
    s = (s & P) + (s >> 31);
    return s >= P ? s - P : s;
}

uint64_t Mod64::sub31(uint64_t a, uint64_t b) {
    const uint64_t P = 0x7fffffffULL;
    return a >= b ? a - b : a + P - b;
}

uint64_t Mod64::mul31(uint64_t a, uint64_t b) {
#ifdef _MSC_VER
    uint64_t hi, lo = _umul128(a, b, &hi);
#else
    __uint128_t p = ( (__uint128_t)a * (__uint128_t)b );
    uint64_t lo = (uint64_t)p;
    uint64_t hi = (uint64_t)(p >> 64);
#endif
    return red31_u128(lo, hi);
}

uint64_t Mod64::pow31(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    while (exp) {
        if (exp & 1) r = mul31(r, base);
        base = mul31(base, base);
        exp >>= 1;
    }
    return r;
}

uint64_t Mod64::inv31(uint64_t x) {
    const uint64_t P = 0x7fffffffULL;
    return pow31(x, P - 2);
}

gf31_2 Mod64::add31_2(gf31_2 x, gf31_2 y) {
    return { add31(x.a,y.a), add31(x.b,y.b) };
}

gf31_2 Mod64::sub31_2(gf31_2 x, gf31_2 y) {
    return { sub31(x.a,y.a), sub31(x.b,y.b) };
}

gf31_2 Mod64::mul31_2(gf31_2 x, gf31_2 y) {
    uint64_t t0 = mul31(x.a,y.a);
    uint64_t t1 = mul31(x.b,y.b);
    uint64_t r0 = sub31(t0,t1);
    uint64_t r1 = add31(mul31(x.b,y.a), mul31(x.a,y.b));
    return { r0, r1 };
}

gf31_2 Mod64::pow31_2(gf31_2 x, uint64_t e) {
    gf31_2 r{1,0};
    while (e) {
        if (e & 1) r = mul31_2(r,x);
        x = mul31_2(x,x);
        e >>= 1;
    }
    return r;
}

gf31_2 Mod64::inv31_2(gf31_2 x) {
    uint64_t aa = mul31(x.a,x.a);
    uint64_t bb = mul31(x.b,x.b);
    uint64_t d = add31(aa, bb);
    uint64_t id = inv31(d);
    uint64_t ra = mul31(x.a, id);
    uint64_t rb = mul31((x.b ? (0x7fffffffULL - x.b) : 0), id);
    return { ra, rb };
}

uint64_t Mod64::add61(uint64_t a, uint64_t b) {
    const uint64_t P = 0x1fffffffffffffffULL;
    uint64_t s = a + b;
    s = (s & P) + (s >> 61);
    return s >= P ? s - P : s;
}

uint64_t Mod64::sub61(uint64_t a, uint64_t b) {
    const uint64_t P = 0x1fffffffffffffffULL;
    return a >= b ? a - b : a + P - b;
}

uint64_t Mod64::mul61(uint64_t a, uint64_t b) {
#ifdef _MSC_VER
    uint64_t hi, lo = _umul128(a, b, &hi);
#else
    __uint128_t p = ( (__uint128_t)a * (__uint128_t)b );
    uint64_t lo = (uint64_t)p;
    uint64_t hi = (uint64_t)(p >> 64);
#endif
    return red61_u128(lo, hi);
}

uint64_t Mod64::pow61(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    while (exp) {
        if (exp & 1) r = mul61(r, base);
        base = mul61(base, base);
        exp >>= 1;
    }
    return r;
}

uint64_t Mod64::inv61(uint64_t x) {
    const uint64_t P = 0x1fffffffffffffffULL;
    return pow61(x, P - 2);
}

gf61_2 Mod64::add61_2(gf61_2 x, gf61_2 y) {
    return { add61(x.a,y.a), add61(x.b,y.b) };
}

gf61_2 Mod64::sub61_2(gf61_2 x, gf61_2 y) {
    return { sub61(x.a,y.a), sub61(x.b,y.b) };
}

gf61_2 Mod64::mul61_2(gf61_2 x, gf61_2 y) {
    uint64_t t0 = mul61(x.a,y.a);
    uint64_t t1 = mul61(x.b,y.b);
    uint64_t r0 = sub61(t0,t1);
    uint64_t r1 = add61(mul61(x.b,y.a), mul61(x.a,y.b));
    return { r0, r1 };
}

gf61_2 Mod64::pow61_2(gf61_2 x, uint64_t e) {
    gf61_2 r{1,0};
    while (e) {
        if (e & 1) r = mul61_2(r,x);
        x = mul61_2(x,x);
        e >>= 1;
    }
    return r;
}

gf61_2 Mod64::inv61_2(gf61_2 x) {
    uint64_t aa = mul61(x.a,x.a);
    uint64_t bb = mul61(x.b,x.b);
    uint64_t d = add61(aa, bb);
    uint64_t id = inv61(d);
    uint64_t ra = mul61(x.a, id);
    uint64_t rb = mul61((x.b ? (0x1fffffffffffffffULL - x.b) : 0), id);
    return { ra, rb };
}

uint64_t mulModP(uint64_t a, uint64_t b) { return Mod64::mulModP(a,b); }
uint64_t powModP(uint64_t base, uint64_t exp) { return Mod64::powModP(base,exp); }
uint64_t invModP(uint64_t x) { return Mod64::invModP(x); }

}
