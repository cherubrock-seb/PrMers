/*
 * Mersenne OpenCL Primality Test Kernel
 *
 * This kernel implements a Lucas–Lehmer-based Mersenne prime test using integer arithmetic,
 * an NTT, and an IDBWT on the GPU via OpenCL.
 *
 * The code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) from Nick Craig‐Wood's
 *     IOCCC 2012 entry.
 *   - The Armprime project (https://www.craig-wood.com/nick/armprime/ and https://github.com/ncw/).
 *   - Yves Gallot (https://github.com/galloty) and his work on Genefer (https://github.com/galloty/genefer22)
 *     for his insights into NTT and IDBWT.
 *
 * Author: Cherubrock
 *
 * This code is released as free software.
 */
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef uint  uint32_t;
typedef ulong uint64_t;

#define MOD_P 0xffffffff00000001UL  // p = 2^64 - 2^32 + 1
#define MOD_P_COMP 0xffffffffU       // 2^64 - p = 2^32 - 1

inline ulong modAdd(const ulong lhs, const ulong rhs)
{
    const uint c = (lhs >= MOD_P - rhs) ? MOD_P_COMP : 0;
    return lhs + rhs + c;
}

inline ulong modSub(const ulong lhs, const ulong rhs)
{
    const uint c = (lhs < rhs) ? MOD_P_COMP : 0;
    return lhs - rhs - c;
}

inline ulong Reduce(const ulong lo, const ulong hi)
{
    // hi_hi * 2^96 + hi_lo * 2^64 + lo = lo + hi_lo * 2^32 - (hi_hi + hi_lo)
    const uint c = (lo >= MOD_P) ? MOD_P_COMP : 0;
    ulong r = lo + c;
    r = modAdd(r, hi << 32);
    r = modSub(r, (hi >> 32) + (uint)hi);
    return r;
}

inline ulong2 modAdd2(ulong2 lhs, ulong2 rhs) {
    ulong2 c = select((ulong2)0, (ulong2)MOD_P_COMP, lhs >= ((ulong2)MOD_P - rhs));
    return lhs + rhs + c;
}

inline ulong2 modSub2(ulong2 lhs, ulong2 rhs) {
    ulong2 c = select((ulong2)0, (ulong2)MOD_P_COMP, lhs < rhs);
    return lhs - rhs - c;
}

inline ulong2 Reduce2(ulong2 lo, ulong2 hi) {
    const ulong2 MOD_P_COMP2 = (ulong2)(MOD_P_COMP, MOD_P_COMP);
    const ulong2 MOD_P2      = (ulong2)(MOD_P, MOD_P);
    ulong2 c = select((ulong2)0, MOD_P_COMP2, lo >= MOD_P2);
    ulong2 r = lo + c;
    ulong2 hi_shifted = hi << 32;
    ulong2 hi_high = hi >> 32;
    ulong2 hi_low = convert_ulong2(convert_uint2(hi));
    ulong2 hi_reduced = hi_high + hi_low;
    r = modAdd2(r, hi_shifted);
    r = modSub2(r, hi_reduced);
    return r;
}

inline ulong2 modMul2(ulong2 lhs, ulong2 rhs) {
    ulong2 lo = lhs * rhs;
    ulong2 hi = mul_hi(lhs, rhs);
    return Reduce2(lo, hi);
}

inline ulong modMul(const ulong lhs, const ulong rhs)
{
    const ulong lo = lhs * rhs, hi = mul_hi(lhs, rhs);
    return Reduce(lo, hi);
}

inline ulong modMuli(ulong x) {
    return modMul(x, (1UL << 48));
}

inline ulong2 digit_adc2(ulong2 lhs, ulong2 digit_width, __private ulong *carry) {
    ulong2 res;
    uint64_t c = *carry;
    #pragma unroll 2
    for (int i = 0; i < 2; i++) {
        uint64_t s = lhs[i] + c;
        res[i] = s & ((1UL << digit_width[i]) - 1UL);
        c = s >> digit_width[i];
    }
    *carry = c;
    return res;
}

inline ulong2 digit_adc2_last(ulong2 lhs, ulong2 digit_width, __private ulong *carry) {
    ulong2 res;
    uint64_t c = *carry;
    uint64_t s = lhs[0] + c;
    res[0] = s & ((1UL << digit_width[0]) - 1UL);
    c = s >> digit_width[0];
    *carry = c;
    res[1] = lhs[1] + c;
    return res;
}

#ifndef LOCAL_PROPAGATION_DEPTH
#define LOCAL_PROPAGATION_DEPTH 8
#endif
#ifndef LOCAL_PROPAGATION_DEPTH_DIV2
#define LOCAL_PROPAGATION_DEPTH_DIV2 2
#endif
#ifndef LOCAL_PROPAGATION_DEPTH_DIV2_MIN
#define LOCAL_PROPAGATION_DEPTH_DIV2_MIN 2
#endif
#ifndef CARRY_WORKER
#define CARRY_WORKER 1
#endif
#define PRAGMA_UNROLL_HELPER(x) _Pragma(#x)
#define PRAGMA_UNROLL(n) PRAGMA_UNROLL_HELPER(unroll n)

__kernel void kernel_sub2(__global ulong* restrict x,
                          __global ulong* restrict digit_width,
                          const ulong n)
{
    if (get_global_id(0) == 0) {
        uint c = 2U;
        while(c != 0U) {
            #pragma unroll
            for(uint i = 0; i < n; i++){
                const int d = digit_width[i];
                ulong val = x[i];
                const ulong b = 1UL << d;
                if (val >= c) {
                    x[i] = modSub(val, c);
                    c = 0U;
                    break;
                } else {
                    const ulong temp = modSub(val, c);
                    x[i] = modAdd(temp, b);
                    c = 1U;
                }
            }
        }
    }
}

__kernel void kernel_carry(__global ulong* restrict x,
                           __global ulong* restrict carry_array,
                           __global ulong* restrict digit_width)
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end = start + LOCAL_PROPAGATION_DEPTH; 
    ulong carry = 0UL; 
    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV2)
    for (ulong i = start; i < end; i += 2) {
        ulong2 x_vec = vload2(0, x + i);
        ulong2 digit_width_vec = vload2(0, digit_width + i);
        x_vec = digit_adc2(x_vec, digit_width_vec, &carry); 
        vstore2(x_vec, 0, x + i);
    }
    
    if (carry != 0) {
        carry_array[gid] = carry;
    }
}

__kernel void kernel_carry_2(__global ulong* restrict x,
                             __global ulong* restrict carry_array,
                             __global ulong* restrict digit_width)
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end = start + LOCAL_PROPAGATION_DEPTH - 2;  

    const ulong prev_gid = (gid == 0) ? (CARRY_WORKER - 1) : (gid - 1);
    ulong carry = carry_array[prev_gid];

    if (carry == 0) return;

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV2_MIN)
    for (ulong i = start; i < end; i += 2) {
        ulong2 x_vec = vload2(0, x + i);
        ulong2 digit_width_vec = vload2(0, digit_width + i);
        x_vec = digit_adc2(x_vec, digit_width_vec, &carry); 
        vstore2(x_vec, 0, x + i);
        if (carry == 0) return;
    }
    if (carry != 0) {
        ulong2 x_vec = vload2(0, x + end);
        ulong2 digit_width_vec = vload2(0, digit_width + end);
        x_vec = digit_adc2_last(x_vec, digit_width_vec, &carry); 
        vstore2(x_vec, 0, x + end);
    }
}

__kernel void kernel_inverse_ntt_radix4_mm(__global ulong* restrict x,
                                             __constant ulong* restrict wi,
                                             const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    ulong2 coeff0 = (ulong2)( x[base + 0*m], x[base + 1*m] );
    ulong2 coeff1 = (ulong2)( x[base + 2*m], x[base + 3*m] );
    ulong2 twiddles0 = (ulong2)(1UL, wi[twiddle_offset + 1]);
    ulong2 twiddles1 = (ulong2)(wi[twiddle_offset + 0], wi[twiddle_offset + 2]);
    ulong2 u0 = modMul2(coeff0, twiddles0);
    ulong2 u1 = modMul2(coeff1, twiddles1);
    ulong v0 = modAdd(u0.x, u0.y);
    ulong v1 = modSub(u0.x, u0.y);
    ulong v2 = modAdd(u1.x, u1.y);
    ulong v3 = modMuli(modSub(u1.y, u1.x));
    x[base + 0*m] = modAdd(v0, v2);
    x[base + 1*m] = modAdd(v1, v3);
    x[base + 2*m] = modSub(v0, v2);
    x[base + 3*m] = modSub(v1, v3);
}

__kernel void kernel_ntt_radix4_last_m1_n4(__global ulong* restrict x,
                                            __global ulong* restrict w,
                                            __global ulong* restrict digit_weight) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    ulong2 coeff0 = (ulong2)( x[4*k + 0], x[4*k + 1] );
    ulong2 coeff1 = (ulong2)( x[4*k + 2], x[4*k + 3] );
    ulong2 dw0 = (ulong2)( digit_weight[4*k + 0], digit_weight[4*k + 1] );
    ulong2 dw1 = (ulong2)( digit_weight[4*k + 2], digit_weight[4*k + 3] );
    const ulong w2 = w[twiddle_offset];
    const ulong w1 = w[twiddle_offset + 1];
    const ulong w12 = w[twiddle_offset + 2];
    ulong2 u0 = modMul2(coeff0, dw0);
    ulong2 u1 = modMul2(coeff1, dw1);
    ulong v0 = modAdd(u0.x, u1.x);
    ulong v1 = modAdd(u0.y, u1.y);
    ulong v2 = modSub(u0.x, u1.x);
    ulong v3 = modMuli(modSub(u0.y, u1.y));
    ulong r0 = modAdd(v0, v1);
    ulong r1 = modMul(modSub(v0, v1), w1);
    ulong r2 = modMul(modAdd(v2, v3), w2);
    ulong r3 = modMul(modSub(v2, v3), w12);
    r0 = modMul(r0, r0);
    r1 = modMul(r1, r1);
    r2 = modMul(r2, r2);
    r3 = modMul(r3, r3);
    x[4*k + 0] = r0;
    x[4*k + 1] = r1;
    x[4*k + 2] = r2;
    x[4*k + 3] = r3;
}

__kernel void kernel_inverse_ntt_radix4_mm_last(__global ulong* restrict x,
                                                 __constant ulong* restrict wi,
                                                 __global ulong* restrict digit_invweight,
                                                 const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    ulong2 coeff0 = (ulong2)( x[base + 0*m], x[base + 1*m] );
    ulong2 coeff1 = (ulong2)( x[base + 2*m], x[base + 3*m] );
    const ulong iw2 = wi[twiddle_offset + 0];
    const ulong iw1 = wi[twiddle_offset + 1];
    const ulong iw12 = wi[twiddle_offset + 2];
    ulong2 twiddles0 = (ulong2)(1UL, iw1);
    ulong2 twiddles1 = (ulong2)(iw2, iw12);
    ulong2 u0 = modMul2(coeff0, twiddles0);
    ulong2 u1 = modMul2(coeff1, twiddles1);
    ulong v0 = modAdd(u0.x, u0.y);
    ulong v1 = modSub(u0.x, u0.y);
    ulong v2 = modAdd(u1.x, u1.y);
    ulong v3 = modMuli(modSub(u1.y, u1.x));
    ulong2 invWeight0 = (ulong2)( digit_invweight[base + 0*m], digit_invweight[base + 1*m] );
    ulong2 invWeight1 = (ulong2)( digit_invweight[base + 2*m], digit_invweight[base + 3*m] );
    ulong t0 = modAdd(v0, v2);
    ulong t1 = modAdd(v1, v3);
    ulong t2 = modSub(v0, v2);
    ulong t3 = modSub(v1, v3);
    ulong r0 = modMul(t0, invWeight0.x);
    ulong r1 = modMul(t1, invWeight0.y);
    ulong r2 = modMul(t2, invWeight1.x);
    ulong r3 = modMul(t3, invWeight1.y);
    x[base + 0*m] = r0;
    x[base + 1*m] = r1;
    x[base + 2*m] = r2;
    x[base + 3*m] = r3;
}

__kernel void kernel_ntt_radix4_last_m1(__global ulong* restrict x,
                                        __global ulong* restrict w) {
    const ulong k = get_global_id(0);
    ulong2 coeff0 = (ulong2)( x[4*k + 0], x[4*k + 1] );
    ulong2 coeff1 = (ulong2)( x[4*k + 2], x[4*k + 3] );
    const ulong twiddle_offset = 6;
    ulong v0 = modAdd(coeff0.x, coeff1.x);
    ulong v1 = modAdd(coeff0.y, coeff1.y);
    ulong v2 = modSub(coeff0.x, coeff1.x);
    ulong v3 = modMuli(modSub(coeff0.y, coeff1.y));
    ulong t0 = modAdd(v0, v1);
    ulong t1 = modSub(v0, v1);
    ulong t2 = modAdd(v2, v3);
    ulong t3 = modSub(v2, v3);
    const ulong f0 = 1UL;
    const ulong f1 = w[twiddle_offset + 1];
    const ulong f2 = w[twiddle_offset + 0];
    const ulong f3 = w[twiddle_offset + 2];
    ulong r0 = modMul(t0, f0);
    ulong r1 = modMul(t1, f1);
    ulong r2 = modMul(t2, f2);
    ulong r3 = modMul(t3, f3);
    r0 = modMul(r0, r0);
    r1 = modMul(r1, r1);
    r2 = modMul(r2, r2);
    r3 = modMul(r3, r3);
    x[4*k + 0] = r0;
    x[4*k + 1] = r1;
    x[4*k + 2] = r2;
    x[4*k + 3] = r3;
}

__kernel void kernel_ntt_radix4_mm_first(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          __global ulong* restrict digit_weight,
                                          const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    const ulong w2 = w[twiddle_offset];
    const ulong w1 = w[twiddle_offset + 1];
    const ulong w12 = w[twiddle_offset + 2];
    ulong2 coeff0 = (ulong2)( x[i + 0*m], x[i + 1*m] );
    ulong2 coeff1 = (ulong2)( x[i + 2*m], x[i + 3*m] );
    ulong2 weight0 = (ulong2)( digit_weight[i + 0*m], digit_weight[i + 1*m] );
    ulong2 weight1 = (ulong2)( digit_weight[i + 2*m], digit_weight[i + 3*m] );
    ulong2 u0 = modMul2(coeff0, weight0);
    ulong2 u1 = modMul2(coeff1, weight1);
    ulong v0 = modAdd(u0.x, u1.x);
    ulong v1 = modAdd(u0.y, u1.y);
    ulong v2 = modSub(u0.x, u1.x);
    ulong v3 = modMuli(modSub(u0.y, u1.y));
    ulong t0 = modAdd(v0, v1);
    ulong t1 = modSub(v0, v1);
    ulong t2 = modAdd(v2, v3);
    ulong t3 = modSub(v2, v3);
    ulong r0 = modMul(t0, 1UL);
    ulong r1 = modMul(t1, w1);
    ulong r2 = modMul(t2, w2);
    ulong r3 = modMul(t3, w12);
    x[i + 0*m] = r0;
    x[i + 1*m] = r1;
    x[i + 2*m] = r2;
    x[i + 3*m] = r3;
}

__kernel void kernel_ntt_radix4_mm(__global ulong* restrict x,
                                    __global ulong* restrict w,
                                    const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    const ulong w2 = w[twiddle_offset];
    const ulong w1 = w[twiddle_offset + 1];
    const ulong w12 = w[twiddle_offset + 2];
    ulong2 coeff0 = (ulong2)( x[i + 0*m], x[i + 1*m] );
    ulong2 coeff1 = (ulong2)( x[i + 2*m], x[i + 3*m] );
    ulong v0 = modAdd(coeff0.x, coeff1.x);
    ulong v1 = modAdd(coeff0.y, coeff1.y);
    ulong v2 = modSub(coeff0.x, coeff1.x);
    ulong v3 = modMuli(modSub(coeff0.y, coeff1.y));
    ulong t0 = modAdd(v0, v1);
    ulong t1 = modSub(v0, v1);
    ulong t2 = modAdd(v2, v3);
    ulong t3 = modSub(v2, v3);
    ulong r0 = modMul(t0, 1UL);
    ulong r1 = modMul(t1, w1);
    ulong r2 = modMul(t2, w2);
    ulong r3 = modMul(t3, w12);
    x[i + 0*m] = r0;
    x[i + 1*m] = r1;
    x[i + 2*m] = r2;
    x[i + 3*m] = r3;
}

__kernel void kernel_inverse_ntt_radix4_m1(__global ulong* restrict x,
                                            __global ulong* restrict wi) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    ulong2 coeff0 = (ulong2)( x[4*k + 0], x[4*k + 1] );
    ulong2 coeff1 = (ulong2)( x[4*k + 2], x[4*k + 3] );
    const ulong iw2 = wi[twiddle_offset];
    const ulong iw1 = wi[twiddle_offset + 1];
    const ulong iw12 = wi[twiddle_offset + 2];
    ulong2 twiddles0 = (ulong2)(1UL, iw1);
    ulong2 twiddles1 = (ulong2)(iw2, iw12);
    ulong2 u0 = modMul2(coeff0, twiddles0);
    ulong2 u1 = modMul2(coeff1, twiddles1);
    ulong v0 = modAdd(u0.x, u0.y);
    ulong v1 = modSub(u0.x, u0.y);
    ulong v2 = modAdd(u1.x, u1.y);
    ulong v3 = modMuli(modSub(u1.y, u1.x));
    ulong r0 = modAdd(v0, v2);
    ulong r1 = modAdd(v1, v3);
    ulong r2 = modSub(v0, v2);
    ulong r3 = modSub(v1, v3);
    x[4*k + 0] = r0;
    x[4*k + 1] = r1;
    x[4*k + 2] = r2;
    x[4*k + 3] = r3;
}

__kernel void kernel_inverse_ntt_radix4_m1_n4(__global ulong* restrict x,
                                               __constant ulong* restrict wi,
                                               __global ulong* restrict digit_invweight) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    ulong2 coeff0 = (ulong2)( x[4*k + 0], x[4*k + 1] );
    ulong2 coeff1 = (ulong2)( x[4*k + 2], x[4*k + 3] );
    ulong2 div0 = (ulong2)( digit_invweight[4*k + 0], digit_invweight[4*k + 1] );
    ulong2 div1 = (ulong2)( digit_invweight[4*k + 2], digit_invweight[4*k + 3] );
    const ulong iw2 = wi[twiddle_offset];
    const ulong iw1 = wi[twiddle_offset + 1];
    const ulong iw12 = wi[twiddle_offset + 2];
    ulong2 twiddles0 = (ulong2)(1UL, iw1);
    ulong2 twiddles1 = (ulong2)(iw2, iw12);
    ulong2 u0 = modMul2(coeff0, twiddles0);
    ulong2 u1 = modMul2(coeff1, twiddles1);
    ulong v0 = modAdd(u0.x, u0.y);
    ulong v1 = modSub(u0.x, u0.y);
    ulong v2 = modAdd(u1.x, u1.y);
    ulong v3 = modMuli(modSub(u1.y, u1.x));
    ulong t0 = modAdd(v0, v2);
    ulong t1 = modAdd(v1, v3);
    ulong t2 = modSub(v0, v2);
    ulong t3 = modSub(v1, v3);
    ulong r0 = modMul(t0, div0.x);
    ulong r1 = modMul(t1, div0.y);
    ulong r2 = modMul(t2, div1.x);
    ulong r3 = modMul(t3, div1.y);
    x[4*k + 0] = r0;
    x[4*k + 1] = r1;
    x[4*k + 2] = r2;
    x[4*k + 3] = r3;
}
