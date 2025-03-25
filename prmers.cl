/*
 * Mersenne OpenCL Primality Test Kernel
 *
 * This kernel implements a Lucas–Lehmer-based Mersenne prime test using integer arithmetic,
 * an NTT, and an IDBWT on the GPU via OpenCL.
 *
 * The code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) from Nick Craig-Wood's
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

inline ulong4 modAdd4(ulong4 lhs, ulong4 rhs) {
    ulong4 c = select((ulong4)0, (ulong4)MOD_P_COMP, lhs >= ((ulong4)MOD_P - rhs));
    return lhs + rhs + c;
}

inline ulong4 modSub4(ulong4 lhs, ulong4 rhs) {
    ulong4 c = select((ulong4)0, (ulong4)MOD_P_COMP, lhs < rhs);
    return lhs - rhs - c;
}


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
    r = modAdd(r, hi << 32);             // lhs * rhs < p^2 => hi * 2^32 < p^2 / 2^32 < p.
    r = modSub(r, (hi >> 32) + (uint)hi);
    return r;
}


inline ulong4 Reduce4(ulong4 lo, ulong4 hi) {
    const ulong4 MOD_P_COMP4 = (ulong4)(MOD_P_COMP, MOD_P_COMP, MOD_P_COMP, MOD_P_COMP);
    const ulong4 MOD_P4      = (ulong4)(MOD_P,      MOD_P,      MOD_P,      MOD_P);

    ulong4 c = select((ulong4)0, MOD_P_COMP4, lo >= MOD_P4);
    ulong4 r = lo + c;

    ulong4 hi_shifted = hi << 32;
    ulong4 hi_high = hi >> 32;
    ulong4 hi_low = convert_ulong4(convert_uint4(hi));
    ulong4 hi_reduced = hi_high + hi_low;

    r = modAdd4(r, hi_shifted);
    r = modSub4(r, hi_reduced);

    return r;
}



inline ulong4 modMul4(ulong4 lhs, ulong4 rhs) {
    ulong4 lo = lhs * rhs;
    ulong4 hi = mul_hi(lhs, rhs);
    return Reduce4(lo, hi);
}

// Compute the 128-bit product of a and b as high:low.
inline void mul128(ulong a, ulong b, __private ulong *hi, __private ulong *lo) {
    *lo = a * b;
    *hi = mul_hi(a, b);
}

inline ulong modMul(const ulong lhs, const ulong rhs)
{
    const ulong lo = lhs * rhs, hi = mul_hi(lhs, rhs);
    return Reduce(lo, hi);
}



// modMuli multiplies by sqrt(-1) mod p, where sqrt(-1) is defined as 2^48 mod p.
inline ulong modMuli(ulong x) {
    return modMul(x, (1UL << 48));
}


// Add-with-carry for a digit of specified width.
inline ulong digit_adc(ulong lhs, int digit_width, __private ulong *carry) {
    // Compute s = lhs + carry and detect overflow.
    ulong s = lhs + (*carry);
    ulong c = (s < lhs) ? 1UL : 0UL;
    // Update carry: new carry = (s >> digit_width) + (c << (64 - digit_width))
    *carry = (s >> digit_width) + (c << (64 - digit_width));
    // Extract the lower digit_width bits.
    ulong res = s & ((1UL << digit_width) - 1UL);
    return res;
}

inline ulong4 digit_adc4(ulong4 lhs, ulong4 digit_width, __private ulong *carry) {
    ulong4 res;
    uint64_t c = *carry;
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        uint64_t s = lhs[i] + c;                        
        res[i] = s & ((1UL << digit_width[i]) - 1UL);    
        c = s >> digit_width[i];
    }
    *carry = c;
    return res;
}

inline ulong4 digit_adc4_last(ulong4 lhs, ulong4 digit_width, __private ulong *carry) {
    ulong4 res;
    uint64_t c = *carry;
    #pragma unroll 3
    for (int i = 0; i < 3; i++) {
        uint64_t s = lhs[i] + c;                        
        res[i] = s & ((1UL << digit_width[i]) - 1UL);    
        c = s >> digit_width[i];
    }
    *carry = c;
    res[3] = lhs[3] + c;
    return res;
}

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
                // Calculate b only once
                const ulong b = 1UL << d;
                if (val >= c) {
                    x[i] = modSub(val, c);
                    c = 0U;
                    break;
                } else {
                    // Reuse the result of Sub to avoid redundant computation
                    const ulong temp = modSub(val, c);
                    x[i] = modAdd(temp, b);
                    c = 1U;
                }
            }
        }
    }
}


#ifndef LOCAL_PROPAGATION_DEPTH
#define LOCAL_PROPAGATION_DEPTH 8
#endif
#ifndef LOCAL_PROPAGATION_DEPTH_DIV4
#define LOCAL_PROPAGATION_DEPTH_DIV4 2
#endif
#ifndef LOCAL_PROPAGATION_DEPTH_DIV4_MIN
#define LOCAL_PROPAGATION_DEPTH_DIV4_MIN 2
#endif
#ifndef CARRY_WORKER
#define CARRY_WORKER 1
#endif
#define PRAGMA_UNROLL_HELPER(x) _Pragma(#x)
#define PRAGMA_UNROLL(n) PRAGMA_UNROLL_HELPER(unroll n)

__kernel void kernel_carry(__global ulong* restrict x,
                           __global ulong* restrict carry_array,
                           __global ulong* restrict digit_width)
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end = start + LOCAL_PROPAGATION_DEPTH; 
    ulong carry = 0UL; 
    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4)
    for (ulong i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);
        ulong4 digit_width_vec = vload4(0, digit_width + i);

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry); 
        vstore4(x_vec, 0, x + i);
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
    const ulong end = start + LOCAL_PROPAGATION_DEPTH - 4;  

    const ulong prev_gid = (gid == 0) ? (CARRY_WORKER - 1) : (gid - 1);
    ulong carry = carry_array[prev_gid];

    if (carry == 0) return;

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4_MIN)
    for (ulong i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);
        ulong4 digit_width_vec = vload4(0, digit_width + i);
        x_vec = digit_adc4(x_vec, digit_width_vec, &carry); 
        vstore4(x_vec, 0, x + i);
        if (carry == 0) return;
    }
    if (carry != 0) {
        ulong4 x_vec = vload4(0, x + end);
        ulong4 digit_width_vec = vload4(0, digit_width + end);
        x_vec = digit_adc4_last(x_vec, digit_width_vec, &carry); 
        vstore4(x_vec, 0, x + end);
    }


}

__kernel void kernel_inverse_ntt_radix4_mm(__global ulong* restrict x,
                                                __constant ulong* restrict wi,
                                                const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    ulong4 coeff = (ulong4)( x[base + 0*m],
                              x[base + 1*m],
                              x[base + 2*m],
                              x[base + 3*m] );
    ulong4 twiddles = (ulong4)(1UL, wi[twiddle_offset + 1], wi[twiddle_offset + 0], wi[twiddle_offset + 2]);
    ulong4 u = modMul4(coeff, twiddles);
    ulong4 v;
    v.s0 = modAdd(u.s0, u.s1);
    v.s1 = modSub(u.s0, u.s1);
    v.s2 = modAdd(u.s2, u.s3);
    v.s3 = modMuli(modSub(u.s3, u.s2));
    x[base + 0*m] = modAdd(v.s0, v.s2);
    x[base + 1*m] = modAdd(v.s1, v.s3);
    x[base + 2*m] = modSub(v.s0, v.s2);
    x[base + 3*m] = modSub(v.s1, v.s3);
}

__kernel void kernel_ntt_radix4_last_m1_n4(__global ulong* restrict x,
                                               __global ulong* restrict w,
                                               __global ulong* restrict digit_weight) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    const ulong4 coeff = vload4(0, x + 4 * k);
    const ulong4 dw = vload4(0, digit_weight + 4 * k);
    const ulong w2 = w[twiddle_offset];
    const ulong w1 = w[twiddle_offset + 1];
    const ulong w12 = w[twiddle_offset + 2];
    const ulong4 u = modMul4(coeff, dw);
    const ulong v0 = modAdd(u.s0, u.s2);
    const ulong v1 = modAdd(u.s1, u.s3);
    const ulong v2 = modSub(u.s0, u.s2);
    const ulong v3 = modMuli(modSub(u.s1, u.s3));
    ulong4 result = (ulong4)( modAdd(v0, v1),
                              modMul(modSub(v0, v1), w1),
                              modMul(modAdd(v2, v3), w2),
                              modMul(modSub(v2, v3), w12) );
    result = modMul4(result, result);
    vstore4(result, 0, x + 4 * k);
}

__kernel void kernel_inverse_ntt_radix4_mm_last(__global ulong* restrict x,
                                                     __constant ulong* restrict wi,
                                                     __global ulong* restrict digit_invweight,
                                                     const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    ulong4 coeff = (ulong4)( x[base + 0*m],
                              x[base + 1*m],
                              x[base + 2*m],
                              x[base + 3*m] );
    const ulong iw2 = wi[twiddle_offset + 0];
    const ulong iw1 = wi[twiddle_offset + 1];
    const ulong iw12 = wi[twiddle_offset + 2];
    ulong4 twiddles = (ulong4)(1UL, iw1, iw2, iw12);
    ulong4 u = modMul4(coeff, twiddles);
    ulong4 v;
    v.s0 = modAdd(u.s0, u.s1);
    v.s1 = modSub(u.s0, u.s1);
    v.s2 = modAdd(u.s2, u.s3);
    v.s3 = modMuli(modSub(u.s3, u.s2));
    ulong4 invWeight = (ulong4)( digit_invweight[base + 0*m],
                                 digit_invweight[base + 1*m],
                                 digit_invweight[base + 2*m],
                                 digit_invweight[base + 3*m] );
    ulong4 temp;
    temp.s0 = modAdd(v.s0, v.s2);
    temp.s1 = modAdd(v.s1, v.s3);
    temp.s2 = modSub(v.s0, v.s2);
    temp.s3 = modSub(v.s1, v.s3);
    ulong4 result = modMul4(temp, invWeight);
    x[base + 0*m] = result.s0;
    x[base + 1*m] = result.s1;
    x[base + 2*m] = result.s2;
    x[base + 3*m] = result.s3;
}

__kernel void kernel_ntt_radix4_last_m1(__global ulong* restrict x,
                                             __global ulong* restrict w) {
    const ulong k = get_global_id(0);
    ulong4 coeff = vload4(0, x + 4 * k);
    const ulong twiddle_offset = 6;
    ulong v0 = modAdd(coeff.s0, coeff.s2);
    ulong v1 = modAdd(coeff.s1, coeff.s3);
    ulong v2 = modSub(coeff.s0, coeff.s2);
    ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));
    ulong4 tmp;
    tmp.s0 = modAdd(v0, v1);
    tmp.s1 = modSub(v0, v1);
    tmp.s2 = modAdd(v2, v3);
    tmp.s3 = modSub(v2, v3);
    ulong4 factors = (ulong4)(1UL, w[twiddle_offset + 1], w[twiddle_offset + 0], w[twiddle_offset + 2]);
    ulong4 result = modMul4(tmp, factors);
    result = modMul4(result, result);
    vstore4(result, 0, x + 4 * k);
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
    ulong4 coeff = (ulong4)( x[i + 0*m], x[i + 1*m], x[i + 2*m], x[i + 3*m] );
    ulong4 weight = (ulong4)( digit_weight[i + 0*m], digit_weight[i + 1*m], digit_weight[i + 2*m], digit_weight[i + 3*m] );
    ulong4 u = modMul4(coeff, weight);
    ulong v0 = modAdd(u.s0, u.s2);
    ulong v1 = modAdd(u.s1, u.s3);
    ulong v2 = modSub(u.s0, u.s2);
    ulong v3 = modMuli(modSub(u.s1, u.s3));
    ulong4 tmp;
    tmp.s0 = modAdd(v0, v1);
    tmp.s1 = modSub(v0, v1);
    tmp.s2 = modAdd(v2, v3);
    tmp.s3 = modSub(v2, v3);
    ulong4 factors = (ulong4)(1UL, w1, w2, w12);
    ulong4 result = modMul4(tmp, factors);
    x[i + 0*m] = result.s0;
    x[i + 1*m] = result.s1;
    x[i + 2*m] = result.s2;
    x[i + 3*m] = result.s3;
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
    ulong4 coeff = (ulong4)( x[i + 0*m], x[i + 1*m], x[i + 2*m], x[i + 3*m] );
    ulong v0 = modAdd(coeff.s0, coeff.s2);
    ulong v1 = modAdd(coeff.s1, coeff.s3);
    ulong v2 = modSub(coeff.s0, coeff.s2);
    ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));
    ulong4 tmp;
    tmp.s0 = modAdd(v0, v1);
    tmp.s1 = modSub(v0, v1);
    tmp.s2 = modAdd(v2, v3);
    tmp.s3 = modSub(v2, v3);
    ulong4 factors = (ulong4)(1UL, w1, w2, w12);
    ulong4 result = modMul4(tmp, factors);
    x[i + 0*m] = result.s0;
    x[i + 1*m] = result.s1;
    x[i + 2*m] = result.s2;
    x[i + 3*m] = result.s3;
}

__kernel void kernel_inverse_ntt_radix4_m1(__global ulong* restrict x,
                                               __global ulong* restrict wi) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    ulong4 coeff = vload4(0, x + 4 * k);
    const ulong iw2 = wi[twiddle_offset];
    const ulong iw1 = wi[twiddle_offset + 1];
    const ulong iw12 = wi[twiddle_offset + 2];
    ulong4 twiddles = (ulong4)(1UL, iw1, iw2, iw12);
    ulong4 u = modMul4(coeff, twiddles);
    ulong v0 = modAdd(u.s0, u.s1);
    ulong v1 = modSub(u.s0, u.s1);
    ulong v2 = modAdd(u.s2, u.s3);
    ulong v3 = modMuli(modSub(u.s3, u.s2));
    ulong4 result = (ulong4)( modAdd(v0, v2),
                              modAdd(v1, v3),
                              modSub(v0, v2),
                              modSub(v1, v3) );
    vstore4(result, 0, x + 4 * k);
}

__kernel void kernel_inverse_ntt_radix4_m1_n4(__global ulong* restrict x,
                                                  __constant ulong* restrict wi,
                                                  __global ulong* restrict digit_invweight) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    ulong4 coeff = vload4(0, x + 4 * k);
    ulong4 div = vload4(0, digit_invweight + 4 * k);
    const ulong iw2 = wi[twiddle_offset];
    const ulong iw1 = wi[twiddle_offset + 1];
    const ulong iw12 = wi[twiddle_offset + 2];
    ulong4 twiddles = (ulong4)(1UL, iw1, iw2, iw12);
    ulong4 u = modMul4(coeff, twiddles);
    ulong v0 = modAdd(u.s0, u.s1);
    ulong v1 = modSub(u.s0, u.s1);
    ulong v2 = modAdd(u.s2, u.s3);
    ulong v3 = modMuli(modSub(u.s3, u.s2));
    ulong4 tmp;
    tmp.s0 = modAdd(v0, v2);
    tmp.s1 = modAdd(v1, v3);
    tmp.s2 = modSub(v0, v2);
    tmp.s3 = modSub(v1, v3);
    ulong4 result = modMul4(tmp, div);
    vstore4(result, 0, x + 4 * k);
}

__kernel void kernel_ntt_radix4_mm_2steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          const ulong m) {
    uint ii;
    ulong k;
    k = (get_global_id(0)/(m/4));
    k = k*m;
    k += get_global_id(0)%(m/4);
    ulong local_x[16];
    int write_index = 0;

    #pragma unroll 4
    for (ii = 0; ii < 4; ii++) {
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;
        const ulong twiddle_offset = 6 * m + 3 * j;
        const ulong w2  = w[twiddle_offset];
        const ulong w1  = w[twiddle_offset + 1];
        const ulong w12 = w[twiddle_offset + 2];
        
        ulong4 coeff = (ulong4)( x[i + 0 * m], x[i + 1 * m], x[i + 2 * m], x[i + 3 * m] );
        ulong v0 = modAdd(coeff.s0, coeff.s2);
        ulong v1 = modAdd(coeff.s1, coeff.s3);
        ulong v2 = modSub(coeff.s0, coeff.s2);
        ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));
        ulong4 tmp;
        tmp.s0 = modAdd(v0, v1);
        tmp.s1 = modSub(v0, v1);
        tmp.s2 = modAdd(v2, v3);
        tmp.s3 = modSub(v2, v3);
        ulong4 factors = (ulong4)(1UL, w1, w2, w12);
        ulong4 result = modMul4(tmp, factors);
        
        local_x[write_index] = result.s0;
        local_x[write_index+1] = result.s1;
        local_x[write_index+2] = result.s2;
        local_x[write_index+3] = result.s3;
        write_index += 4;
        k += m/4;
    }
    
    const ulong new_m = m / 4;
    write_index = 0;
    int indice1[16] = {0, 4, 8, 12,
                       1, 5, 9, 13,
                       2, 6, 10, 14,
                       3, 7, 11, 15};
    k = (get_global_id(0)/(m/4));
    k = k*m;
    k += get_global_id(0)%(m/4);
    
    #pragma unroll 4
    for (ii = 0; ii < 4; ii++) {
        const ulong j = k & (new_m - 1);
        const ulong i = 4 * (k - j) + j;
        const ulong twiddle_offset = 6 * new_m + 3 * j;
        const ulong w2  = w[twiddle_offset];
        const ulong w1  = w[twiddle_offset + 1];
        const ulong w12 = w[twiddle_offset + 2];
        
        ulong4 coeff = (ulong4)( local_x[indice1[write_index]],
                                 local_x[indice1[write_index + 1]],
                                 local_x[indice1[write_index + 2]],
                                 local_x[indice1[write_index + 3]] );
        write_index += 4;
        ulong v0 = modAdd(coeff.s0, coeff.s2);
        ulong v1 = modAdd(coeff.s1, coeff.s3);
        ulong v2 = modSub(coeff.s0, coeff.s2);
        ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));
        ulong4 tmp;
        tmp.s0 = modAdd(v0, v1);
        tmp.s1 = modSub(v0, v1);
        tmp.s2 = modAdd(v2, v3);
        tmp.s3 = modSub(v2, v3);
        ulong4 factors = (ulong4)(1UL, w1, w2, w12);
        ulong4 result = modMul4(tmp, factors);
        
        x[i + 0 * new_m] = result.s0;
        x[i + 1 * new_m] = result.s1;
        x[i + 2 * new_m] = result.s2;
        x[i + 3 * new_m] = result.s3;
        k += m/4;
    }
}
