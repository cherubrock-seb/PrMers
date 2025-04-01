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

inline ulong4 digit_adc4(ulong4 lhs, int4 digit_width, __private ulong *restrict carry) {
    ulong4 res;
    ulong c = *carry;

    if (digit_width.s0 == digit_width.s1 &&
        digit_width.s1 == digit_width.s2 &&
        digit_width.s2 == digit_width.s3) {

        ulong w = (ulong) digit_width.s0;
        ulong mask = (1UL << ((uint)w)) - 1UL;

        ulong s = lhs.s0 + c;
        res.s0 = s & mask;
        c = s >> w;

        s = lhs.s1 + c;
        res.s1 = s & mask;
        c = s >> w;

        s = lhs.s2 + c;
        res.s2 = s & mask;
        c = s >> w;

        s = lhs.s3 + c;
        res.s3 = s & mask;
        c = s >> w;

        *carry = c;
    }
    else if (digit_width.s0 == digit_width.s1 &&
             digit_width.s1 == digit_width.s2) {

        ulong w = (ulong) digit_width.s0;
        ulong mask = (1UL << ((uint)w)) - 1UL;

        ulong s = lhs.s0 + c;
        res.s0 = s & mask;
        c = s >> w;

        s = lhs.s1 + c;
        res.s1 = s & mask;
        c = s >> w;

        s = lhs.s2 + c;
        res.s2 = s & mask;
        c = s >> w;

        // Pour le 4ème, on utilise sa propre largeur
        ulong w3 = (ulong) digit_width.s3;
        ulong mask3 = (1UL << ((uint)w3)) - 1UL;
        s = lhs.s3 + c;
        res.s3 = s & mask3;
        c = s >> ((uint)w3);

        *carry = c;
    }
    else if (digit_width.s0 == digit_width.s1 &&
             digit_width.s2 == digit_width.s3) {

        ulong w01 = (ulong) digit_width.s0;
        ulong mask01 = (1UL << ((uint)w01)) - 1UL;
        ulong s = lhs.s0 + c;
        res.s0 = s & mask01;
        c = s >> w01;
        s = lhs.s1 + c;
        res.s1 = s & mask01;
        c = s >> w01;

        ulong w23 = (ulong) digit_width.s2;
        ulong mask23 = (1UL << ((uint)w23)) - 1UL;
        s = lhs.s2 + c;
        res.s2 = s & mask23;
        c = s >> w23;
        s = lhs.s3 + c;
        res.s3 = s & mask23;
        c = s >> w23;

        *carry = c;
    }
    // Cas 4 : Alternance (s0 == s2) et (s1 == s3)
    else if (digit_width.s0 == digit_width.s2 &&
             digit_width.s1 == digit_width.s3) {

        ulong w0 = (ulong) digit_width.s0;
        ulong mask0 = (1UL << ((uint)w0)) - 1UL;
        ulong s = lhs.s0 + c;
        res.s0 = s & mask0;
        c = s >> w0;

        ulong w1 = (ulong) digit_width.s1;
        ulong mask1 = (1UL << ((uint)w1)) - 1UL;
        s = lhs.s1 + c;
        res.s1 = s & mask1;
        c = s >> w1;

        // Pour s2, on réutilise la largeur de s0
        s = lhs.s2 + c;
        res.s2 = s & mask0;
        c = s >> w0;

        // Pour s3, on réutilise la largeur de s1
        s = lhs.s3 + c;
        res.s3 = s & mask1;
        c = s >> w1;

        *carry = c;
    }
    // Cas générique
    else {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            ulong s = lhs[i] + c;
            res[i] = s & ((1UL << ((uint)digit_width[i])) - 1UL);
            c = s >> ((uint)digit_width[i]);
        }
        *carry = c;
    }

    return res;
}

inline ulong4 digit_adc4_last(ulong4 lhs, int4 digit_width, __private ulong *carry) {
    ulong4 res;
    uint64_t c = *carry;
    #pragma unroll 3
    for (int i = 0; i < 3; i++) {
        uint64_t s = lhs[i] + c;                        
        res[i] = s & ((1UL << ((uint)digit_width[i])) - 1UL);
        c = s >> ((uint)digit_width[i]);

    }
    *carry = c;
    res[3] = lhs[3] + c;
    return res;
}

__kernel void kernel_sub2(__global ulong* restrict x,
                          __global int* restrict digit_width,
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
                           __global int* restrict digit_width)
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end = start + LOCAL_PROPAGATION_DEPTH; 
    ulong carry = 0UL; 
    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4)
    for (ulong i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);

        int4 digit_width_vec = (int4)(digit_width[i],
                                      digit_width[i + 1],
                                      digit_width[i + 2],
                                      digit_width[i + 3]);

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);
        vstore4(x_vec, 0, x + i);
    }
    
    if (carry != 0) {
        carry_array[gid] = carry;
    }
}



__kernel void kernel_carry_2(__global ulong* restrict x,
                             __global ulong* restrict carry_array,
                             __global int* restrict digit_width)  // CHANGEMENT ICI
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

        ulong4 digit_width_vec = (ulong4)(
            digit_width[i + 0],
            digit_width[i + 1],
            digit_width[i + 2],
            digit_width[i + 3]
        );

        x_vec = digit_adc4(x_vec, convert_int4(digit_width_vec), &carry);

        vstore4(x_vec, 0, x + i);

        if (carry == 0) return;
    }

    if (carry != 0) {
        ulong4 x_vec = vload4(0, x + end);

        ulong4 digit_width_vec = (ulong4)(
            digit_width[end + 0],
            digit_width[end + 1],
            digit_width[end + 2],
            digit_width[end + 3]
        );

        x_vec = digit_adc4_last(x_vec, convert_int4(digit_width_vec), &carry); 
        vstore4(x_vec, 0, x + end);
    }
}

static inline ulong4 butterfly(const ulong4 u) {
    return (ulong4)(
        modAdd(u.s0, u.s1),
        modSub(u.s0, u.s1),
        modAdd(u.s2, u.s3),
        modMuli(modSub(u.s3, u.s2))
    );
}

__kernel void kernel_inverse_ntt_radix4_mm(__global ulong* restrict x,
                                            __constant ulong* restrict wi,
                                            const ulong m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    
    const ulong a = x[base];
    const ulong b = x[base + m];
    const ulong c = x[base + 2 * m];
    const ulong d = x[base + 3 * m];
    const ulong4 coeff = (ulong4)(a, b, c, d);
    
    const ulong4 tmp = vload4(0, wi + twiddle_offset);
    const ulong4 twiddles = (ulong4)(1UL, tmp.s1, tmp.s0, tmp.s2);
    
    const ulong4 u = modMul4(coeff, twiddles);
    const ulong4 r = butterfly(u);
    
    x[base]         = modAdd(r.s0, r.s2);
    x[base + m]     = modAdd(r.s1, r.s3);
    x[base + 2 * m] = modSub(r.s0, r.s2);
    x[base + 3 * m] = modSub(r.s1, r.s3);
}
__kernel void kernel_ntt_radix4_last_m1_n4(__global ulong* restrict x,
                                           __global ulong* restrict w,
                                           __global ulong* restrict digit_weight) {
    const ulong k = get_global_id(0);
    const ulong base = 4 * k;
    ulong4 coeff = vload4(0, x + base);
    ulong4 dw = vload4(0, digit_weight + base);
    ulong4 u = modMul4(coeff, dw);
    ulong4 tw = vload4(0, w + 6);
    const ulong tw1 = tw.s1, tw0 = tw.s0, tw2 = tw.s2;
    const ulong t0 = modAdd(u.s0, u.s2);
    const ulong t1 = modAdd(u.s1, u.s3);
    const ulong t2 = modSub(u.s0, u.s2);
    const ulong t3 = modSub(u.s1, u.s3);
    const ulong t3i = modMuli(t3);
    const ulong r0 = modAdd(t0, t1);
    const ulong r1 = modMul(modSub(t0, t1), tw1);
    const ulong r2 = modMul(modAdd(t2, t3i), tw0);
    const ulong r3 = modMul(modSub(t2, t3i), tw2);
    ulong4 r = (ulong4)(r0, r1, r2, r3);
    r = modMul4(r, r);
    vstore4(r, 0, x + base);
}



__kernel void kernel_inverse_ntt_radix4_mm_last(__global ulong* restrict x,
                                                 __constant ulong* restrict wi,
                                                 __global ulong* restrict digit_invweight,
                                                 const ulong m)
{
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 3 * (2 * m + j);
    ulong4 coeff = (ulong4)(x[base], x[base + m], x[base + 2*m], x[base + 3*m]);
    const ulong4 twiddles = (ulong4)(1UL, wi[twiddle_offset+1], wi[twiddle_offset+0], wi[twiddle_offset+2]);
    ulong4 u = modMul4(coeff, twiddles);
    ulong4 v = (ulong4)(modAdd(u.s0, u.s1),
                         modSub(u.s0, u.s1),
                         modAdd(u.s2, u.s3),
                         modMuli(modSub(u.s3, u.s2)));
    ulong4 invWeight = (ulong4)(digit_invweight[base],
                                 digit_invweight[base + m],
                                 digit_invweight[base + 2*m],
                                 digit_invweight[base + 3*m]);
    ulong4 temp = (ulong4)(modAdd(v.s0, v.s2),
                            modAdd(v.s1, v.s3),
                            modSub(v.s0, v.s2),
                            modSub(v.s1, v.s3));
    ulong4 result = modMul4(temp, invWeight);
    x[base] = result.s0;
    x[base + m] = result.s1;
    x[base + 2*m] = result.s2;
    x[base + 3*m] = result.s3;
}


__kernel void kernel_ntt_radix4_last_m1(__global ulong* restrict x,
                                        __global ulong* restrict w)
{
    const ulong k = get_global_id(0);
    ulong4 coeff = vload4(0, x + 4 * k);
    ulong4 wvec = vload4(0, w + 6);
    ulong4 fac = (ulong4)(1UL, wvec.s1, wvec.s0, wvec.s2);
    
    ulong a = modAdd(coeff.s0, coeff.s2);
    ulong b = modAdd(coeff.s1, coeff.s3);
    ulong c = modSub(coeff.s0, coeff.s2);
    ulong d = modMuli(modSub(coeff.s1, coeff.s3));
    
    coeff.s0 = modAdd(a, b);
    coeff.s1 = modSub(a, b);
    coeff.s2 = modAdd(c, d);
    coeff.s3 = modSub(c, d);
    
    coeff = modMul4(coeff, fac);
    coeff = modMul4(coeff, coeff); 
    
    vstore4(coeff, 0, x + 4 * k);
}



__kernel void kernel_ntt_radix4_mm_first(__global ulong* restrict x,
                                         __global ulong* restrict w,
                                         __global ulong* restrict digit_weight,
                                         const ulong m)
{
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    const ulong tw = 6 * m + 3 * j;
    const ulong w2 = w[tw];
    const ulong w1 = w[tw + 1];
    const ulong w12 = w[tw + 2];
    const ulong i0 = i, i1 = i + m, i2 = i + (m << 1), i3 = i + 3 * m;
    ulong4 c = (ulong4)(x[i0], x[i1], x[i2], x[i3]);
    ulong4 wt = (ulong4)(digit_weight[i0], digit_weight[i1], digit_weight[i2], digit_weight[i3]);
    c = modMul4(c, wt);
    const ulong a = modAdd(c.s0, c.s2);
    const ulong b = modAdd(c.s1, c.s3);
    const ulong d = modSub(c.s0, c.s2);
    const ulong e = modMuli(modSub(c.s1, c.s3));
    c.s0 = modAdd(a, b);
    c.s1 = modSub(a, b);
    c.s2 = modAdd(d, e);
    c.s3 = modSub(d, e);
    ulong4 fac = (ulong4)(1UL, w1, w2, w12);
    c = modMul4(c, fac);
    x[i0] = c.s0;
    x[i1] = c.s1;
    x[i2] = c.s2;
    x[i3] = c.s3;
}

__kernel void kernel_ntt_radix4_mm(__global ulong* restrict x,
                                   __global ulong* restrict w,
                                   const ulong m)
{
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    const ulong tw = 6 * m + 3 * j;
    const ulong w2 = w[tw];
    const ulong w1 = w[tw + 1];
    const ulong w12 = w[tw + 2];
    const ulong i0 = i, i1 = i + m, i2 = i + (m << 1), i3 = i + 3 * m;
    ulong4 c = (ulong4)(x[i0], x[i1], x[i2], x[i3]);
    const ulong a = modAdd(c.s0, c.s2);
    const ulong b = modAdd(c.s1, c.s3);
    const ulong d = modSub(c.s0, c.s2);
    const ulong e = modMuli(modSub(c.s1, c.s3));
    c.s0 = modAdd(a, b);
    c.s1 = modSub(a, b);
    c.s2 = modAdd(d, e);
    c.s3 = modSub(d, e);
    ulong4 fac = (ulong4)(1UL, w1, w2, w12);
    c = modMul4(c, fac);
    x[i0] = c.s0;
    x[i1] = c.s1;
    x[i2] = c.s2;
    x[i3] = c.s3;
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
    coeff.s0 = modAdd(v0, v2);
    coeff.s1 = modAdd(v1, v3);
    coeff.s2 = modSub(v0, v2);
    coeff.s3 = modSub(v1, v3);
    ulong4 result = modMul4(coeff, div);
    vstore4(result, 0, x + 4 * k);
}



__kernel void kernel_ntt_radix4_inverse_mm_2steps(__global ulong* restrict x,
                                          __global ulong* restrict wi,
                                          const ulong m) {
    uint ii;
    ulong k;
    k = (get_global_id(0))/(m);
    k = k*m*4;
    k += get_global_id(0)%(m);
    //printf("get_global_id(0) = %lu ; start k=%lu",get_global_id(0) ,k);
    ulong local_x[16];
    int write_index = 0;

    #pragma unroll 4
    for (ii = 0; ii < 4; ii++) {
        const ulong j = k & (m - 1);
        const ulong base = 4 * (k - j) + j;
        //printf("get_global_id(0) = %lu ; base=%lu",get_global_id(0) ,base);
        const ulong twiddle_offset = 6 * m + 3 * j;
        //printf("get_global_id(0) = %lu ; m=%lu; (%lu,%lu,%lu,%lu)",get_global_id(0),m,base + 0 * m,base + 1 * m,base + 2 * m,base + 3 * m);
        ulong4 coeff = (ulong4)( x[base + 0 * m], x[base + 1 * m], x[base + 2 * m], x[base + 3 * m] );        
        const ulong4 tmp = vload4(0, wi + twiddle_offset);
        const ulong4 twiddles = (ulong4)(1UL, tmp.s1, tmp.s0, tmp.s2);
        ulong4 u = modMul4(coeff, twiddles);
        const ulong4 r = butterfly(u);        
        local_x[write_index]       = modAdd(r.s0, r.s2);
        local_x[write_index+1]     = modAdd(r.s1, r.s3);
        local_x[write_index+2]     = modSub(r.s0, r.s2);
        local_x[write_index+3]     = modSub(r.s1, r.s3);
        //printf("WRITE get_global_id(0) = %lu ; m=%lu; write_index = (%i,%i,%i,%i)",get_global_id(0),m,write_index,write_index+1, write_index+2,write_index+3);
        
        write_index += 4;
        k += m;
    }
    
    const ulong new_m = m * 4;
    write_index = 0;
    k = (get_global_id(0))/(m);
    k = k*m*4;
    k += get_global_id(0)%(m);

    #pragma unroll 4
    for (ii = 0; ii < 4; ii++) {
        const ulong j = k & (new_m - 1);
        const ulong base = 4 * (k - j) + j;
        const ulong twiddle_offset = 6 * new_m + 3 * j;

ulong4 coeff = (ulong4)(
    local_x[((write_index  ) % 4) * 4 + (write_index  ) / 4],
    local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4],
    local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4],
    local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]
);
        write_index += 4;
        const ulong4 tmp = vload4(0, wi + twiddle_offset);
        const ulong4 twiddles = (ulong4)(1UL, tmp.s1, tmp.s0, tmp.s2);
        ulong4 u = modMul4(coeff, twiddles);
        const ulong4 r = butterfly(u);
        
        x[base + 0 * new_m]      = modAdd(r.s0, r.s2);
        x[base + 1 * new_m]      = modAdd(r.s1, r.s3);
        x[base + 2 * new_m]      = modSub(r.s0, r.s2);
        x[base + 3 * new_m]      = modSub(r.s1, r.s3);
        k += m;
    }
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
        
        ulong4 coeff = (ulong4)(
            local_x[((write_index  ) % 4) * 4 + (write_index  ) / 4],
            local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4],
            local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4],
            local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]
        );
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


__kernel void kernel_ntt_radix4_mm_3steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          ulong m) {
    uint ii;
    ulong k;
    k = (get_global_id(0)/(m/16));
    k = k*m;
    k += get_global_id(0)%(m/16);
    ulong local_x[64];
    int write_index = 0;
    uint iii;
    
    #pragma unroll 4
    for (iii = 0; iii < 4; iii++) {
        k = (get_global_id(0)/(m/16));
        k = k*m;
        k += get_global_id(0)%(m/16);
        k += iii*(m/16);
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
            ////printf("Step 1 Thread %lu i=%lu m=%lu write_index = %lu",get_global_id(0), i,m, write_index);
            write_index += 4;
            k += m/4;
        }  
    }  
    m = m / 4;
    write_index = 0;
    int indice1[64] = {0, 4, 8, 12,
                       1, 5, 9, 13,
                       2, 6, 10, 14,
                       3, 7, 11, 15,
                       16,20,24,28,
                       17,21,25,29,
                       18,22,26,30,
                       19,23,27,31,
                       32,36,40,44,
                       33,37,41,45,
                       34,38,42,46,
                       35,39,43,47,
                       48,52,56,60,
                       49,53,57,61,
                       50,54,58,62,
                       51,55,59,63,
                       };

    k = (get_global_id(0)/(m*4/16));
    k = k*m*4;
    k += get_global_id(0)%(m*4/16);
    
    
    #pragma unroll 4
    for (iii = 0; iii < 4; iii++) {
        k = (get_global_id(0)/(m*4/16));
        k = k*m*4;
        k += get_global_id(0)%(m*4/16);
        k += iii*(m/4);
        #pragma unroll 4
        for (ii = 0; ii < 4; ii++) {
            const ulong j = k & (m - 1);
            const ulong i = 4 * (k - j) + j;
            const ulong twiddle_offset = 6 * m + 3 * j;
            const ulong w2  = w[twiddle_offset];
            const ulong w1  = w[twiddle_offset + 1];
            const ulong w12 = w[twiddle_offset + 2];
            
            ulong4 coeff = (ulong4)( local_x[indice1[write_index]],
                                    local_x[indice1[write_index + 1]],
                                    local_x[indice1[write_index + 2]],
                                    local_x[indice1[write_index + 3]] );
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
            
            local_x[indice1[write_index]] = result.s0;
            local_x[indice1[write_index + 1]] = result.s1;
            local_x[indice1[write_index + 2]] = result.s2;
            local_x[indice1[write_index + 3]] = result.s3;
            ////printf("Step 2 Thread %lu i=%lu m=%lu indice1[write_index] = %lu",get_global_id(0), i,m,indice1[write_index]);
            write_index += 4;
            k += m;
        }  
    }


    m = m / 4;
    write_index = 0;
    int indice2[64] = {
        0, 16, 32, 48,
        4, 20, 36, 52,
        8, 24, 40, 56,
        12, 28, 44, 60,

        1, 17, 33, 49,
        5, 21, 37, 53,
        9, 25, 41, 57,
        13, 29, 45, 61,

        2, 18, 34, 50,
        6, 22, 38, 54,
        10, 26, 42, 58,
        14, 30, 46, 62,

        3, 19, 35, 51,
        7, 23, 39, 55,
        11, 27, 43, 59,
        15, 31, 47, 63
    };



    k = (get_global_id(0)/(m*16/16));
    k = k*m*16;
    k += get_global_id(0)%(m*16/16);
    
    
    #pragma unroll 4
    for (iii = 0; iii < 4; iii++) {
        k = (get_global_id(0)/(m*16/16));
        k = k*m*16;
        k += get_global_id(0)%(m*16/16);
        k += iii*(m*4);
        #pragma unroll 4
        for (ii = 0; ii < 4; ii++) {
            const ulong j = k & (m - 1);
            const ulong i = 4 * (k - j) + j;
            const ulong twiddle_offset = 6 * m + 3 * j;
            const ulong w2  = w[twiddle_offset];
            const ulong w1  = w[twiddle_offset + 1];
            const ulong w12 = w[twiddle_offset + 2];
            
            ulong4 coeff = (ulong4)( local_x[indice2[write_index]],
                                    local_x[indice2[write_index + 1]],
                                    local_x[indice2[write_index + 2]],
                                    local_x[indice2[write_index + 3]] );
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
            
            x[i + 0 * m] = result.s0;
            x[i + 1 * m] = result.s1;
            x[i + 2 * m] = result.s2;
            x[i + 3 * m] = result.s3;
            ////printf("Step 3 Thread %lu i=%lu m=%lu indice2[write_index] = %lu",get_global_id(0), i,m,indice2[write_index]);

            write_index += 4;
            k += m;
        }  
    }

}

__kernel void kernel_ntt_radix2_square_radix2(__global ulong* restrict x)
{
    const uint idx = get_global_id(0) << 1; // équivalent à get_global_id(0) * 2

    ulong2 u = vload2(0, x + idx);

    ulong s = modAdd(u.x, u.y);
    ulong d = modSub(u.x, u.y);

    s = modMul(s, s);
    d = modMul(d, d);

    ulong r0 = modAdd(s, d);
    ulong r1 = modSub(s, d);

    vstore2((ulong2)(r0, r1), 0, x + idx);
}

