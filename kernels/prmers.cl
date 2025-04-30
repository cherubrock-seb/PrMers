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
typedef uint  uint32_t;
typedef ulong uint64_t;

// Definition of the prime modulus and its complement
#define MOD_P  0xffffffff00000001UL  // p = 2^64 - 2^32 + 1
#define MOD_P_COMP 0xffffffffU       // 2^64 - p = 2^32 - 1

// Constants stored in __constant memory for fast GPU access
__constant ulong  mod_p_const         = MOD_P;
__constant uint   mod_p_comp_const    = MOD_P_COMP;
__constant ulong4 mod_p4_const        = (ulong4)(MOD_P, MOD_P, MOD_P, MOD_P);
__constant ulong4 mod_p_comp4_const   = (ulong4)(MOD_P_COMP, MOD_P_COMP, MOD_P_COMP, MOD_P_COMP);
__constant ulong2 mod_p2_const       = (ulong2)(MOD_P,      MOD_P);
__constant ulong2 mod_p_comp2_const  = (ulong2)(MOD_P_COMP, MOD_P_COMP);
// Modular addition for vectors of 4 ulong
inline ulong4 modAdd4(ulong4 lhs, ulong4 rhs) {
    // 'select' avoids branching (if-else) on GPU
    ulong4 c = select((ulong4)0, mod_p_comp4_const, lhs >= (mod_p4_const - rhs));
    return lhs + rhs + c;
}

// Modular subtraction for vectors of 4 ulong
inline ulong4 modSub4(ulong4 lhs, ulong4 rhs) {
    ulong4 c = select((ulong4)0, mod_p_comp4_const, lhs < rhs);
    return lhs - rhs - c;
}

// Modular addition for scalar ulong
inline ulong modAdd(const ulong lhs, const ulong rhs) {
    const uint c = (lhs >= mod_p_const - rhs) ? mod_p_comp_const : 0;
    return lhs + rhs + c;
}

// Modular subtraction for scalar ulong
inline ulong modSub(const ulong lhs, const ulong rhs) {
    const uint c = (lhs < rhs) ? mod_p_comp_const : 0;
    return lhs - rhs - c;
}

// Modular reduction of a 128-bit product (lo, hi) to a 64-bit result mod p
inline ulong Reduce(const ulong lo, const ulong hi) {
    const uint c = (lo >= mod_p_const) ? mod_p_comp_const : 0;
    ulong r = lo + c;
    r = modAdd(r, hi << 32);                           // Add hi * 2^32
    r = modSub(r, (hi >> 32) + (uint)hi);              // Subtract hi_high + hi_low
    return r;
}

// Vectorized version of Reduce: reduces four (lo, hi) pairs simultaneously
inline ulong4 Reduce4(ulong4 lo, ulong4 hi) {
    ulong4 c = select((ulong4)0, mod_p_comp4_const, lo >= mod_p4_const);
    ulong4 r = lo + c;

    ulong4 hi_shifted = hi << 32;
    ulong4 hi_high = hi >> 32;
    ulong4 hi_low  = convert_ulong4(convert_uint4(hi));
    ulong4 hi_reduced = hi_high + hi_low;

    r = modAdd4(r, hi_shifted);
    r = modSub4(r, hi_reduced);
    return r;
}

// Modular multiplication of 4 ulong vectors
inline ulong4 modMul4(ulong4 lhs, ulong4 rhs) {
    ulong4 lo = lhs * rhs;
    ulong4 hi = mul_hi(lhs, rhs);
    return Reduce4(lo, hi);
}



// Computes the full 128-bit product of two 64-bit integers
inline void mul128(ulong a, ulong b, __private ulong *hi, __private ulong *lo) {
    *lo = a * b;
    *hi = mul_hi(a, b);
}

// Scalar modular multiplication using full 128-bit product
inline ulong modMul(const ulong lhs, const ulong rhs) {
    const ulong lo = lhs * rhs;
    const ulong hi = mul_hi(lhs, rhs);
    return Reduce(lo, hi);
}


// Modular addition for vectors of 2 ulong
inline ulong2 modAdd2(ulong2 lhs, ulong2 rhs) {
    ulong2 c = select((ulong2)0,
                      mod_p_comp2_const,
                      lhs >= (mod_p2_const - rhs));
    return lhs + rhs + c;
}

// Modular subtraction for vectors of 2 ulong
inline ulong2 modSub2(ulong2 lhs, ulong2 rhs) {
    ulong2 c = select((ulong2)0,
                      mod_p_comp2_const,
                      lhs < rhs);
    return lhs - rhs - c;
}

inline ulong2 Reduce2(ulong2 lo, ulong2 hi) {
    ulong2 c = select((ulong2)0, mod_p_comp2_const, lo >= mod_p2_const);
    ulong2 r = lo + c;

    ulong2 hi_shifted = hi << 32;
    ulong2 hi_high    = hi >> 32;
    ulong2 hi_low     = convert_ulong2(convert_uint2(hi));
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


inline ulong4 modMul3_2(const ulong4 lhs,
                        const ulong2 w02,    // vload2 non-swappé
                        const ulong  w3)
{
    ulong2 m12 = modMul2(
        (ulong2)(lhs.s1, lhs.s2),
        (ulong2)(w02.s1, w02.s0)
    );
    ulong m3 = modMul(lhs.s3, w3);
    return (ulong4)(lhs.s0, m12.s0, m12.s1, m3);
}

inline ulong4 modMul3(const ulong4 lhs,
                      const ulong w1,
                      const ulong w2,
                      const ulong w3) {
    ulong4 r;
    r.s0 = lhs.s0;
    r.s1 = modMul(lhs.s1, w1);
    r.s2 = modMul(lhs.s2, w2);
    r.s3 = modMul(lhs.s3, w3);
    return r;
}

// Multiply x by sqrt(-1) mod p, where sqrt(-1) is defined as 2^48 mod p
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

        ulong mask = (1UL << (digit_width.s0)) - 1UL;

        ulong s = lhs.s0 + c;
        res.s0 = s & mask;
        c = s >> digit_width.s0;

        s = lhs.s1 + c;
        res.s1 = s & mask;
        c = s >> digit_width.s0;

        s = lhs.s2 + c;
        res.s2 = s & mask;
        c = s >> digit_width.s0;

        s = lhs.s3 + c;
        res.s3 = s & mask;
        c = s >> digit_width.s0;

        *carry = c;
    }
    else if (digit_width.s0 == digit_width.s1 &&
             digit_width.s1 == digit_width.s2) {

        ulong mask = (1UL << ( digit_width.s0)) - 1UL;

        ulong s = lhs.s0 + c;
        res.s0 = s & mask;
        c = s >>  digit_width.s0;

        s = lhs.s1 + c;
        res.s1 = s & mask;
        c = s >>  digit_width.s0;

        s = lhs.s2 + c;
        res.s2 = s & mask;
        c = s >>  digit_width.s0;

        ulong mask3 = (1UL << (digit_width.s3)) - 1UL;
        s = lhs.s3 + c;
        res.s3 = s & mask3;
        c = s >> (digit_width.s3);

        *carry = c;
    }
    else if (digit_width.s0 == digit_width.s1 &&
             digit_width.s2 == digit_width.s3) {

        ulong mask01 = (1UL << (digit_width.s0)) - 1UL;
        ulong s = lhs.s0 + c;
        res.s0 = s & mask01;
        c = s >> digit_width.s0;
        s = lhs.s1 + c;
        res.s1 = s & mask01;
        c = s >> digit_width.s0;
        ulong mask23 = (1UL << (digit_width.s2)) - 1UL;
        s = lhs.s2 + c;
        res.s2 = s & mask23;
        c = s >> digit_width.s2;
        s = lhs.s3 + c;
        res.s3 = s & mask23;
        c = s >> digit_width.s2;

        *carry = c;
    }
    else if (digit_width.s0 == digit_width.s2 &&
             digit_width.s1 == digit_width.s3) {

        ulong mask0 = (1UL << (digit_width.s0)) - 1UL;
        ulong s = lhs.s0 + c;
        res.s0 = s & mask0;
        c = s >> digit_width.s0;

        ulong mask1 = (1UL << (digit_width.s1)) - 1UL;
        s = lhs.s1 + c;
        res.s1 = s & mask1;
        c = s >> digit_width.s1;

        // Pour s2, on réutilise la largeur de s0
        s = lhs.s2 + c;
        res.s2 = s & mask0;
        c = s >> digit_width.s0;

        // Pour s3, on réutilise la largeur de s1
        s = lhs.s3 + c;
        res.s3 = s & mask1;
        c = s >> digit_width.s1;

        *carry = c;
    }
    else {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            ulong s = lhs[i] + c;
            res[i] = s & ((1UL << (digit_width[i])) - 1UL);
            c = s >> (digit_width[i]);
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
        res[i] = s & ((1UL << (digit_width[i])) - 1UL);
        c = s >> (digit_width[i]);

    }
    *carry = c;
    res[3] = lhs[3] + c;
    return res;
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
#ifndef WORKER_NTT
#define WORKER_NTT 1
#endif
#ifndef WORKER_NTT_2_STEPS
#define WORKER_NTT_2_STEPS 1
#endif
#ifndef MODULUS_P
#define MODULUS_P 0 
#endif
#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif
#ifndef LOCAL_SIZE2
#define LOCAL_SIZE2 64
#endif
#ifndef LOCAL_SIZE3
#define LOCAL_SIZE3 128
#endif
#ifndef TRANSFORM_SIZE_N
#define TRANSFORM_SIZE_N 1
#endif

inline int get_digit_width(uint i) {
    uint j = i + 1;

    uint64_t pj  = (uint64_t)(MODULUS_P) * j;
    uint64_t pj1 = (uint64_t)(MODULUS_P) * i;

    uint64_t ceil1 = (pj - 1U) / (uint64_t)(TRANSFORM_SIZE_N);
    uint64_t ceil2 = (pj1 - 1U) / (uint64_t)(TRANSFORM_SIZE_N);

    return (int)(ceil1 - ceil2);
}


__kernel void kernel_sub2(__global ulong* restrict x)
{
    if (get_global_id(0) == 0) {
        uint c = 2U;
        while(c != 0U) {

            for(uint i = 0; i < TRANSFORM_SIZE_N; i++){
                const int d = get_digit_width(i);
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
                           __global ulong* restrict carry_array
                           )
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end = start + LOCAL_PROPAGATION_DEPTH; 
    ulong carry = 0UL; 
    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4)
    for (ulong i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);

        int4 digit_width_vec = (int4)(
            get_digit_width(i),
            get_digit_width(i + 1),
            get_digit_width(i + 2),
            get_digit_width(i + 3)
        );

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);
        vstore4(x_vec, 0, x + i);
    }
    
    if (carry != 0) {
        carry_array[gid] = carry;
    }
}



__kernel void kernel_carry_2(__global ulong* restrict x,
                             __global ulong* restrict carry_array) 
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

        int4 digit_width_vec = (int4)(
            get_digit_width(i),
            get_digit_width(i + 1),
            get_digit_width(i + 2),
            get_digit_width(i + 3)
        );
        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);

        vstore4(x_vec, 0, x + i);

        if (carry == 0) return;
    }

    if (carry != 0) {
        ulong4 x_vec = vload4(0, x + end);

        int4 digit_width_vec = (int4)(
            get_digit_width(end),
            get_digit_width(end + 1),
            get_digit_width(end + 2),
            get_digit_width(end + 3)
        );

        x_vec = digit_adc4_last(x_vec, digit_width_vec, &carry); 
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
                                           __global ulong* restrict wi,
                                            const uint m) {
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    
    const ulong a = x[base];
    const ulong b = x[base + m];
    const ulong c = x[base + 2 * m];
    const ulong d = x[base + 3 * m];
    const ulong4 coeff = (ulong4)(a, b, c, d);
    
    
    ulong4 u = modMul3_2(coeff, vload2(0, wi + twiddle_offset), wi[twiddle_offset + 2]);

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
                                                 __global ulong* restrict wi,
                                                 __global ulong* restrict digit_invweight,
                                                 const uint m)
{
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong base = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
    ulong4 coeff = (ulong4)(x[base], x[base + m], x[base + 2*m], x[base + 3*m]);
    ulong4 u = modMul3_2(coeff, vload2(0, wi + twiddle_offset), wi[twiddle_offset + 2]);

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
    
    ulong a = modAdd(coeff.s0, coeff.s2);
    ulong b = modAdd(coeff.s1, coeff.s3);
    ulong c = modSub(coeff.s0, coeff.s2);
    ulong d = modMuli(modSub(coeff.s1, coeff.s3));
    
    coeff.s0 = modAdd(a, b);
    coeff.s1 = modSub(a, b);
    coeff.s2 = modAdd(c, d);
    coeff.s3 = modSub(c, d);
    
    coeff = modMul3_2(coeff, vload2(0, w + 6), w[8 + 2]);

    coeff = modMul4(coeff, coeff); 
    
    vstore4(coeff, 0, x + 4 * k);
}



__kernel void kernel_ntt_radix4_mm_first(__global ulong* restrict x,
                                         __global ulong* restrict w,
                                         __global ulong* restrict digit_weight,
                                         const uint m)
{
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;

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

    c = modMul3_2(c, vload2(0, w + twiddle_offset), w[twiddle_offset + 2]);
    
    x[i0] = c.s0;
    x[i1] = c.s1;
    x[i2] = c.s2;
    x[i3] = c.s3;
}

__kernel void kernel_ntt_radix4_mm(__global ulong* restrict x,
                                   __global ulong* restrict w,
                                   const uint m)
{
    const ulong k = get_global_id(0);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    const ulong twiddle_offset = 6 * m + 3 * j;
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
    
    c = modMul3_2(c, vload2(0, w + twiddle_offset), w[twiddle_offset + 2]);
    
    x[i0] = c.s0;
    x[i1] = c.s1;
    x[i2] = c.s2;
    x[i3] = c.s3;
}


__kernel void kernel_inverse_ntt_radix4_m1(__global ulong* restrict x,
                                               __global ulong* restrict wi) {
    const ulong k = get_global_id(0);
    ulong4 coeff = vload4(0, x + 4 * k);

    ulong4 u = modMul3_2(coeff, vload2(0, wi + 6), wi[8]);
    
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
                                                 __global ulong* restrict wi,
                                                  __global ulong* restrict digit_invweight) {
    const ulong k = get_global_id(0);
    const ulong twiddle_offset = 6;
    ulong4 coeff = vload4(0, x + 4 * k);
    ulong4 div = vload4(0, digit_invweight + 4 * k);
    ulong4 u = modMul3_2(coeff, vload2(0, wi + twiddle_offset), wi[twiddle_offset + 2]);

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



#if WORKER_NTT_2_STEPS <= 0xFFFFFFFF
typedef uint gid_t;
#else
typedef ulong gid_t;
#endif


__kernel void kernel_ntt_radix4_inverse_mm_2steps(__global ulong* restrict x,
                                                  __global ulong* restrict wi,
                                                  const uint m) {
    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;
    int write_index = 0;
    const gid_t gid        = get_global_id(0);
    const gid_t group      = gid / m;
    const gid_t local_id   = gid % m;
    gid_t k_first          = group * m * 4 + local_id;
    const gid_t j          = local_id;

    gid_t base             = 4 * (k_first - j) + j;
    uint  tw_offset        = 6 * m + 3 * j;
    ulong tw1              = wi[tw_offset];
    ulong tw2              = wi[tw_offset + 1];
    ulong tw3              = wi[tw_offset + 2];
    ulong r, r2;

    #pragma unroll 4
    for (int pass = 0; pass < 4; pass++) {
        local_x[write_index    ] = x[base];
        local_x[write_index + 1] = modMul(x[base + m],           tw2);
        local_x[write_index + 2] = modMul(x[base + (m << 1)],    tw1);
        local_x[write_index + 3] = modMul(x[base + ((m << 1) + m)], tw3);

        r  = modAdd(local_x[write_index    ], local_x[write_index + 1]);
        r2 = modSub(local_x[write_index    ], local_x[write_index + 1]);
        local_x[write_index    ] = r;
        local_x[write_index + 1] = r2;

        r  = modAdd(local_x[write_index + 2], local_x[write_index + 3]);
        r2 = modMuli(modSub(local_x[write_index + 3], local_x[write_index + 2]));
        local_x[write_index + 2] = r;
        local_x[write_index + 3] = r2;

        r  = modAdd(local_x[write_index    ], local_x[write_index + 2]);
        r2 = modSub(local_x[write_index    ], local_x[write_index + 2]);
        local_x[write_index    ] = r;
        local_x[write_index + 2] = r2;
        r  = modAdd(local_x[write_index + 1], local_x[write_index + 3]);
        local_x[write_index + 3] = modSub(local_x[write_index + 1], local_x[write_index + 3]);
        local_x[write_index + 1] = r;

        write_index += 4;
        base        += 4 * m;
        k_first     += m;
    }

    const gid_t new_m     = m * 4;
    write_index          = 0;
    gid_t k_second       = group * m * 4 + local_id;

    #pragma unroll 4
    for (int pass = 0; pass < 4; pass++) {
        gid_t j2            = k_second & (new_m - 1);
        const gid_t base2   = 4 * (k_second - j2) + j2;
        const gid_t tw_off2 = 6 * new_m + 3 * j2;
        tw1 = wi[tw_off2];
        tw2 = wi[tw_off2 + 1];
        tw3 = wi[tw_off2 + 2];

        local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4] = modMul(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4], tw2);
        local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4] = modMul(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], tw1);
        local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4] = modMul(local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4], tw3);

        r  = modAdd(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4],
                    local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4]);
        r2 = modSub(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4],
                    local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4]);
        local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4] = r;
        local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4] = r2;

        r  = modAdd(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4],
                    local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4]);
        r2 = modMuli(modSub(local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4],
                            local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4]));

        x[base2]                            = modAdd(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4], r);
        x[base2 + (new_m << 1)]            = modSub(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4], r);
        x[base2 + new_m]                    = modAdd(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4], r2);
        x[base2 + ((new_m << 1) + new_m)]  = modSub(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4], r2);

        write_index += 4;
        k_second    += m;
    }
}


__kernel void kernel_ntt_radix4_mm_2steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          const uint m) {

    const gid_t gid = get_global_id(0);
    const gid_t group = gid / (m / 4);
    const gid_t local_id = gid % (m / 4);
    gid_t k_first = group * m + local_id;

    //ulong local_x[16];
    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;
    int write_index = 0;
    ulong twiddle1;
    ulong twiddle2;
    ulong twiddle3;

    #pragma unroll 4
    for (uint pass = 0; pass < 4; pass++) {
        const gid_t j = k_first & (m - 1);
        const gid_t i = 4 * (k_first - j) + j;
        const gid_t twiddle_offset = 6 * m + 3 * j;

        twiddle1 = w[twiddle_offset];
        twiddle2 = w[twiddle_offset + 1];
        twiddle3 = w[twiddle_offset + 2];

        local_x[write_index]    = x[i];
        local_x[write_index+1]  = x[i + m];
        local_x[write_index+2]  = x[i + (m << 1)];
        local_x[write_index+3]  = x[i + ((m << 1) + m)];

        ulong r = modAdd(local_x[write_index], local_x[write_index+2]);
        ulong r2  = modSub(local_x[write_index], local_x[write_index+2]);
        local_x[write_index] = r;
        local_x[write_index + 2] = r2;


        r = modAdd(local_x[write_index+1], local_x[write_index+3]);
        r2 = modMuli(modSub(local_x[write_index+1], local_x[write_index+3]));
        local_x[write_index + 1] = r;
        local_x[write_index+3] = r2;

        r                    =   modAdd(local_x[write_index], local_x[write_index + 1]);
        r2                   =   modSub(local_x[write_index], local_x[write_index + 1]);
        local_x[write_index] = r;
        local_x[write_index + 1] = r2;
        r                          =   modAdd(local_x[write_index + 2], local_x[write_index+3]);
        r2  =   modSub(local_x[write_index + 2], local_x[write_index+3]);
        local_x[write_index + 2] = r;
        local_x[write_index + 3]  = r2;
        
        local_x[write_index+1] = modMul(local_x[write_index+1], twiddle2);
        local_x[write_index+2] = modMul(local_x[write_index+2], twiddle1);
        local_x[write_index+3] = modMul(local_x[write_index+3], twiddle3);
        write_index += 4;
        k_first += m / 4;
    }

    const gid_t new_m = m / 4;
    write_index = 0;

    gid_t k_second = group * m + local_id;
    
    
    const gid_t j = local_id;
    const gid_t twiddle_offset = 6 * new_m + 3 * j;
    gid_t i = 4 * (k_second - j) + j;
    twiddle1 = w[twiddle_offset];
    twiddle2 = w[twiddle_offset + 1];
    twiddle3 = w[twiddle_offset + 2];

    #pragma unroll 4
    for (uint pass = 0; pass < 4; pass++) {
        ulong r = modAdd(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4]);
        ulong r2  = modSub(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4]);
        local_x[((write_index) % 4) * 4 + (write_index) / 4] = r;
        local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4]= r2;


        r = modAdd(local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);
        r2 = modMuli(modSub(local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]));
        local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4] = r;
        local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4] = r2;


        x[i]                    =   modAdd(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]);
        r2                   =   modSub(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]);
        local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]= r2;
        
        r                    =   modAdd(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);
        r2                   =   modSub(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);

        x[i + new_m] = modMul(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4],twiddle2);
        x[i + (new_m << 1)] = modMul(r,twiddle1);
        x[i + ((new_m << 1) + new_m)] = modMul(r2,twiddle3);
        write_index += 4;
        i += m;
    }

}

__kernel void kernel_ntt_radix4_mm_3steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          uint m) {
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
                
            ulong4 coeff = (ulong4)( x[i + 0 * m], x[i + 1 * m], x[i + 2 * m], x[i + 3 * m] );
            ulong v0 = modAdd(coeff.s0, coeff.s2);
            ulong v1 = modAdd(coeff.s1, coeff.s3);
            ulong v2 = modSub(coeff.s0, coeff.s2);
            ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));
            coeff.s0 = modAdd(v0, v1);
            coeff.s1 = modSub(v0, v1);
            coeff.s2 = modAdd(v2, v3);
            coeff.s3 = modSub(v2, v3);
            ulong4 result = modMul3_2(coeff, vload2(0, w + twiddle_offset), w[twiddle_offset + 2]);

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
            //const ulong i = 4 * (k - j) + j;
            const ulong twiddle_offset = 6 * m + 3 * j;
            
            ulong4 coeff = (ulong4)( local_x[indice1[write_index]],
                                    local_x[indice1[write_index + 1]],
                                    local_x[indice1[write_index + 2]],
                                    local_x[indice1[write_index + 3]] );
            ulong v0 = modAdd(coeff.s0, coeff.s2);
            ulong v1 = modAdd(coeff.s1, coeff.s3);
            ulong v2 = modSub(coeff.s0, coeff.s2);
            ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));
            coeff.s0 = modAdd(v0, v1);
            coeff.s1 = modSub(v0, v1);
            coeff.s2 = modAdd(v2, v3);
            coeff.s3 = modSub(v2, v3);
            ulong4 result = modMul3_2(coeff, vload2(0, w + twiddle_offset), w[twiddle_offset + 2]);

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
            
            ulong4 coeff = (ulong4)( local_x[indice2[write_index]],
                                    local_x[indice2[write_index + 1]],
                                    local_x[indice2[write_index + 2]],
                                    local_x[indice2[write_index + 3]] );
            ulong v0 = modAdd(coeff.s0, coeff.s2);
            ulong v1 = modAdd(coeff.s1, coeff.s3);
            ulong v2 = modSub(coeff.s0, coeff.s2);
            ulong v3 = modMuli(modSub(coeff.s1, coeff.s3));

            coeff.s0 = modAdd(v0, v1);
            coeff.s1 = modSub(v0, v1);
            coeff.s2 = modAdd(v2, v3);
            coeff.s3 = modSub(v2, v3);

            ulong4 result = modMul3_2(coeff, vload2(0, w + twiddle_offset), w[twiddle_offset + 2]);
            
            x[i + 0 * m] = result.s0;
            x[i + 1 * m] = result.s1;
            x[i + 2 * m] = result.s2;
            x[i + 3 * m] = result.s3;

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


__kernel void kernel_ntt_radix2(__global ulong* restrict x)
{
    const uint idx = get_global_id(0) << 1; // équivalent à get_global_id(0) * 2

    ulong2 u = vload2(0, x + idx);

    ulong s = modAdd(u.x, u.y);
    ulong d = modSub(u.x, u.y);

    vstore2((ulong2)(s, d), 0, x + idx);
}

__kernel void kernel_ntt_radix4_radix2_square_radix2_radix4(
    __global ulong* restrict x,
    __global ulong* restrict w,
    __global ulong* restrict wi)
{
    //const int m = 2;
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint global_base_idx = gid * 8;
    uint local_base_idx  = lid * 8;
    
    __local ulong localX[LOCAL_SIZE3 * 8];

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        localX[local_base_idx + i] = x[global_base_idx + i];
    }

    ulong2 _w12_01    = vload2(0, w  + 12);
    ulong  _w12_2     = w[14];
    ulong4 twiddles_w12  = (ulong4)(_w12_01.s0, _w12_01.s1, _w12_2, 0);

    ulong2 _w15_01    = vload2(0, w  + 15);
    ulong  _w15_2     = w[17];
    ulong4 twiddles_w15  = (ulong4)(_w15_01.s0, _w15_01.s1, _w15_2, 0);

    ulong2 _wi12_01   = vload2(0, wi + 12);
    ulong  _wi12_2    = wi[14];
    ulong4 twiddles_wi12 = (ulong4)(_wi12_01.s0, _wi12_01.s1, _wi12_2, 0);

    ulong2 _wi15_01   = vload2(0, wi + 15);
    ulong  _wi15_2    = wi[17];
    ulong4 twiddles_wi15 = (ulong4)(_wi15_01.s0, _wi15_01.s1, _wi15_2, 0);
        
    {
        ulong a = modAdd(localX[local_base_idx + 0], localX[local_base_idx + 4]);
        ulong b = modAdd(localX[local_base_idx + 2], localX[local_base_idx + 6]);
        ulong d = modSub(localX[local_base_idx + 0], localX[local_base_idx + 4]);
        ulong e = modMuli(modSub(localX[local_base_idx + 2], localX[local_base_idx + 6]));

        localX[local_base_idx + 0] = modAdd(a, b);
        localX[local_base_idx + 2] = modMul(modSub(a, b), twiddles_w12.s1);
        localX[local_base_idx + 4] = modMul(modAdd(d, e), twiddles_w12.s0);
        localX[local_base_idx + 6] = modMul(modSub(d, e), twiddles_w12.s2);
    }

    {
        ulong a = modAdd(localX[local_base_idx + 1], localX[local_base_idx + 5]);
        ulong b = modAdd(localX[local_base_idx + 3], localX[local_base_idx + 7]);
        ulong d = modSub(localX[local_base_idx + 1], localX[local_base_idx + 5]);
        ulong e = modMuli(modSub(localX[local_base_idx + 3], localX[local_base_idx + 7]));

        localX[local_base_idx + 1] = modAdd(a, b);
        localX[local_base_idx + 3] = modMul(modSub(a, b), twiddles_w15.s1);
        localX[local_base_idx + 5] = modMul(modAdd(d, e), twiddles_w15.s0);
        localX[local_base_idx + 7] = modMul(modSub(d, e), twiddles_w15.s2);
    }

    {
        #pragma unroll 4
        for (int i = 0; i < 8; i += 2) {
            ulong s = modAdd(localX[local_base_idx + i], localX[local_base_idx + i + 1]);
            ulong d = modSub(localX[local_base_idx + i], localX[local_base_idx + i + 1]);
            s = modMul(s, s);
            d = modMul(d, d);
            localX[local_base_idx + i]     = modAdd(s, d);
            localX[local_base_idx + i + 1] = modSub(s, d);
        }
    }

    {
        ulong4 coeff = (ulong4)(localX[local_base_idx + 0], localX[local_base_idx + 2], localX[local_base_idx + 4], localX[local_base_idx + 6]);
        coeff = modMul3(coeff,twiddles_wi12.s1, twiddles_wi12.s0, twiddles_wi12.s2);
        
        ulong4 r = butterfly(coeff);
        localX[local_base_idx + 0] = modAdd(r.s0, r.s2);
        localX[local_base_idx + 2] = modAdd(r.s1, r.s3);
        localX[local_base_idx + 4] = modSub(r.s0, r.s2);
        localX[local_base_idx + 6] = modSub(r.s1, r.s3);
    }

    {
        ulong4 coeff = (ulong4)(localX[local_base_idx + 1], localX[local_base_idx + 3], localX[local_base_idx + 5], localX[local_base_idx + 7]);
        coeff = modMul3(coeff,twiddles_wi15.s1, twiddles_wi15.s0, twiddles_wi15.s2);
        
        ulong4 r = butterfly(coeff);
        localX[local_base_idx + 1] = modAdd(r.s0, r.s2);
        localX[local_base_idx + 3] = modAdd(r.s1, r.s3);
        localX[local_base_idx + 5] = modSub(r.s0, r.s2);
        localX[local_base_idx + 7] = modSub(r.s1, r.s3);
    }

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        x[global_base_idx + i] = localX[local_base_idx + i];
    }
}

__kernel void kernel_pointwise_mul(__global ulong* a,
                                   __global const ulong* b) {
  size_t i = get_global_id(0);
  a[i] = modMul(a[i], b[i]);
}


__kernel void kernel_carry_mul_base(
    __global ulong* restrict x,
    __global ulong* restrict carry_array,
    const ulong        base        
) {
    const ulong gid   = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end   = start + LOCAL_PROPAGATION_DEPTH;
    ulong carry = 0UL;

    for (ulong i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);
        int4 digit_width_vec = (int4)(
            get_digit_width(i),
            get_digit_width(i + 1),
            get_digit_width(i + 2),
            get_digit_width(i + 3)
        );

        ulong4 b4 = (ulong4)(base, base, base, base);
        x_vec = x_vec * b4;

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);

        vstore4(x_vec, 0, x + i);
    }
    carry_array[gid] = carry;
}
