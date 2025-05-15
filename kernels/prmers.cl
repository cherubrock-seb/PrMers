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
    ulong d01 = digit_width.s0 ^ digit_width.s1;
    ulong d12 = digit_width.s1 ^ digit_width.s2;
    ulong d23 = digit_width.s2 ^ digit_width.s3;
    ulong d02 = digit_width.s0 ^ digit_width.s2;
    ulong d13 = digit_width.s1 ^ digit_width.s3;

    bool all4  = !(d01 | d12 | d23);     // s0==s1==s2==s3
    bool eq3   = !(d01 | d12);           // s0==s1==s2
    bool pair01_23 = !d01 && !d23;       // s0==s1 && s2==s3
    bool cross = !d02 && !d13;           // s0==s2 && s1==s3

    if (all4) {
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
    else if (eq3) {

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

        mask = (1UL << (digit_width.s3)) - 1UL;
        s = lhs.s3 + c;
        res.s3 = s & mask;
        c = s >> (digit_width.s3);

        *carry = c;
    }
     else if (pair01_23) {

        ulong mask01 = (1UL << (digit_width.s0)) - 1UL;
        ulong s = lhs.s0 + c;
        res.s0 = s & mask01;
        c = s >> digit_width.s0;
        s = lhs.s1 + c;
        res.s1 = s & mask01;
        c = s >> digit_width.s0;
        mask01 = (1UL << (digit_width.s2)) - 1UL;
        s = lhs.s2 + c;
        res.s2 = s & mask01;
        c = s >> digit_width.s2;
        s = lhs.s3 + c;
        res.s3 = s & mask01;
        c = s >> digit_width.s2;

        *carry = c;
    }
    else if (cross) {

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
#define TRANSFORM_SIZE_N 8
#endif

#define TRANSFORM_SIZE_N_DIV4  (TRANSFORM_SIZE_N / 4)
#define TRANSFORM_SIZE_N_DIV8  (TRANSFORM_SIZE_N / 8)

inline int get_digit_width(uint i) {
    uint j = i + 1;

    uint64_t pj  = (uint64_t)(MODULUS_P) * j;
    uint64_t pj1 = (uint64_t)(MODULUS_P) * i;

    uint64_t ceil1 = (pj - 1U) / (uint64_t)(TRANSFORM_SIZE_N);
    uint64_t ceil2 = (pj1 - 1U) / (uint64_t)(TRANSFORM_SIZE_N);

    return (int)(ceil1 - ceil2);
}


inline int4 get_digit_width4(uint i){
    uint64_t P = (uint64_t)MODULUS_P, N = (uint64_t)TRANSFORM_SIZE_N;
    uint64_t u = (uint64_t)i * P - 1, v;
    int4 r;
    v = u + P;    r.s0 = (int)(v/N - u/N);
    u = v;        v = u + P;    r.s1 = (int)(v/N - u/N);
    u = v;        v = u + P;    r.s2 = (int)(v/N - u/N);
    u = v;        v = u + P;    r.s3 = (int)(v/N - u/N);
    return r;
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


#ifndef DIGIT_WIDTH_VALUE_1
#define DIGIT_WIDTH_VALUE_1 1
#endif

#ifndef DIGIT_WIDTH_VALUE_2
#define DIGIT_WIDTH_VALUE_2 1
#endif
__kernel void kernel_carry(
    __global ulong*        restrict x,
    __global ulong*        restrict carry_array,
    __global const ulong4* restrict digitWidthMaskPacked
) {
    const uint gid   = get_global_id(0);
    const uint start = gid * LOCAL_PROPAGATION_DEPTH;
    const uint end   = start + LOCAL_PROPAGATION_DEPTH;
    ulong carry = 0UL;

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4)
    for (uint i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);

        uint blk    = i >> 6;      // i/64
        uint group4 = blk >> 2;    // (i/64)/4

        ulong4 chunk4 = digitWidthMaskPacked[group4];

        ulong  chunk = chunk4[ blk & 3 ];

        uchar4 m = (uchar4)(
            (chunk >> ((i & 63) + 0)) & 1,
            (chunk >> ((i & 63) + 1)) & 1,
            (chunk >> ((i & 63) + 2)) & 1,
            (chunk >> ((i & 63) + 3)) & 1
        );

        int4 digit_width_vec = (int4)(
            m.s0 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
            m.s1 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
            m.s2 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
            m.s3 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1
        );

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);

        vstore4(x_vec, 0, x + i);
    }

    if (carry != 0) {
        carry_array[gid] = carry;
    }
}

#define CARRY_WORKER_MIN_1 (CARRY_WORKER - 1)
__kernel void kernel_carry_2(__global ulong* restrict x,
                             __global ulong* restrict carry_array,
                             __global const ulong4* restrict digitWidthMaskPacked)
{
    const uint gid = get_global_id(0);
    const uint prev_gid = (gid == 0) ? (CARRY_WORKER_MIN_1) : (gid - 1);
    ulong carry = carry_array[prev_gid];
    if (carry == 0) return;
    const uint start = gid * LOCAL_PROPAGATION_DEPTH;
    const uint end = start + LOCAL_PROPAGATION_DEPTH - 4;  

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4_MIN)
    for (uint i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);
        uint blk    = i >> 6;      // i/64
        uint group4 = blk >> 2;    // (i/64)/4

        ulong4 chunk4 = digitWidthMaskPacked[group4];

        ulong  chunk = chunk4[ blk & 3 ];

        uchar4 m = (uchar4)(
            (chunk >> ((i & 63) + 0)) & 1,
            (chunk >> ((i & 63) + 1)) & 1,
            (chunk >> ((i & 63) + 2)) & 1,
            (chunk >> ((i & 63) + 3)) & 1
        );

        int4 digit_width_vec = (int4)(
            m.s0 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
            m.s1 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
            m.s2 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
            m.s3 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1
        );
        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);

        vstore4(x_vec, 0, x + i);

        if (carry == 0) return;
    }


    ulong4 x_vec = vload4(0, x + end);
    uint blk    = end >> 6;      // end/64
    uint group4 = blk >> 2;    // (end/64)/4

    ulong4 chunk4 = digitWidthMaskPacked[group4];

    ulong  chunk = chunk4[ blk & 3 ];

    uchar4 m = (uchar4)(
        (chunk >> ((end & 63) + 0)) & 1,
        (chunk >> ((end & 63) + 1)) & 1,
        (chunk >> ((end & 63) + 2)) & 1,
        (chunk >> ((end & 63) + 3)) & 1
    );

    int4 digit_width_vec = (int4)(
        m.s0 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
        m.s1 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
        m.s2 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1,
        m.s3 ? DIGIT_WIDTH_VALUE_2 : DIGIT_WIDTH_VALUE_1
    );
    x_vec = digit_adc4_last(x_vec, digit_width_vec, &carry); 
    vstore4(x_vec, 0, x + end);

}

static inline ulong4 butterfly(const ulong4 u) {
    return (ulong4)(
        modAdd(u.s0, u.s1),
        modSub(u.s0, u.s1),
        modAdd(u.s2, u.s3),
        modMuli(modSub(u.s3, u.s2))
    );
}

__kernel void kernel_inverse_ntt_radix4_mm(__global ulong2* restrict x,
                                           __global ulong2* restrict wi,
                                            const uint m) {
    uint k = get_global_id(0)*2;
    const uint j = k & (m - 1);
    const uint i = 4 * (k - j) + j;
    const ulong twiddle_offset = (6 * m + 3 * j) >> 1;
    

    const ulong2 twi1_a  = wi[twiddle_offset    ];
    ulong2 twi1_bb = wi[twiddle_offset + 1];
    const ulong2 twi1_cc = wi[twiddle_offset + 2];
    ulong twi2_a = twi1_bb.s0;
    twi1_bb = (ulong2)(twi1_bb.s1,twi1_cc.s0);

    k = m/2;
    const uint i0 = i >> 1;
    const uint i1 = i0 + k;
    const uint i2 = i1 + k;
    k = i2 + k;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);
        
    c = modMul3_2(c, twi1_a, twi2_a);
    
    v0.s0 = modAdd(c.s0, c.s1);
    v0.s1  = modSub(c.s0, c.s1);
    v1.s0 = modAdd(c.s2, c.s3);
    v1.s1 = modMuli(modSub(c.s3, c.s2));
    c.s0 = modAdd(v0.s0, v1.s0);
    c.s1 = modAdd(v0.s1, v1.s1);
    c.s2 = modSub(v0.s0, v1.s0);
    c.s3 = modSub(v0.s1, v1.s1);
    
    c2 = modMul3_2(c2, twi1_bb, twi1_cc.s1);
    
    v0.s0 = modAdd(c2.s0, c2.s1);
    v0.s1  = modSub(c2.s0, c2.s1);
    v1.s0 = modAdd(c2.s2, c2.s3);
    v1.s1 = modMuli(modSub(c2.s3, c2.s2));
    c2.s0 = modAdd(v0.s0, v1.s0);
    c2.s1 = modAdd(v0.s1, v1.s1);
    c2.s2 = modSub(v0.s0, v1.s0);
    c2.s3 = modSub(v0.s1, v1.s1);

    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k] = (ulong2)(c.s3, c2.s3);    
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



__kernel void kernel_inverse_ntt_radix4_mm_last(__global ulong2* restrict x,
                                                 __global ulong2* restrict wi,
                                                 __global ulong* restrict digit_invweight,
                                                 const uint m)
{
    uint k = get_global_id(0)*2;
    const uint j = k & (m - 1);
    const uint i = 4 * (k - j) + j;
    const ulong twiddle_offset = (6 * m + 3 * j) >> 1;
    

    const ulong2 twi1_a  = wi[twiddle_offset    ];
    ulong2 twi1_bb = wi[twiddle_offset + 1];
    const ulong2 twi1_cc = wi[twiddle_offset + 2];
    ulong twi2_a = twi1_bb.s0;
    twi1_bb = (ulong2)(twi1_bb.s1,twi1_cc.s0);

    k = m/2;
    const uint i0 = i >> 1;
    const uint i1 = i0 + k;
    const uint i2 = i1 + k;
    k = i2 + k;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);
        
    c = modMul3_2(c, twi1_a, twi2_a);
    
    v0.s0 = modAdd(c.s0, c.s1);
    v0.s1  = modSub(c.s0, c.s1);
    v1.s0 = modAdd(c.s2, c.s3);
    v1.s1 = modMuli(modSub(c.s3, c.s2));
    c.s0 = modAdd(v0.s0, v1.s0);
    c.s1 = modAdd(v0.s1, v1.s1);
    c.s2 = modSub(v0.s0, v1.s0);
    c.s3 = modSub(v0.s1, v1.s1);
    
    c2 = modMul3_2(c2, twi1_bb, twi1_cc.s1);
    
    v0.s0 = modAdd(c2.s0, c2.s1);
    v0.s1  = modSub(c2.s0, c2.s1);
    v1.s0 = modAdd(c2.s2, c2.s3);
    v1.s1 = modMuli(modSub(c2.s3, c2.s2));
    c2.s0 = modAdd(v0.s0, v1.s0);
    c2.s1 = modAdd(v0.s1, v1.s1);
    c2.s2 = modSub(v0.s0, v1.s0);
    c2.s3 = modSub(v0.s1, v1.s1);
                     
    c = modMul4(c, (ulong4)(digit_invweight[i],
                                 digit_invweight[i + m],
                                 digit_invweight[i + 2*m],
                                 digit_invweight[i + 3*m]));
    c2 = modMul4(c2, (ulong4)(digit_invweight[i+1],
                                 digit_invweight[i+1 + m],
                                 digit_invweight[i+1 + 2*m],
                                 digit_invweight[i+1 + 3*m]));
    
    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k] = (ulong2)(c.s3, c2.s3);
}


#ifndef W6
  #define W6 0
#endif
#ifndef W7
  #define W7 0
#endif
#ifndef W10
  #define W10 0
#endif



__kernel void kernel_ntt_radix4_last_m1(__global ulong4* restrict x)
{
    const uint k = get_global_id(0);
    ulong4 coeff = x[k];
    
    ulong a = modAdd(coeff.s0, coeff.s2);
    ulong b = modAdd(coeff.s1, coeff.s3);
    ulong c = modSub(coeff.s0, coeff.s2);
    ulong d = modMuli(modSub(coeff.s1, coeff.s3));
    
    coeff.s0 = modAdd(a, b);
    coeff.s1 = modSub(a, b);
    coeff.s2 = modAdd(c, d);
    coeff.s3 = modSub(c, d);
    coeff = modMul3_2(coeff, (ulong2)(W6,W7), W10);
    x[k] = modMul4(coeff, coeff);
}



__kernel void kernel_ntt_radix4_mm_first(__global ulong2* restrict x,
                                         __global ulong2* restrict w,
                                         __global ulong2* restrict digit_weight)
{
    uint k = get_global_id(0)*2;
    const uint j = k & (TRANSFORM_SIZE_N_DIV4 - 1);
    const uint i = 4 * (k - j) + j;
    k = (6*TRANSFORM_SIZE_N_DIV4 + 3*j) >> 1;

    const ulong2 tw1_a  = w[k    ];
    ulong2 tw1_bb = w[k + 1];
    const ulong2 tw1_cc = w[k + 2];

    ulong tw2_a = tw1_bb.s0;
    tw1_bb = (ulong2)(tw1_bb.s1,tw1_cc.s0);


    const uint i0 = i >> 1;
    const uint i1 = i0 + TRANSFORM_SIZE_N_DIV8;
    const uint i2 = i1 + TRANSFORM_SIZE_N_DIV8;
    k = i2 + TRANSFORM_SIZE_N_DIV8;

    ulong2 v0 = modMul2(x[i0],digit_weight[i0]);
    ulong2 v1 = modMul2(x[i1],digit_weight[i1]);
    const ulong2 v2 = modMul2(x[i2],digit_weight[i2]);
    const ulong2 v3 = modMul2(x[k],digit_weight[k]);

    ulong4 c = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);

    v0.s0 = modAdd(c.s0, c.s2);
    v0.s1  = modAdd(c.s1, c.s3);
    v1.s0 = modSub(c.s0, c.s2);
    v1.s1 = modMuli(modSub(c.s1, c.s3));
    c.s0 = modAdd(v0.s0, v0.s1);
    c.s1 = modSub(v0.s0, v0.s1);
    c.s2 = modAdd(v1.s0, v1.s1);
    c.s3 = modSub(v1.s0, v1.s1);
    
    c = modMul3_2(c, tw1_a, tw2_a);
    

    v0.s0 = modAdd(c2.s0, c2.s2);
    v0.s1 = modAdd(c2.s1, c2.s3);
    v1.s0 = modSub(c2.s0, c2.s2);
    v1.s1 = modMuli(modSub(c2.s1, c2.s3));
    c2.s0 = modAdd(v0.s0, v0.s1);
    c2.s1 = modSub(v0.s0, v0.s1);
    c2.s2 = modAdd(v1.s0, v1.s1);
    c2.s3 = modSub(v1.s0, v1.s1);
    
    c2 = modMul3_2(c2, tw1_bb, tw1_cc.s1);

    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k] = (ulong2)(c.s3, c2.s3);
}


__kernel void kernel_ntt_radix4_mm_m4(__global ulong2* restrict x,
                                   __global ulong2* restrict w)
{
    uint k = get_global_id(0)*2;
    const uint j = k & (3);
    const uint i = 4 * (k - j) + j;
    k = (24 + 3*j) >> 1;

    const ulong2 tw1_a  = w[k    ];
    ulong2 tw1_bb = w[k + 1];
    const ulong2 tw1_cc = w[k + 2];

    ulong tw2_a = tw1_bb.s0;
    tw1_bb = (ulong2)(tw1_bb.s1,tw1_cc.s0);


   
    const uint i0 = i >> 1;
    const uint i1 = i0 + 2;
    const uint i2 = i1 + 2;
    k = i2 + 2;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);

    v0.s0 = modAdd(c.s0, c.s2);
    v0.s1  = modAdd(c.s1, c.s3);
    v1.s0 = modSub(c.s0, c.s2);
    v1.s1 = modMuli(modSub(c.s1, c.s3));
    c.s0 = modAdd(v0.s0, v0.s1);
    c.s1 = modSub(v0.s0, v0.s1);
    c.s2 = modAdd(v1.s0, v1.s1);
    c.s3 = modSub(v1.s0, v1.s1);
    
    c = modMul3_2(c, tw1_a, tw2_a);
    

    v0.s0 = modAdd(c2.s0, c2.s2);
    v0.s1 = modAdd(c2.s1, c2.s3);
    v1.s0 = modSub(c2.s0, c2.s2);
    v1.s1 = modMuli(modSub(c2.s1, c2.s3));
    c2.s0 = modAdd(v0.s0, v0.s1);
    c2.s1 = modSub(v0.s0, v0.s1);
    c2.s2 = modAdd(v1.s0, v1.s1);
    c2.s3 = modSub(v1.s0, v1.s1);
    
    c2 = modMul3_2(c2, tw1_bb, tw1_cc.s1);

    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k] = (ulong2)(c.s3, c2.s3);
}


__kernel void kernel_ntt_radix4_mm_m8(__global ulong2* restrict x,
                                   __global ulong2* restrict w)
{
    uint k = get_global_id(0)*2;
    const uint j = k & (7);
    const uint i = 4 * (k - j) + j;
    k = (48 + 3*j) >> 1;

    const ulong2 tw1_a  = w[k    ];
    ulong2 tw1_bb = w[k + 1];
    const ulong2 tw1_cc = w[k + 2];

    ulong tw2_a = tw1_bb.s0;
    tw1_bb = (ulong2)(tw1_bb.s1,tw1_cc.s0);


    
    const uint i0 = i >> 1;
    const uint i1 = i0 + 4;
    const uint i2 = i1 + 4;
    k = i2 + 4;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);

    v0.s0 = modAdd(c.s0, c.s2);
    v0.s1  = modAdd(c.s1, c.s3);
    v1.s0 = modSub(c.s0, c.s2);
    v1.s1 = modMuli(modSub(c.s1, c.s3));
    c.s0 = modAdd(v0.s0, v0.s1);
    c.s1 = modSub(v0.s0, v0.s1);
    c.s2 = modAdd(v1.s0, v1.s1);
    c.s3 = modSub(v1.s0, v1.s1);
    
    c = modMul3_2(c, tw1_a, tw2_a);
    

    v0.s0 = modAdd(c2.s0, c2.s2);
    v0.s1 = modAdd(c2.s1, c2.s3);
    v1.s0 = modSub(c2.s0, c2.s2);
    v1.s1 = modMuli(modSub(c2.s1, c2.s3));
    c2.s0 = modAdd(v0.s0, v0.s1);
    c2.s1 = modSub(v0.s0, v0.s1);
    c2.s2 = modAdd(v1.s0, v1.s1);
    c2.s3 = modSub(v1.s0, v1.s1);
    
    c2 = modMul3_2(c2, tw1_bb, tw1_cc.s1);

    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k] = (ulong2)(c.s3, c2.s3);
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
    uint k_first          = group * m * 4 + local_id;
    

    uint base             = 4 * (k_first - local_id) + local_id;
    const uint  tw_offset  = 6 * m + 3 * local_id;
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

    const uint new_m     = m * 4;
    write_index          = 0;
    k_first       = group * m * 4 + local_id;

    #pragma unroll 4
    for (int pass = 0; pass < 4; pass++) {
        const gid_t j2            = k_first & (new_m - 1);
        const gid_t base2   = 4 * (k_first - j2) + j2;
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
        k_first    += m;
    }
}


__kernel void kernel_ntt_radix4_mm_2steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          const uint m) {

    const gid_t gid = get_global_id(0);
    const gid_t group = gid / (m / 4);
    const gid_t local_id = gid % (m / 4);
    uint k_first = group * m + local_id;

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

    const uint new_m = m / 4;
    write_index = 0;
    const uint twiddle_offset = 6 * new_m + 3 * local_id;
    k_first = 4 * (group * m) + local_id;
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


        x[k_first]                    =   modAdd(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]);
        r2                   =   modSub(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]);
        local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]= r2;
        
        r                    =   modAdd(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);
        r2                   =   modSub(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);

        x[k_first + new_m] = modMul(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4],twiddle2);
        x[k_first + (new_m << 1)] = modMul(r,twiddle1);
        x[k_first + ((new_m << 1) + new_m)] = modMul(r2,twiddle3);
        write_index += 4;
        k_first += m;
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

__kernel void kernel_ntt_radix2_square_radix2(__global ulong2* restrict x)
{
    const uint gid = get_global_id(0);
    ulong2 u = x[gid];

    ulong s = modAdd(u.x, u.y);
    ulong d = modSub(u.x, u.y);

    s = modMul(s, s);
    d = modMul(d, d);


    x[gid] = (ulong2)(modAdd(s, d), modSub(s, d));
}


__kernel void kernel_ntt_radix2(__global ulong* restrict x)
{
    const uint idx = get_global_id(0) << 1; // équivalent à get_global_id(0) * 2

    ulong2 u = vload2(0, x + idx);

    ulong s = modAdd(u.x, u.y);
    ulong d = modSub(u.x, u.y);

    vstore2((ulong2)(s, d), 0, x + idx);
}

#define ELEMENTS_PER_WORKITEM 8
#define VECTORS_PER_WORKITEM 4
#ifndef W12_01_X
  #define W12_01_X 0
#endif
#ifndef W12_01_Y
  #define W12_01_Y 0
#endif

#ifndef W15_01_X
  #define W15_01_X 0
#endif
#ifndef W15_01_Y
  #define W15_01_Y 0
#endif

#ifndef W15_2_X
  #define W15_2_X 0
#endif
#ifndef W15_2_Y
  #define W15_2_Y 0
#endif

#ifndef WI12_01_X
  #define WI12_01_X 0
#endif
#ifndef WI12_01_Y
  #define WI12_01_Y 0
#endif

#ifndef WI15_01_X
  #define WI15_01_X 0
#endif
#ifndef WI15_01_Y
  #define WI15_01_Y 0
#endif

#ifndef WI15_2_X
  #define WI15_2_X 0
#endif
#ifndef WI15_2_Y
  #define WI15_2_Y 0
#endif

__kernel void kernel_ntt_radix4_radix2_square_radix2_radix4(
    __global ulong2* restrict x)
{
    //const int m = 2;
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint global_base_vec = gid * VECTORS_PER_WORKITEM;
    uint  local_base_vec = lid * VECTORS_PER_WORKITEM;
    
    __local ulong2 localX[LOCAL_SIZE3 * VECTORS_PER_WORKITEM];

    #pragma unroll
    for (int v = 0; v < VECTORS_PER_WORKITEM; ++v) {
        localX[local_base_vec + v] = x[global_base_vec + v];
    }


        
    ulong a = modAdd(localX[local_base_vec + 0].s0, localX[local_base_vec + 2].s0);
    ulong b = modAdd(localX[local_base_vec + 1].s0, localX[local_base_vec + 3].s0);
    ulong d = modSub(localX[local_base_vec + 0].s0, localX[local_base_vec + 2].s0);
    ulong e = modMuli(modSub(localX[local_base_vec + 1].s0, localX[local_base_vec + 3].s0));

    localX[local_base_vec + 0].s0 = modAdd(a, b);
    localX[local_base_vec + 1].s0 = modMul(modSub(a, b), W12_01_Y);
    localX[local_base_vec + 2].s0 = modMul(modAdd(d, e), W12_01_X);
    localX[local_base_vec + 3].s0 = modMul(modSub(d, e), W15_01_X);


    a = modAdd(localX[local_base_vec + 0].s1, localX[local_base_vec + 2].s1);
    b = modAdd(localX[local_base_vec + 1].s1, localX[local_base_vec + 3].s1);
    d = modSub(localX[local_base_vec + 0].s1, localX[local_base_vec + 2].s1);
    e = modMuli(modSub(localX[local_base_vec + 1].s1, localX[local_base_vec + 3].s1));


    localX[local_base_vec + 0].s1 = modAdd(a, b);
    localX[local_base_vec + 1].s1 = modMul(modSub(a, b), W15_2_X);
    localX[local_base_vec + 2].s1 = modMul(modAdd(d, e), W15_01_Y);
    localX[local_base_vec + 3].s1 = modMul(modSub(d, e), W15_2_Y);

    #pragma unroll 4
    for (int i = 0; i < 4; i += 1) {
        ulong s = modAdd(localX[local_base_vec + i].s0, localX[local_base_vec + i].s1);
        ulong d = modSub(localX[local_base_vec + i].s0, localX[local_base_vec + i].s1);
        s = modMul(s, s);
        d = modMul(d, d);
        localX[local_base_vec + i].s0     = modAdd(s, d);
        localX[local_base_vec + i].s1 = modSub(s, d);
    }
    

    ulong r =  modMul(localX[local_base_vec + 1].s0, WI12_01_Y); 
    ulong rs0 = modAdd(localX[local_base_vec + 0].s0, r);
    ulong rs1 = modSub(localX[local_base_vec + 0].s0, r);
    r =  modMul(localX[local_base_vec + 2].s0, WI12_01_X);
    ulong rr = modMul(localX[local_base_vec + 3].s0, WI15_01_X);
    ulong rs2 = modAdd(r, rr);
    ulong rs3 = modMuli(modSub(rr, r));

    localX[local_base_vec + 0].s0 = modAdd(rs0, rs2);
    localX[local_base_vec + 1].s0 = modAdd(rs1, rs3);
    localX[local_base_vec + 2].s0 = modSub(rs0, rs2);
    localX[local_base_vec + 3].s0 = modSub(rs1, rs3);



    r =  modMul(localX[local_base_vec + 1].s1, WI15_2_X); 
    rs0 = modAdd(localX[local_base_vec + 0].s1, r);
    rs1 = modSub(localX[local_base_vec + 0].s1, r);
    r =  modMul(localX[local_base_vec + 2].s1, WI15_01_Y);
    rr = modMul(localX[local_base_vec + 3].s1, WI15_2_Y);
    rs2 = modAdd(r, rr);
    rs3 = modMuli(modSub(rr, r));

    localX[local_base_vec + 0].s1 = modAdd(rs0, rs2);
    localX[local_base_vec + 1].s1 = modAdd(rs1, rs3);
    localX[local_base_vec + 2].s1 = modSub(rs0, rs2);
    localX[local_base_vec + 3].s1 = modSub(rs1, rs3);
    

    #pragma unroll
    for (int v = 0; v < VECTORS_PER_WORKITEM; ++v) {
        x[global_base_vec + v] = localX[local_base_vec + v];
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


static inline uint mod3(__global uint* W, uint wordCount) {
    uint r = 0;
    for (uint i = 0; i < wordCount; ++i)
        r = (r + (W[i] % 3)) % 3;
    return r;
}

static void doDiv3(uint E, __global uint* W, uint wordCount) {
    uint r = (3 - mod3(W, wordCount)) % 3;
    int topBits = E % 32;
    {
        ulong t = ((ulong)r << topBits) + (ulong)W[wordCount-1];
        W[wordCount-1] = (uint)(t / 3);
        r = (uint)(t % 3);
    }
    for (int i = wordCount - 2; i >= 0; --i) {
        ulong t = ((ulong)r << 32) + (ulong)W[i];
        W[i] = (uint)(t / 3);
        r    = (uint)(t % 3);
    }
}

static inline void doDiv9(uint E, __global uint* W, uint wordCount) {
    doDiv3(E, W, wordCount);
    doDiv3(E, W, wordCount);
}


static void compactBits(
    __global const ulong* x,
    uint                  digitCount,
    uint                  E,
    __global uint*        out,
    uint*                 outCount)
{
    int carry = 0;
    uint outWord = 0;
    int haveBits = 0;
    uint p = 0, o = 0;
    uint totalWords = (E - 1) / 32 + 1;

    for (uint i = 0; i < totalWords; ++i) out[i] = 0;

    for (p = 0; p < digitCount; ++p) {
        int w = get_digit_width(p);
        ulong v64 = (ulong)carry + x[p];
        carry = (int)(v64 >> w);
        uint v = (uint)(v64 & (((ulong)1 << w) - 1));

        int topBits = 32 - haveBits;
        outWord |= v << haveBits;
        if (w >= topBits) {
            out[o++] = outWord;
            outWord = (w > topBits) ? (v >> topBits) : 0;
            haveBits = w - topBits;
        } else {
            haveBits += w;
        }
    }
    if (haveBits > 0 || carry) {
        out[o++] = outWord;
        for (uint i = 1; carry && i < o; ++i) {
            ulong sum = (ulong)out[i] + (ulong)carry;
            out[i] = (uint)(sum & 0xFFFFFFFFUL);
            carry = (int)(sum >> 32);
        }
    }
    *outCount = o;
}




#pragma OPENCL EXTENSION cl_khr_printf : warn

#if defined(cl_khr_printf) || (defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 120)
  #define HAS_PRINTF 1
#else
  #define HAS_PRINTF 0
#endif



__kernel void kernel_res64_display(
    __global const ulong* x,             
    uint                  exponent,   
    uint                  digitCount,
    uint                  mode,
    uint                  iter,
    __global uint*        out
) {
#if HAS_PRINTF
    if (get_global_id(0) != 0) return;
    uint wordCount;
    compactBits(x, digitCount, exponent, out, &wordCount);
    printf("Iter: %u | Res64: %08X%08X\n",
           iter,        // itération courante
           out[1],      // haut 32 bits
           out[0]       // bas 32 bits
    );
#else
    (void)x; (void)exponent; (void)digitCount;
    (void)mode; (void)iter; (void)out;
#endif
}
