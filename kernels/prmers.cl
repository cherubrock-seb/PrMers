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
__constant ulong4 zero4 = (ulong4)(0);

// Modular addition for vectors of 4 ulong
inline ulong4 modAdd4(ulong4 lhs, ulong4 rhs) {
    // 'select' avoids branching (if-else) on GPU
    ulong4 c = select((ulong4)0, mod_p_comp4_const, lhs >= (mod_p4_const - rhs));
    return lhs + rhs + c;
}

// Modular subtraction for vectors of 4 ulong
inline ulong4 modSub4(const ulong4 lhs, const ulong4 rhs) {
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
    ulong c = select(0u, mod_p_comp_const, lo >= mod_p_const);
    c += lo;
    c = modAdd(c, hi << 32);                           // Add hi * 2^32
    c = modSub(c, (hi >> 32) + (uint)hi);              // Subtract hi_high + hi_low
    return c;
}

// Vectorized version of Reduce: reduces four (lo, hi) pairs simultaneously
inline ulong4 Reduce4(const ulong4 lo, const ulong4 hi) {
    ulong4 c = select(zero4, mod_p_comp4_const, lo >= mod_p4_const);
    c += lo;
    ulong4 hi_shifted = hi << 32;
    ulong4 hi_reduced = (hi >> 32) + convert_ulong4(convert_uint4(hi));
    c = modAdd4(c, hi_shifted);   // c devient r
    c = modSub4(c, hi_reduced);
    return c;
}


// Modular multiplication of 4 ulong vectors
inline ulong4 modMul4(const ulong4 lhs, const ulong4 rhs) {
    const ulong4 lo = lhs * rhs;
    const ulong4 hi = mul_hi(lhs, rhs);
    return Reduce4(lo, hi);
}



// Computes the full 128-bit product of two 64-bit integers
inline void mul128(const ulong a, const ulong b, __private ulong *restrict hi, __private ulong *restrict lo) {
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

inline ulong2 Reduce2(const ulong2 lo, const ulong2 hi) {
    ulong2 c = select((ulong2)0, mod_p_comp2_const, lo >= mod_p2_const);
    c += lo;

    ulong2 hi_shifted = hi << 32;
    ulong2 hi_reduced    = (hi >> 32) + convert_ulong2(convert_uint2(hi));
    

    c = modAdd2(c, hi_shifted);
    c = modSub2(c, hi_reduced);
    return c;
}


inline ulong2 modMul2(const ulong2 lhs, const ulong2 rhs) {
    ulong2 lo = lhs * rhs;
    ulong2 hi = mul_hi(lhs, rhs);
    return Reduce2(lo, hi);
}


inline ulong4 modMul3_2(const ulong4 lhs,
                        const ulong2 w02,
                        const ulong w3)
{
    ulong2 x = (ulong2)(lhs.s1, lhs.s2);
    x = modMul2(x, (ulong2)(w02.s1, w02.s0));
    ulong m = modMul(lhs.s3, w3);
    return (ulong4)(lhs.s0, x.s0, x.s1, m);
}

inline ulong modMulPminus1(const ulong x) {
    ulong r = mod_p_const - x;
    r += (r >= mod_p_const) ? mod_p_comp_const : 0;
    return r;
}
inline ulong4 modMul3_2_w10(const ulong4 lhs)
{
    ulong m = modMulPminus1(lhs.s3);
    return (ulong4)(lhs.s0, lhs.s1, lhs.s2, m);
}


inline ulong4 modMul3(const ulong4 lhs,
                      const ulong w1,
                      const ulong w2,
                      const ulong w3)
{
    return (ulong4)(
        lhs.s0,
        modMul(lhs.s1, w1),
        modMul(lhs.s2, w2),
        modMul(lhs.s3, w3)
    );
}


inline ulong Reduce48(const ulong lo, const ulong hi) {
    ulong r      = lo;
    ulong rhs    = hi << 32;
    r           += rhs + ((r >= mod_p_const - rhs) ? mod_p_comp_const : 0);
    ulong rhs2   = (hi >> 32) + (uint)hi;
    r           -= rhs2 + ((r < rhs2) ? mod_p_comp_const : 0);
    return r;
}

inline ulong modMuli(const ulong x) {
    return Reduce48(x << 48, x >> 16);
}


inline ulong Reduce24(const ulong lo, const ulong hi) {
    ulong r    = lo;
    ulong rhs  = hi << 32;  
    r           += rhs 
                   + ((r >= mod_p_const - rhs) ? mod_p_comp_const : 0);
    ulong rhs2 = (hi >> 32) + (uint32_t)hi;
    r           -= rhs2 
                   + ((r < rhs2) ? mod_p_comp_const : 0);
    return r;
}

inline ulong modMul2Pow24(const ulong x) {
    return Reduce24( x << 24,
                     x >> 40 );
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
    ulong c = *carry;
    #pragma unroll 3
    for (int i = 0; i < 3; i++) {
        ulong s = lhs[i] + c;                        
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
#ifndef LOCAL_SIZE5
#define LOCAL_SIZE5 64
#endif
#ifndef TRANSFORM_SIZE_N
#define TRANSFORM_SIZE_N 8
#endif

#define TRANSFORM_SIZE_N_DIV4  (TRANSFORM_SIZE_N / 4)
#define TRANSFORM_SIZE_N_DIV8  (TRANSFORM_SIZE_N / 8)

inline int get_digit_width(uint i) {
    uint j = i + 1;

    ulong pj  = (ulong)(MODULUS_P) * j;
    ulong pj1 = (ulong)(MODULUS_P) * i;

    ulong ceil1 = (pj - 1U) / (ulong)(TRANSFORM_SIZE_N);
    ulong ceil2 = (pj1 - 1U) / (ulong)(TRANSFORM_SIZE_N);

    return (int)(ceil1 - ceil2);
}


inline int4 get_digit_width4(uint i){
    ulong P = (ulong)MODULUS_P, N = (ulong)TRANSFORM_SIZE_N;
    ulong u = (ulong)i * P - 1, v;
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

__kernel void kernel_sub1(__global ulong* restrict x)
{
    if (get_global_id(0) == 0) {
        uint c = 1U;
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
    __global ulong*       restrict x,
    __global ulong*       restrict carry_array,
    __global const ulong* restrict maskPacked
){
    const uint gid   = get_global_id(0);
    const uint start = gid * LOCAL_PROPAGATION_DEPTH;
    const uint end   = start + LOCAL_PROPAGATION_DEPTH;

    const int4 DW1 = (int4)(DIGIT_WIDTH_VALUE_1);
    const int4 DW2 = (int4)(DIGIT_WIDTH_VALUE_2);

    ulong carry = 0UL;

    uint base = start >> 6;
    uint off  = start & 63;

    ulong lo = maskPacked[base + 0];
    ulong hi = maskPacked[base + 1];

    ulong sh     = (ulong)((64 - off) & 63);
    ulong nzmask = (ulong)0 - (ulong)(off != 0);
    ulong merged = (lo >> off) | ((hi << sh) & nzmask);

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4)
    for (uint i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);

        uint nib = (uint)(merged & 0xFULL);

        uint4 sel = (uint4)(
            (nib >> 0) & 1U,
            (nib >> 1) & 1U,
            (nib >> 2) & 1U,
            (nib >> 3) & 1U
        ) * (uint)0xFFFFFFFFU;

        const int4 digit_width_vec = bitselect(DW1, DW2, as_int4(sel));

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);
        vstore4(x_vec, 0, x + i);

        sh = (ulong)0 - (off == 60);
        nzmask = maskPacked[base + 2];

        merged = select(merged >> 4, hi, sh);
        lo     = select(lo, hi, sh);
        hi     = select(hi, nzmask, sh);

        off  = (off + 4) & 63;
        base += sh;
    }

    carry_array[gid] = carry;
}




#define CONST_SCALAR_MUL 3UL
__constant ulong4 CONST_SCALAR_VEC = (ulong4)(CONST_SCALAR_MUL, CONST_SCALAR_MUL, CONST_SCALAR_MUL, CONST_SCALAR_MUL);

__kernel void kernel_carry_mul_3(
    __global ulong*       restrict x,
    __global ulong*       restrict carry_array,
    __global const ulong* restrict maskPacked
) {
    const uint gid   = get_global_id(0);
    const uint start = gid * LOCAL_PROPAGATION_DEPTH;
    const uint end   = start + LOCAL_PROPAGATION_DEPTH;

    const int4 DW1 = (int4)(DIGIT_WIDTH_VALUE_1);
    const int4 DW2 = (int4)(DIGIT_WIDTH_VALUE_2);

    ulong carry1 = 0UL;
    ulong carry  = 0UL;

    uint base = start >> 6;
    uint off  = start & 63;

    ulong lo = maskPacked[base + 0];
    ulong hi = maskPacked[base + 1];

    ulong sh     = (ulong)((64 - off) & 63);
    ulong nzmask = (ulong)0 - (ulong)(off != 0);
    ulong merged = (lo >> off) | ((hi << sh) & nzmask);

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4)
    for (uint i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);

        uint nib = (uint)(merged & 0xFULL);
        int4 sel = (int4)(
            (nib >> 0) & 1,
            (nib >> 1) & 1,
            (nib >> 2) & 1,
            (nib >> 3) & 1
        ) * -1;

        sel = bitselect(DW1, DW2, as_int4(sel));

        x_vec = digit_adc4(x_vec, sel, &carry1);
        ulong4 lo_vec = x_vec * CONST_SCALAR_VEC;
        x_vec = digit_adc4(lo_vec, sel, &carry);
        carry = carry + 3UL * carry1;
        vstore4(x_vec, 0, x + i);

        uint wrap = (off + 4U) >> 6;
        ulong wrapMask = (ulong)0 - (ulong)wrap;
        uint off_next = (off + 4U) & 63U;

        ulong nextChunk = maskPacked[base + 2];

        ulong inject = ((ulong)0 - (ulong)(off == 0)) & (hi << 60);
        ulong cand0  = (merged >> 4) | inject;

        ulong lo_n = select(lo, hi, wrapMask);
        ulong hi_n = select(hi, nextChunk, wrapMask);

        sh     = (ulong)((64 - off_next) & 63);
        nzmask = (ulong)0 - (ulong)(off_next != 0);
        ulong cand1    = (lo_n >> off_next) | ((hi_n << sh) & nzmask);

        merged = select(cand0, cand1, wrapMask);

        off  = off_next;
        base += wrap;
        lo   = lo_n;
        hi   = hi_n;
    }

    carry_array[gid] = carry;
}

#define CARRY_WORKER_MIN_1 (CARRY_WORKER - 1)
__kernel void kernel_carry_2(__global ulong* restrict x,
                             __global ulong* restrict carry_array,
                             __global const ulong* restrict maskPacked)
{
    const uint gid = get_global_id(0);
    const uint prev_gid = (gid == 0) ? (CARRY_WORKER_MIN_1) : (gid - 1);
    ulong carry = carry_array[prev_gid];
    if (carry == 0) return;

    const uint start = gid * LOCAL_PROPAGATION_DEPTH;
    const uint end = start + LOCAL_PROPAGATION_DEPTH - 4;

    const int4 DW1 = (int4)(DIGIT_WIDTH_VALUE_1);
    const int4 DW2 = (int4)(DIGIT_WIDTH_VALUE_2);

    uint base = start >> 6;
    uint off  = start & 63;

    ulong lo = maskPacked[base + 0];
    ulong hi = maskPacked[base + 1];

    ulong sh     = (ulong)((64 - off) & 63);
    ulong nzmask = (ulong)0 - (ulong)(off != 0);
    ulong merged = (lo >> off) | ((hi << sh) & nzmask);

    PRAGMA_UNROLL(LOCAL_PROPAGATION_DEPTH_DIV4_MIN)
    for (uint i = start; i < end; i += 4) {
        ulong4 x_vec = vload4(0, x + i);

        uint nib = (uint)(merged & 0xFULL);
        uint4 sel = (uint4)(
            (nib >> 0) & 1U,
            (nib >> 1) & 1U,
            (nib >> 2) & 1U,
            (nib >> 3) & 1U
        ) * (uint)0xFFFFFFFFU;

        int4 digit_width_vec = bitselect(DW1, DW2, as_int4(sel));

        x_vec = digit_adc4(x_vec, digit_width_vec, &carry);
        vstore4(x_vec, 0, x + i);
        if (carry == 0) return;

        uint wrap = (off + 4U) >> 6;
        ulong wrapMask = (ulong)0 - (ulong)wrap;
        uint off_next = (off + 4U) & 63U;

        ulong nextChunk = maskPacked[base + 2];
        ulong inject = ((ulong)0 - (ulong)(off == 0)) & (hi << 60);
        ulong cand0  = (merged >> 4) | inject;

        ulong lo_n = select(lo, hi, wrapMask);
        ulong hi_n = select(hi, nextChunk, wrapMask);

        sh     = (ulong)((64 - off_next) & 63);
        nzmask = (ulong)0 - (ulong)(off_next != 0);
        ulong cand1    = (lo_n >> off_next) | ((hi_n << sh) & nzmask);

        merged = select(cand0, cand1, wrapMask);

        off  = off_next;
        base += wrap;
        lo   = lo_n;
        hi   = hi_n;
    }

    ulong4 x_vec = vload4(0, x + end);

    base = (uint)(merged & 0xFULL);
    int4 sel = (int4)(
        (base >> 0) & 1,
        (base >> 1) & 1,
        (base >> 2) & 1,
        (base >> 3) & 1
    ) * -1;


    sel = bitselect(DW1, DW2, (sel));

    x_vec = digit_adc4_last(x_vec, sel, &carry);
    vstore4(x_vec, 0, x + end);
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


__kernel void kernel_ntt_radix4_last_m1_n4_nosquare(__global ulong* restrict x,
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
    //r = modMul4(r, r);
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
    coeff = modMul3_2_w10(coeff);
    x[k] = modMul4(coeff, coeff);
}


__kernel void kernel_ntt_radix4_last_m1_nosquare(__global ulong4* restrict x)
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
    x[k] = modMul3_2_w10(coeff);
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
__kernel void kernel_ntt_radix4_mm_m2(__global ulong2* restrict x,
                                      __global ulong2* restrict w)
{
    uint k = get_global_id(0) * 2;
    const uint j = k & 1;
    const uint i = 4 * (k - j) + j;
    k = (12 + 3 * j) >> 1;

    const ulong2 tw1_a  = w[k];
    ulong2 tw1_bb = w[k + 1];
    const ulong2 tw1_cc = w[k + 2];

    ulong tw2_a = tw1_bb.s0;
    tw1_bb = (ulong2)(tw1_bb.s1, tw1_cc.s0);

    const uint i0 = i >> 1;
    const uint i1 = i0 + 1;
    const uint i2 = i1 + 1;
    k = i2 + 1;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c  = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);

    v0.s0 = modAdd(c.s0, c.s2);
    v0.s1 = modAdd(c.s1, c.s3);
    v1.s0 = modSub(c.s0, c.s2);
    v1.s1 = modMuli(modSub(c.s1, c.s3));
    c.s0  = modAdd(v0.s0, v0.s1);
    c.s1  = modSub(v0.s0, v0.s1);
    c.s2  = modAdd(v1.s0, v1.s1);
    c.s3  = modSub(v1.s0, v1.s1);

    c = modMul3_2(c, tw1_a, tw2_a);

    v0.s0 = modAdd(c2.s0, c2.s2);
    v0.s1 = modAdd(c2.s1, c2.s3);
    v1.s0 = modSub(c2.s0, c2.s2);
    v1.s1 = modMuli(modSub(c2.s1, c2.s3));
    c2.s0  = modAdd(v0.s0, v0.s1);
    c2.s1  = modSub(v0.s0, v0.s1);
    c2.s2  = modAdd(v1.s0, v1.s1);
    c2.s3  = modSub(v1.s0, v1.s1);

    c2 = modMul3_2(c2, tw1_bb, tw1_cc.s1);

    x[i0] = (ulong2)(c.s0,  c2.s0);
    x[i1] = (ulong2)(c.s1,  c2.s1);
    x[i2] = (ulong2)(c.s2,  c2.s2);
    x[k]  = (ulong2)(c.s3,  c2.s3);
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


__kernel void kernel_ntt_radix4_mm_m16(__global ulong2* restrict x,
                                       __global ulong2* restrict w)
{
    uint k = get_global_id(0) * 2;
    const uint j = k & 15;
    const uint i = 4 * (k - j) + j;
    k = (96 + 3 * j) >> 1;

    const ulong2 tw1_a  = w[k];
    ulong2 tw1_bb = w[k + 1];
    const ulong2 tw1_cc = w[k + 2];

    ulong tw2_a = tw1_bb.s0;
    tw1_bb = (ulong2)(tw1_bb.s1, tw1_cc.s0);

    const uint i0 = i >> 1;
    const uint i1 = i0 + 8;
    const uint i2 = i1 + 8;
    k = i2 + 8;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c  = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);

    v0.s0 = modAdd(c.s0, c.s2);
    v0.s1 = modAdd(c.s1, c.s3);
    v1.s0 = modSub(c.s0, c.s2);
    v1.s1 = modMuli(modSub(c.s1, c.s3));
    c.s0  = modAdd(v0.s0, v0.s1);
    c.s1  = modSub(v0.s0, v0.s1);
    c.s2  = modAdd(v1.s0, v1.s1);
    c.s3  = modSub(v1.s0, v1.s1);

    c = modMul3_2(c, tw1_a, tw2_a);

    v0.s0 = modAdd(c2.s0, c2.s2);
    v0.s1 = modAdd(c2.s1, c2.s3);
    v1.s0 = modSub(c2.s0, c2.s2);
    v1.s1 = modMuli(modSub(c2.s1, c2.s3));
    c2.s0  = modAdd(v0.s0, v0.s1);
    c2.s1  = modSub(v0.s0, v0.s1);
    c2.s2  = modAdd(v1.s0, v1.s1);
    c2.s3  = modSub(v1.s0, v1.s1);

    c2 = modMul3_2(c2, tw1_bb, tw1_cc.s1);

    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k]   = (ulong2)(c.s3, c2.s3);
}

__kernel void kernel_ntt_radix4_mm_m32(__global ulong2* restrict x,
                                       __global ulong2* restrict w)
{
    uint k = get_global_id(0) * 2;
    const uint j = k & 31;
    const uint i = 4 * (k - j) + j;
    k = (192 + 3 * j) >> 1;

    const ulong2 tw1_a  = w[k];
    ulong2 tw1_bb = w[k + 1];
    const ulong2 tw1_cc = w[k + 2];

    ulong tw2_a = tw1_bb.s0;
    tw1_bb = (ulong2)(tw1_bb.s1, tw1_cc.s0);

    const uint i0 = i >> 1;
    const uint i1 = i0 + 16;
    const uint i2 = i1 + 16;
    k = i2 + 16;

    ulong2 v0 = x[i0];
    ulong2 v1 = x[i1];
    const ulong2 v2 = x[i2];
    const ulong2 v3 = x[k];

    ulong4 c  = (ulong4)(v0.s0, v1.s0, v2.s0, v3.s0);
    ulong4 c2 = (ulong4)(v0.s1, v1.s1, v2.s1, v3.s1);

    v0.s0 = modAdd(c.s0, c.s2);
    v0.s1 = modAdd(c.s1, c.s3);
    v1.s0 = modSub(c.s0, c.s2);
    v1.s1 = modMuli(modSub(c.s1, c.s3));
    c.s0  = modAdd(v0.s0, v0.s1);
    c.s1  = modSub(v0.s0, v0.s1);
    c.s2  = modAdd(v1.s0, v1.s1);
    c.s3  = modSub(v1.s0, v1.s1);

    c = modMul3_2(c, tw1_a, tw2_a);

    v0.s0 = modAdd(c2.s0, c2.s2);
    v0.s1 = modAdd(c2.s1, c2.s3);
    v1.s0 = modSub(c2.s0, c2.s2);
    v1.s1 = modMuli(modSub(c2.s1, c2.s3));
    c2.s0  = modAdd(v0.s0, v0.s1);
    c2.s1  = modSub(v0.s0, v0.s1);
    c2.s2  = modAdd(v1.s0, v1.s1);
    c2.s3  = modSub(v1.s0, v1.s1);

    c2 = modMul3_2(c2, tw1_bb, tw1_cc.s1);

    x[i0] = (ulong2)(c.s0, c2.s0);
    x[i1] = (ulong2)(c.s1, c2.s1);
    x[i2] = (ulong2)(c.s2, c2.s2);
    x[k]   = (ulong2)(c.s3, c2.s3);
}


#ifndef WI6
  #define WI6 0
#endif
#ifndef WI7
  #define WI7 0
#endif
#ifndef WI8
  #define WI8 0
#endif

__kernel void kernel_inverse_ntt_radix4_m1(__global ulong4* restrict x) {
    const ulong k = get_global_id(0);
    ulong4 coeff = x[k];

    ulong4 u = modMul3_2(coeff,(ulong2)(WI6,WI7), WI8);
    
    ulong v0 = modAdd(u.s0, u.s1);
    ulong v1 = modSub(u.s0, u.s1);
    ulong v2 = modAdd(u.s2, u.s3);
    ulong v3 = modMuli(modSub(u.s3, u.s2));
    ulong4 result = (ulong4)( modAdd(v0, v2),
                              modAdd(v1, v3),
                              modSub(v0, v2),
                              modSub(v1, v3) );
    x[k] = result;
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
    const gid_t gid      = get_global_id(0);
    const gid_t group    = gid / m;
    const gid_t local_id = gid % m;
    uint k_first         = group * m * 4 + local_id;

    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;

    uint write_index = 0;
    uint base        = 4 * (k_first - local_id) + local_id;
    const uint tw_offset = 6 * m + 3 * local_id;
    ulong2 tw12     = vload2(0, wi + tw_offset);
    ulong tw3       = wi[tw_offset + 2];
    ulong r, r2;

    #pragma unroll 4
    for (uint pass = 0; pass < 4; ++pass) {
        ulong a0 = x[base];
        ulong a1 = modMul(x[base + m],           tw12.s1);
        ulong a2 = modMul(x[base + (m << 1)],    tw12.s0);
        ulong a3 = modMul(x[base + ((m << 1) + m)], tw3);

        r  = modAdd(a0, a1);
        r2 = modSub(a0, a1);
        a0 = r; a1 = r2;

        r  = modAdd(a2, a3);
        r2 = modMuli(modSub(a3, a2));
        a2 = r; a3 = r2;

        r  = modAdd(a0, a2);
        r2 = modSub(a0, a2);
        a0 = r; a2 = r2;

        r  = modAdd(a1, a3);
        a3 = modSub(a1, a3);
        a1 = r;

        local_x[write_index    ] = a0;
        local_x[write_index + 1] = a1;
        local_x[write_index + 2] = a2;
        local_x[write_index + 3] = a3;

        write_index += 4;
        base        += 4 * m;
        k_first     += m;
    }

    const uint new_m = m * 4;
    write_index     = 0;
    k_first         = group * m * 4 + local_id;

    #pragma unroll 4
    for (uint pass = 0; pass < 4; ++pass) {
        const gid_t j2       = k_first & (new_m - 1);
        const gid_t base2    = 4 * (k_first - j2) + j2;
        const gid_t tw_off2  = 6 * new_m + 3 * j2;
        tw12 = vload2(0, wi + tw_off2);
        tw3  = wi[tw_off2 + 2];

        uint idx0 = ((write_index    ) % 4) * 4 + ((write_index    ) / 4);
        uint idx1 = ((write_index + 1) % 4) * 4 + ((write_index + 1) / 4);
        uint idx2 = ((write_index + 2) % 4) * 4 + ((write_index + 2) / 4);
        uint idx3 = ((write_index + 3) % 4) * 4 + ((write_index + 3) / 4);

        ulong b0 = local_x[idx0];
        ulong b1 = modMul(local_x[idx1], tw12.s1);
        ulong b2 = modMul(local_x[idx2], tw12.s0);
        ulong b3 = modMul(local_x[idx3], tw3);

        r  = modAdd(b0, b1);
        r2 = modSub(b0, b1);
        b0 = r; b1 = r2;

        r  = modAdd(b2, b3);
        r2 = modMuli(modSub(b3, b2));
        b2 = r; b3 = r2;

        x[base2               ] = modAdd(b0, b2);
        x[base2 + (new_m << 1)] = modSub(b0, b2);
        x[base2 + new_m       ] = modAdd(b1, b3);
        x[base2 + ((new_m << 1) + new_m)] = modSub(b1, b3);

        write_index += 4;
        k_first     += m;
    }
}

__kernel void kernel_ntt_radix4_inverse_mm_2steps_last(__global ulong* restrict x,
                                                  __global ulong* restrict wi,
                                                  __global ulong* restrict digit_invweight,
                                                  const uint m) {
    const gid_t gid      = get_global_id(0);
    const gid_t group    = gid / m;
    const gid_t local_id = gid % m;
    uint k_first         = group * m * 4 + local_id;

    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;

    uint write_index = 0;
    uint base        = 4 * (k_first - local_id) + local_id;
    const uint tw_offset = 6 * m + 3 * local_id;
    ulong2 tw12     = vload2(0, wi + tw_offset);
    ulong tw3       = wi[tw_offset + 2];
    ulong r, r2;

    #pragma unroll 4
    for (uint pass = 0; pass < 4; ++pass) {
        ulong a0 = x[base];
        ulong a1 = modMul(x[base + m],           tw12.s1);
        ulong a2 = modMul(x[base + (m << 1)],    tw12.s0);
        ulong a3 = modMul(x[base + ((m << 1) + m)], tw3);

        r  = modAdd(a0, a1);
        r2 = modSub(a0, a1);
        a0 = r; a1 = r2;

        r  = modAdd(a2, a3);
        r2 = modMuli(modSub(a3, a2));
        a2 = r; a3 = r2;

        r  = modAdd(a0, a2);
        r2 = modSub(a0, a2);
        a0 = r; a2 = r2;

        r  = modAdd(a1, a3);
        a3 = modSub(a1, a3);
        a1 = r;

        local_x[write_index    ] = a0;
        local_x[write_index + 1] = a1;
        local_x[write_index + 2] = a2;
        local_x[write_index + 3] = a3;

        write_index += 4;
        base        += 4 * m;
        k_first     += m;
    }

    const uint new_m = m * 4;
    write_index     = 0;
    k_first         = group * m * 4 + local_id;

    #pragma unroll 4
    for (uint pass = 0; pass < 4; ++pass) {
        const gid_t j2       = k_first & (new_m - 1);
        const gid_t base2    = 4 * (k_first - j2) + j2;
        const gid_t tw_off2  = 6 * new_m + 3 * j2;
        tw12 = vload2(0, wi + tw_off2);
        tw3  = wi[tw_off2 + 2];

        uint idx0 = ((write_index    ) % 4) * 4 + ((write_index    ) / 4);
        uint idx1 = ((write_index + 1) % 4) * 4 + ((write_index + 1) / 4);
        uint idx2 = ((write_index + 2) % 4) * 4 + ((write_index + 2) / 4);
        uint idx3 = ((write_index + 3) % 4) * 4 + ((write_index + 3) / 4);

        ulong b0 = local_x[idx0];
        ulong b1 = modMul(local_x[idx1], tw12.s1);
        ulong b2 = modMul(local_x[idx2], tw12.s0);
        ulong b3 = modMul(local_x[idx3], tw3);

        r  = modAdd(b0, b1);
        r2 = modSub(b0, b1);
        b0 = r; b1 = r2;

        r  = modAdd(b2, b3);
        r2 = modMuli(modSub(b3, b2));
        b2 = r; b3 = r2;

        x[base2               ] = modMul(modAdd(b0, b2),digit_invweight[base2]);
        x[base2 + (new_m << 1)] = modMul(modSub(b0, b2),digit_invweight[base2 + (new_m << 1)]);
        x[base2 + new_m       ] = modMul(modAdd(b1, b3),digit_invweight[base2 + new_m]);
        x[base2 + ((new_m << 1) + new_m)] = modMul(modSub(b1, b3),digit_invweight[base2 + ((new_m << 1) + new_m)]);

        write_index += 4;
        k_first     += m;
    }
}

__kernel void kernel_ntt_radix4_mm_2steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          const uint m) {

    const gid_t gid = get_global_id(0);
    const gid_t group = gid / (m / 4);
    const gid_t local_id = gid % (m / 4);
    uint k_first = group * m + local_id;

    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;

    #pragma unroll 4
    for (uint p = 0; p < 4; ++p) {
        uint k0       = k_first + p * (m >> 2);
        uint j        = k0 & (m - 1);
        uint base     = 4 * (k0 - j) + j;
        ulong2 tw12   = vload2(0, w + 6 * m + 3 * j);
        ulong  tw3    = w[6 * m + 3 * j + 2];
        ulong  a0     = x[base];
        ulong  a1     = x[base + m];
        ulong  a2     = x[base + (m << 1)];
        ulong  a3     = x[base + ((m << 1) + m)];
        ulong  t      = modSub(a0, a2);
        a0            = modAdd(a0, a2);
        a2            = t;
        t             = modMuli(modSub(a1, a3));
        a1            = modAdd(a1, a3);
        a3            = t;
        t             = modSub(a0, a1);
        a0            = modAdd(a0, a1);
        a1            = t;
        t             = modSub(a2, a3);
        a2            = modAdd(a2, a3);
        a3            = t;
        local_x[p*4 + 0] = a0;
        local_x[p*4 + 1] = modMul(a1, tw12.s1);
        local_x[p*4 + 2] = modMul(a2, tw12.s0);
        local_x[p*4 + 3] = modMul(a3, tw3);
    }


    const uint new_m = m / 4;
    
    const uint twiddle_offset = 6 * new_m + 3 * local_id;
    k_first = 4 * (group * m) + local_id;
    ulong2 twiddle1_2 = vload2(0, w + twiddle_offset);
    ulong twiddle3 = w[twiddle_offset + 2];
    
    uint write_index = 0;
    #pragma unroll 4
    for (uint pass = 0; pass < 4; ++pass) {
        uint idx0 = pass;
        uint idx1 = pass + 4;
        uint idx2 = pass + 8;
        uint idx3 = pass + 12;

        ulong a0 = local_x[idx0];
        ulong a1 = local_x[idx1];
        ulong a2 = local_x[idx2];
        ulong a3 = local_x[idx3];

        ulong t  = modSub(a0, a2);
        a0 = modAdd(a0, a2);
        a2 = t;
        t  = modMuli(modSub(a1, a3));
        a1 = modAdd(a1, a3);
        a3 = t;

        ulong s0 = modAdd(a0, a1);
        ulong s1 = modSub(a0, a1);
        ulong s2 = modAdd(a2, a3);
        ulong s3 = modSub(a2, a3);

        x[k_first]              = s0;
        x[k_first + new_m]      = modMul(s1, twiddle1_2.s1);
        x[k_first + (new_m<<1)] = modMul(s2, twiddle1_2.s0);
        x[k_first + new_m*3]    = modMul(s3, twiddle3);

        k_first += m;
    }



}

__kernel void kernel_ntt_radix4_mm_2steps_first(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          __global ulong* restrict digit_weight,
                                          const uint m
                                          ) {


    const gid_t gid = get_global_id(0);
    const gid_t group = gid / (m / 4);
    const gid_t local_id = gid % (m / 4);
    uint k_first = group * m + local_id;

    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;

    #pragma unroll 4
    for (uint p = 0; p < 4; ++p) {
        uint k0       = k_first + p * (m >> 2);
        uint j        = k0 & (m - 1);
        uint base     = 4 * (k0 - j) + j;
        ulong2 tw12   = vload2(0, w + 6 * m + 3 * j);
        ulong  tw3    = w[6 * m + 3 * j + 2];
        ulong  a0     = modMul(x[base],digit_weight[base]);
        ulong  a1     = modMul(x[base + m], digit_weight[base + m]);
        ulong  a2     = modMul(x[base + (m << 1)], digit_weight[base + (m << 1)]);
        ulong  a3     = modMul(x[base + ((m << 1) + m)], digit_weight[base + ((m << 1) + m)]);
        ulong  t      = modSub(a0, a2);
        a0            = modAdd(a0, a2);
        a2            = t;
        t             = modMuli(modSub(a1, a3));
        a1            = modAdd(a1, a3);
        a3            = t;
        t             = modSub(a0, a1);
        a0            = modAdd(a0, a1);
        a1            = t;
        t             = modSub(a2, a3);
        a2            = modAdd(a2, a3);
        a3            = t;
        local_x[p*4 + 0] = a0;
        local_x[p*4 + 1] = modMul(a1, tw12.s1);
        local_x[p*4 + 2] = modMul(a2, tw12.s0);
        local_x[p*4 + 3] = modMul(a3, tw3);
    }


    const uint new_m = m / 4;
    
    const uint twiddle_offset = 6 * new_m + 3 * local_id;
    k_first = 4 * (group * m) + local_id;
    ulong2 twiddle1_2 = vload2(0, w + twiddle_offset);
    ulong twiddle3 = w[twiddle_offset + 2];
    
    uint write_index = 0;
    #pragma unroll 4
    for (uint pass = 0; pass < 4; ++pass) {
        uint idx0 = pass;
        uint idx1 = pass + 4;
        uint idx2 = pass + 8;
        uint idx3 = pass + 12;

        ulong a0 = local_x[idx0];
        ulong a1 = local_x[idx1];
        ulong a2 = local_x[idx2];
        ulong a3 = local_x[idx3];

        ulong t  = modSub(a0, a2);
        a0 = modAdd(a0, a2);
        a2 = t;
        t  = modMuli(modSub(a1, a3));
        a1 = modAdd(a1, a3);
        a3 = t;

        ulong s0 = modAdd(a0, a1);
        ulong s1 = modSub(a0, a1);
        ulong s2 = modAdd(a2, a3);
        ulong s3 = modSub(a2, a3);

        x[k_first]              = s0;
        x[k_first + new_m]      = modMul(s1, twiddle1_2.s1);
        x[k_first + (new_m<<1)] = modMul(s2, twiddle1_2.s0);
        x[k_first + new_m*3]    = modMul(s3, twiddle3);

        k_first += m;
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

#define VECTORS_PER_WORKITEM 4
#ifndef W15_01_Y
  #define W15_01_Y 0
#endif
#ifndef W15_2_Y
  #define W15_2_Y 0
#endif
#ifndef WI15_01_Y
  #define WI15_01_Y 0
#endif
#ifndef WI15_2_X
  #define WI15_2_X 0
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
    localX[local_base_vec + 1].s0 = modSub(a, b);
    localX[local_base_vec + 2].s0 = modAdd(d, e);
    localX[local_base_vec + 3].s0 = modSub(d, e);


    a = modAdd(localX[local_base_vec + 0].s1, localX[local_base_vec + 2].s1);
    b = modAdd(localX[local_base_vec + 1].s1, localX[local_base_vec + 3].s1);
    d = modSub(localX[local_base_vec + 0].s1, localX[local_base_vec + 2].s1);
    e = modMuli(modSub(localX[local_base_vec + 1].s1, localX[local_base_vec + 3].s1));


    localX[local_base_vec + 0].s1 = modAdd(a, b);
    localX[local_base_vec + 1].s1 = modMuli(modSub(a, b));
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
    

    ulong r =  localX[local_base_vec + 1].s0; 
    ulong rs0 = modAdd(localX[local_base_vec + 0].s0, r);
    ulong rs1 = modSub(localX[local_base_vec + 0].s0, r);
    r =  localX[local_base_vec + 2].s0;
    ulong rr = localX[local_base_vec + 3].s0;
    ulong rs2 = modAdd(r, rr);
    ulong rs3 = modMuli(modSub(rr, r));

    x[global_base_vec + 0].s0 = modAdd(rs0, rs2);
    x[global_base_vec + 1].s0 = modAdd(rs1, rs3);
    x[global_base_vec + 2].s0 = modSub(rs0, rs2);
    x[global_base_vec + 3].s0 = modSub(rs1, rs3);



    r =  modMul(localX[local_base_vec + 1].s1, WI15_2_X); 
    rs0 = modAdd(localX[local_base_vec + 0].s1, r);
    rs1 = modSub(localX[local_base_vec + 0].s1, r);
    r =  modMul(localX[local_base_vec + 2].s1, WI15_01_Y);
    rr = modMul2Pow24(localX[local_base_vec + 3].s1);
    rs2 = modAdd(r, rr);
    rs3 = modMuli(modSub(rr, r));

    x[global_base_vec  + 0].s1 = modAdd(rs0, rs2);
    x[global_base_vec  + 1].s1 = modAdd(rs1, rs3);
    x[global_base_vec  + 2].s1 = modSub(rs0, rs2);
    x[global_base_vec  + 3].s1 = modSub(rs1, rs3);

}

__kernel void kernel_ntt_radix4_square_radix4(__global ulong4* restrict x)
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
    coeff = modMul3_2_w10(coeff);
    coeff = modMul4(coeff, coeff);
    
    ulong4 u = modMul3_2(coeff,(ulong2)(WI6,WI7), WI8);
    
    ulong v0 = modAdd(u.s0, u.s1);
    ulong v1 = modSub(u.s0, u.s1);
    ulong v2 = modAdd(u.s2, u.s3);
    ulong v3 = modMuli(modSub(u.s3, u.s2));
    ulong4 result = (ulong4)( modAdd(v0, v2),
                              modAdd(v1, v3),
                              modSub(v0, v2),
                              modSub(v1, v3) );
    x[k] = result;
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


__kernel void kernel_res64_display(
    __global const ulong* x,             
    uint                  exponent,   
    uint                  digitCount,
    uint                  mode,
    uint                  iter,
    __global uint*        out
) {
    if (get_global_id(0) != 0) return;
    uint wordCount;
    compactBits(x, digitCount, exponent, out, &wordCount);
    printf("Iter: %u | Res64: %08X%08X\n",
           iter,        // itération courante
           out[1],      // haut 32 bits
           out[0]       // bas 32 bits
    );
}
#define TRANSFORM_SIZE_N_DIV5        (TRANSFORM_SIZE_N / 5)
#define C_2_TRANSFORM_SIZE_N_DIV5    (2*(TRANSFORM_SIZE_N / 5))
#define C_3_TRANSFORM_SIZE_N_DIV5    (3*(TRANSFORM_SIZE_N / 5))
#define C_4_TRANSFORM_SIZE_N_DIV5    (4*(TRANSFORM_SIZE_N / 5))

#define PR5_A0 1373043270956696022UL
#define PR5_A1 211587555138949697UL
#define PR5_A2 15820824984080659046UL
#define PR5_A3 1041288259238279555UL

#define PR5_INV2 9223372034707292161UL
#define PR5_F2   17239578304317096532UL
#define PR5_F3   1584630826095645719UL
#define PR5_F4   17193868255037355068UL
#define PR5_F5   165877505859208233UL

__constant ulong4 K_4  = (ulong4)(PR5_A0, PR5_A0, PR5_A0, PR5_A0);
__constant ulong2 K_2  = (ulong2)(PR5_A0, PR5_A0);
__constant ulong4 K2_4 = (ulong4)(PR5_A1, PR5_A1, PR5_A1, PR5_A1);

#define F2 PR5_F2
#define F3 PR5_F3
#define F4 PR5_F4
#define F5 PR5_F5

__constant ulong2 F25_2 = (ulong2)(F2, F5);
__constant ulong2 F34_2 = (ulong2)(F3, F4);

#define LOCAL_SIZE5_TIMES2 (2*LOCAL_SIZE5)
#define LOCAL_SIZE5_TIMES3 (3*LOCAL_SIZE5)
#define LOCAL_SIZE5_TIMES4 (4*LOCAL_SIZE5)

__kernel void kernel_ntt_radix5_mm_first(
    __global ulong * restrict x,
    __global const ulong2 * restrict w5,
    __global const ulong * restrict digit_weight)
{
    __local ulong lm[5*LOCAL_SIZE5];

    const uint lid = get_local_id(0);
    const uint k = get_global_id(0);

    ulong d0,d1,d2,d3,d4;

    d0 = digit_weight[k];
    d1 = digit_weight[k +     TRANSFORM_SIZE_N_DIV5];
    d2 = digit_weight[k + C_2_TRANSFORM_SIZE_N_DIV5];
    d3 = digit_weight[k + C_3_TRANSFORM_SIZE_N_DIV5];
    d4 = digit_weight[k + C_4_TRANSFORM_SIZE_N_DIV5];

    ulong u0 = modMul(x[k], d0);
    lm[LOCAL_SIZE5+lid]       = modMul(x[k +     TRANSFORM_SIZE_N_DIV5], d1);
    lm[LOCAL_SIZE5_TIMES2+lid]= modMul(x[k + C_2_TRANSFORM_SIZE_N_DIV5], d2);
    lm[LOCAL_SIZE5_TIMES3+lid]= modMul(x[k + C_3_TRANSFORM_SIZE_N_DIV5], d3);
    lm[LOCAL_SIZE5_TIMES4+lid]= modMul(x[k + C_4_TRANSFORM_SIZE_N_DIV5], d4);

    d0 = modAdd(lm[LOCAL_SIZE5+lid],        lm[LOCAL_SIZE5_TIMES4+lid]);
    d1 = modSub(lm[LOCAL_SIZE5+lid],        lm[LOCAL_SIZE5_TIMES4+lid]);
    d2 = modAdd(lm[LOCAL_SIZE5_TIMES2+lid], lm[LOCAL_SIZE5_TIMES3+lid]);
    d3 = modSub(lm[LOCAL_SIZE5_TIMES2+lid], lm[LOCAL_SIZE5_TIMES3+lid]);

    ulong z0 = modAdd(d0, d2);
    x[k] = modAdd(z0, u0);
    ulong2 v25 = modMul2(F25_2, (ulong2)(modSub(d0, d2), modSub(d1, d3)));
    ulong2 v34 = modMul2(F34_2, (ulong2)(d3, d1));

    d0 = modAdd(u0, v25.s0);
    d1 = modSub(v34.s1, v25.s1);
    d2 = modSub(u0, v25.s0);
    d3 = modAdd(v34.s0, v25.s1);

    d4 = modAdd(d0, d1);//z1
    z0 = modSub(d0, d1);//z4
    d0 = modAdd(d2, d3);//z2
    d1 = modSub(d2, d3);//z3
    v25 = w5[(uint)k << 1];
    v34 = w5[((uint)k << 1) + 1];

    
    x[k +     TRANSFORM_SIZE_N_DIV5]     = modMul(modSub(d0, lm[LOCAL_SIZE5_TIMES4+lid]), v25.s0);
    x[k + C_2_TRANSFORM_SIZE_N_DIV5]     = modMul(modSub(z0, lm[LOCAL_SIZE5_TIMES2+lid]), v25.s1);
    x[k + C_3_TRANSFORM_SIZE_N_DIV5]     = modMul(modSub(d4, lm[LOCAL_SIZE5_TIMES3+lid]), v34.s0);
    x[k + C_4_TRANSFORM_SIZE_N_DIV5]     = modMul(modSub(d1, lm[LOCAL_SIZE5+lid]),       v34.s1);

}

__kernel void kernel_ntt_inverse_radix5_mm_last(
    __global ulong * restrict x,
    __global const ulong2 * restrict invw5,
    __global const ulong * restrict digit_inv_weight)
{
    __local ulong lm[5*LOCAL_SIZE5];

    const uint lid = get_local_id(0);
    const uint k = get_global_id(0);

    ulong d0,d1,d2,d3,d4;
    ulong2 v25 = invw5[(uint)k << 1];
    ulong2 v34 = invw5[((uint)k << 1) + 1];


    ulong u0 = x[k];

    lm[LOCAL_SIZE5+lid]       = modMul(x[k +     TRANSFORM_SIZE_N_DIV5],       v25.s0);
    lm[LOCAL_SIZE5_TIMES2+lid]= modMul(x[k + C_2_TRANSFORM_SIZE_N_DIV5],v25.s1);
    lm[LOCAL_SIZE5_TIMES3+lid]= modMul(x[k + C_3_TRANSFORM_SIZE_N_DIV5],v34.s0);
    lm[LOCAL_SIZE5_TIMES4+lid]= modMul(x[k + C_4_TRANSFORM_SIZE_N_DIV5],v34.s1);

    d0 = modAdd(lm[LOCAL_SIZE5_TIMES4+lid], lm[LOCAL_SIZE5+lid]);
    d1 = modSub(lm[LOCAL_SIZE5_TIMES4+lid], lm[LOCAL_SIZE5+lid]);
    d2 = modAdd(lm[LOCAL_SIZE5_TIMES3+lid], lm[LOCAL_SIZE5_TIMES2+lid]);
    d3 = modSub(lm[LOCAL_SIZE5_TIMES3+lid], lm[LOCAL_SIZE5_TIMES2+lid]);

    ulong z0 = modAdd(d0, d2);
    x[k] = modMul(modAdd(z0, u0), digit_inv_weight[k]);
    v25 = modMul2(F25_2, (ulong2)(modSub(d0, d2), modSub(d1, d3)));
    v34 = modMul2(F34_2, (ulong2)(d3, d1));

    ulong x2 = v25.s0;
    ulong x5 = v25.s1;
    ulong x3 = v34.s0;
    ulong x4 = v34.s1;


    d0 = modAdd(u0, v25.s0);
    d1 = modSub(v34.s1, v25.s1);
    d2 = modSub(u0, v25.s0);
    d3 = modAdd(v34.s0, v25.s1);

    d4 = modAdd(d0, d1);//z1
    z0 = modSub(d0, d1);//z4
    d0 = modAdd(d2, d3);//z2
    d1 = modSub(d2, d3);//z3



    
    x[k +     TRANSFORM_SIZE_N_DIV5] = modMul(modSub(d0, lm[LOCAL_SIZE5+lid]),        digit_inv_weight[k +     TRANSFORM_SIZE_N_DIV5]);
    x[k + C_2_TRANSFORM_SIZE_N_DIV5] = modMul(modSub(z0, lm[LOCAL_SIZE5_TIMES3+lid]), digit_inv_weight[k + C_2_TRANSFORM_SIZE_N_DIV5]);
    x[k + C_3_TRANSFORM_SIZE_N_DIV5] = modMul(modSub(d4, lm[LOCAL_SIZE5_TIMES2+lid]), digit_inv_weight[k + C_3_TRANSFORM_SIZE_N_DIV5]);
    x[k + C_4_TRANSFORM_SIZE_N_DIV5] = modMul(modSub(d1, lm[LOCAL_SIZE5_TIMES4+lid]), digit_inv_weight[k + C_4_TRANSFORM_SIZE_N_DIV5]);

}



#if __OPENCL_VERSION__ < 200
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
inline void atomic_min_u(__global volatile uint* p, uint v) {
    uint old = *p;
    while (v < old) {
        uint prev = atomic_cmpxchg((__global volatile int*)p, (int)old, (int)v);
        if ((uint)prev == old) break;
        old = (uint)prev;
    }
}
inline void atomic_xchg_u(__global volatile uint* p, uint v) {
    (void)atomic_xchg((__global volatile int*)p, (int)v);
}
#else
#define atomic_min_u atomic_min
#define atomic_xchg_u atomic_xchg
#endif

__kernel void check_equal(__global const ulong* a,
                          __global const ulong* b,
                          __global volatile uint* out_ok,
                          uint n) {
    uint gid = get_global_id(0);

    if (a[gid] != b[gid]) {
        atomic_xchg_u(out_ok, 0u);
    }

    
}
