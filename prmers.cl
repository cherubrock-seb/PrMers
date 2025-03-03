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

inline ulong Add(const ulong lhs, const ulong rhs)
{
    const uint c = (lhs >= MOD_P - rhs) ? MOD_P_COMP : 0;
    return lhs + rhs + c;
}

inline ulong Sub(const ulong lhs, const ulong rhs)
{
    const uint c = (lhs < rhs) ? MOD_P_COMP : 0;
    return lhs - rhs - c;
}

inline ulong Reduce(const ulong lo, const ulong hi)
{
    // hi_hi * 2^96 + hi_lo * 2^64 + lo = lo + hi_lo * 2^32 - (hi_hi + hi_lo)
    const uint c = (lo >= MOD_P) ? MOD_P_COMP : 0;
    ulong r = lo + c;
    r = Add(r, hi << 32);             // lhs * rhs < p^2 => hi * 2^32 < p^2 / 2^32 < p.
    r = Sub(r, (hi >> 32) + (uint)hi);
    return r;
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

// Correct modular addition accounting for potential overflow.
inline ulong modAdd_correct(ulong a, ulong b) {
    ulong s = a + b;
    int carry = (s < a) ? 1 : 0;
    if(carry)
        s += 0xffffffffUL;
    if(s >= MOD_P)
        s -= MOD_P;
    return s;
}

// Correct modular subtraction.
inline ulong modSub_correct(ulong a, ulong b) {
    if (a >= b){
        return a - b;
    }
    else{
        return MOD_P - b + a;
    }
}

__kernel void kernel_sub2(__global ulong* restrict x,
                          __global int* restrict digit_width,
                          const ulong n)
{
    if (get_global_id(0) == 0) {
        uint c = 2U;
        while(c != 0U) {
            for(uint i = 0; i < n; i++){
                const int d = digit_width[i];
                ulong val = x[i];
                // Calculate b only once
                const ulong b = 1UL << d;
                if (val >= c) {
                    x[i] = Sub(val, c);
                    c = 0U;
                    break;
                } else {
                    // Reuse the result of Sub to avoid redundant computation
                    const ulong temp = Sub(val, c);
                    x[i] = Add(temp, b);
                    c = 1U;
                }
            }
        }
    }
}


#ifndef LOCAL_PROPAGATION_DEPTH
#define LOCAL_PROPAGATION_DEPTH 8
#endif
#ifndef CARRY_WORKER
#define CARRY_WORKER 1
#endif

__kernel void kernel_carry(__global ulong* restrict x,
                           __global ulong* restrict carry_array,
                           __global int* restrict digit_width,
                           const ulong n,
                           __global int* restrict flag)
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    const ulong end = start + LOCAL_PROPAGATION_DEPTH;
    ulong carry = 0UL;
    
    #pragma unroll
    for (ulong i = start; i < end; i++) {
        x[i] = digit_adc(x[i], digit_width[i], &carry);
    }
    carry_array[gid] = carry;
}

__kernel void kernel_carry_2(__global ulong* restrict x,
                             __global ulong* restrict carry_array,
                             __global ulong* restrict carry_array_out,
                             __global int* restrict digit_width,
                             const ulong n,
                             __global int* restrict flag)
{
    const ulong gid = get_global_id(0);
    const ulong start = gid * LOCAL_PROPAGATION_DEPTH;
    // The loop executes over LOCAL_PROPAGATION_DEPTH-1 elements, the last one being used to accumulate the carry.
    const ulong end = start + LOCAL_PROPAGATION_DEPTH - 1;
    // Retrieve the carry from the previous block: if gid == 0, use the last block (circular).
    const ulong prev_gid = (gid == 0) ? (get_global_size(0) - 1) : (gid - 1);
    ulong carry = carry_array[prev_gid];
    
    #pragma unroll
    for (ulong i = start; i < end; i++) {
        x[i] = digit_adc(x[i], digit_width[i], &carry);
    }
    x[end] += carry;
}




__kernel void kernel_ntt_radix4(__global ulong* restrict x,
                                __global ulong* restrict w,
                                const ulong n,
                                const ulong m) {
    const ulong k = get_global_id(0);
    if (k >= n / 4) return;
    
    // Contiguous case: if m == 1, the data is stored contiguously and vector loading can be used
    if (m == 1) {
        // Vector loading of the 4 coefficients
        ulong4 coeff = vload4(0, x + 4 * k);
        
        // wm points to the twiddle factor table, in read-only mode via "restrict"
        const __global ulong* restrict wm = w + (3 * 2 * m);
        
        // For j == 0, directly retrieve the factors
        const ulong w2  = wm[0];
        const ulong w1  = wm[1];
        const ulong w12 = wm[2];
        
        // Extract coefficients
        ulong u0 = coeff.s0;
        ulong u1 = coeff.s1;
        ulong u2 = coeff.s2;
        ulong u3 = coeff.s3;
        
        // Perform butterfly transformation
        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));
        
        // Calculate the results
        ulong4 result = (ulong4)( Add(v0, v1),
                                  modMul(Sub(v0, v1), w1),
                                  modMul(Add(v2, v3), w2),
                                  modMul(Sub(v2, v3), w12) );
        
        // Vector store of the result
        vstore4(result, 0, x + 4 * k);
    } else {
        // Compute indices to access non-contiguous coefficients
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;
        
        // Declare wm with restrict to optimize access
        __global ulong* restrict wm = w + (3 * 2 * m);
        
        // Load twiddle factors for index j
        const ulong w2  = wm[3 * j + 0];
        const ulong w1  = wm[3 * j + 1];
        const ulong w12 = wm[3 * j + 2];
        
        // Individual loading of coefficients
        ulong u0 = x[i + 0 * m];
        ulong u1 = x[i + 1 * m];
        ulong u2 = x[i + 2 * m];
        ulong u3 = x[i + 3 * m];
        
        // Perform radix-4 transformation
        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));
        
        // Write the results to global memory
        x[i + 0 * m] = Add(v0, v1);
        x[i + 1 * m] = modMul(Sub(v0, v1), w1);
        x[i + 2 * m] = modMul(Add(v2, v3), w2);
        x[i + 3 * m] = modMul(Sub(v2, v3), w12);
    }
}


__kernel void kernel_inverse_ntt_radix4(__global ulong* restrict x,
                                         __global ulong* restrict wi,
                                         const ulong n,
                                         const ulong m) {
    const ulong k = get_global_id(0);
    if (k >= n / 4) return;
    
    // Pre-calculate the address of the inverse table (read-only)
    __global const ulong* restrict invwm = wi + (3 * 2 * m);
    
    // If m equals 1, the data is contiguous and vector operations can be used
    if (m == 1) {
        // k & (m-1) is always 0, so coefficients are stored contiguously
        ulong4 coeff = vload4(0, x + 4 * k);
        
        const ulong iw2  = invwm[0];
        const ulong iw1  = invwm[1];
        const ulong iw12 = invwm[2];
        
        ulong u0 = coeff.s0;
        ulong u1 = modMul(coeff.s1, iw1);
        ulong u2 = modMul(coeff.s2, iw2);
        ulong u3 = modMul(coeff.s3, iw12);
        
        ulong v0 = Add(u0, u1);
        ulong v1 = Sub(u0, u1);
        ulong v2 = Add(u2, u3);
        ulong v3 = modMuli(Sub(u3, u2));
        
        // Final computation and write using a vector operation
        ulong4 result = (ulong4)(Add(v0, v2), Add(v1, v3),
                                 Sub(v0, v2), Sub(v1, v3));
        vstore4(result, 0, x + 4 * k);
    }
    else {
        // Compute indices for potentially non-contiguous memory access
        const ulong j = k & (m - 1);
        const ulong base = 4 * (k - j) + j;
        
        // Individual loading of coefficients
        ulong coeff0 = x[base + 0 * m];
        ulong coeff1 = x[base + 1 * m];
        ulong coeff2 = x[base + 2 * m];
        ulong coeff3 = x[base + 3 * m];
        
        // Load the "twiddle" factors from the precomputed table
        const ulong iw2  = invwm[3 * j + 0];
        const ulong iw1  = invwm[3 * j + 1];
        const ulong iw12 = invwm[3 * j + 2];
        
        // Modular multiplications with inlining (assumed implemented inline)
        ulong u0 = coeff0;
        ulong u1 = modMul(coeff1, iw1);
        ulong u2 = modMul(coeff2, iw2);
        ulong u3 = modMul(coeff3, iw12);
        
        // Butterfly combination
        ulong v0 = Add(u0, u1);
        ulong v1 = Sub(u0, u1);
        ulong v2 = Add(u2, u3);
        ulong v3 = modMuli(Sub(u3, u2));
        
        // Calculate final coefficients and write to global memory
        x[base + 0 * m] = Add(v0, v2);
        x[base + 1 * m] = Add(v1, v3);
        x[base + 2 * m] = Sub(v0, v2);
        x[base + 3 * m] = Sub(v1, v3);
    }
}



__kernel void kernel_square(__global ulong* restrict x, const ulong n) {
    const size_t gid = get_global_id(0);
    const size_t gsize = get_global_size(0);
    for (size_t i = gid; i < n; i += gsize) {
        x[i] = modMul(x[i], x[i]);
    }
}


__kernel void kernel_precomp(__global ulong* restrict x,
                             __global ulong* restrict digit_weight,
                             const ulong n) {
    const size_t gid = get_global_id(0);
    // Assume that n is a multiple of 4 and that x and digit_weight are aligned.
    if(gid * 4 < n) {
        ulong4 vx = vload4(0, x + gid * 4);
        ulong4 vw = vload4(0, digit_weight + gid * 4);
        // Apply modMul to each component.
        vx.s0 = modMul(vx.s0, vw.s0);
        vx.s1 = modMul(vx.s1, vw.s1);
        vx.s2 = modMul(vx.s2, vw.s2);
        vx.s3 = modMul(vx.s3, vw.s3);
        vstore4(vx, 0, x + gid * 4);
    }
}

__kernel void kernel_postcomp(__global ulong* restrict x,
                              __global ulong* restrict digit_invweight,
                              const ulong n) {
    const size_t gid = get_global_id(0);
    // Assume that n is a multiple of 4 and that x and digit_invweight are aligned.
    if(gid * 4 < n) {
        ulong4 vx = vload4(0, x + gid * 4);
        ulong4 vw = vload4(0, digit_invweight + gid * 4);
        // Apply modMul to each component.
        vx.s0 = modMul(vx.s0, vw.s0);
        vx.s1 = modMul(vx.s1, vw.s1);
        vx.s2 = modMul(vx.s2, vw.s2);
        vx.s3 = modMul(vx.s3, vw.s3);
        vstore4(vx, 0, x + gid * 4);
    }
}
