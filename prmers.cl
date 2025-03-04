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
                                const ulong m,
                                __local ulong* local_w) {
    // Préchargement des twiddle factors dans la mémoire locale.
    // On charge les 3*m éléments situés à l'offset 3*2*m dans w.
    const ulong twiddle_offset = 3 * 2 * m;
    const ulong T = 3 * m;  // Nombre d'éléments à charger.
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    for (ulong i = lid; i < T; i += lsize) {
        local_w[i] = w[twiddle_offset + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const ulong k = get_global_id(0);
    if (k >= n / 4) return;

    // Cas contigu (m == 1) : utilisation de vload4/vstore4.
    if (m == 1) {
        ulong4 coeff = vload4(0, x + 4 * k);
        // Utilisation des facteurs chargés en mémoire locale.
        const ulong w2  = local_w[0];
        const ulong w1  = local_w[1];
        const ulong w12 = local_w[2];

        ulong u0 = coeff.s0;
        ulong u1 = coeff.s1;
        ulong u2 = coeff.s2;
        ulong u3 = coeff.s3;

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        ulong4 result = (ulong4)( Add(v0, v1),
                                  modMul(Sub(v0, v1), w1),
                                  modMul(Add(v2, v3), w2),
                                  modMul(Sub(v2, v3), w12) );
        vstore4(result, 0, x + 4 * k);
    } else {
        // Cas non contigu : indices calculés en fonction de j.
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;

        // Utilisation des twiddle factors depuis la mémoire locale.
        const ulong w2  = local_w[3 * j + 0];
        const ulong w1  = local_w[3 * j + 1];
        const ulong w12 = local_w[3 * j + 2];

        ulong u0 = x[i + 0 * m];
        ulong u1 = x[i + 1 * m];
        ulong u2 = x[i + 2 * m];
        ulong u3 = x[i + 3 * m];

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        x[i + 0 * m] = Add(v0, v1);
        x[i + 1 * m] = modMul(Sub(v0, v1), w1);
        x[i + 2 * m] = modMul(Add(v2, v3), w2);
        x[i + 3 * m] = modMul(Sub(v2, v3), w12);
    }
}

__kernel void kernel_ntt_radix4_alt(__global ulong* restrict x,
                                    __global ulong* restrict w,
                                    const ulong n,
                                    const ulong m) {
    const ulong k = get_global_id(0);
    if (k >= n / 4) return;

    // Calculate the base offset for twiddle factors in global memory.
    const ulong twiddle_offset = 3 * 2 * m;
    
    if (m == 1) {
        // When m == 1, coefficients are stored contiguously.
        ulong4 coeff = vload4(0, x + 4 * k);
        const __global ulong* wm = w + twiddle_offset;
        // Use global memory twiddle factors.
        const ulong w2  = wm[0];
        const ulong w1  = wm[1];
        const ulong w12 = wm[2];

        ulong u0 = coeff.s0;
        ulong u1 = coeff.s1;
        ulong u2 = coeff.s2;
        ulong u3 = coeff.s3;

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        ulong4 result = (ulong4)( Add(v0, v1),
                                  modMul(Sub(v0, v1), w1),
                                  modMul(Add(v2, v3), w2),
                                  modMul(Sub(v2, v3), w12) );
        vstore4(result, 0, x + 4 * k);
    } else {
        // For non-contiguous data: compute j and index.
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;
        const __global ulong* wm = w + twiddle_offset;

        // Each work-item fetches its own required twiddle factors from global memory.
        const ulong w2  = wm[3 * j + 0];
        const ulong w1  = wm[3 * j + 1];
        const ulong w12 = wm[3 * j + 2];

        ulong u0 = x[i + 0 * m];
        ulong u1 = x[i + 1 * m];
        ulong u2 = x[i + 2 * m];
        ulong u3 = x[i + 3 * m];

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        x[i + 0 * m] = Add(v0, v1);
        x[i + 1 * m] = modMul(Sub(v0, v1), w1);
        x[i + 2 * m] = modMul(Add(v2, v3), w2);
        x[i + 3 * m] = modMul(Sub(v2, v3), w12);
    }
}


__kernel void kernel_inverse_ntt_radix4(__global ulong* restrict x,
                                         __global ulong* restrict wi,
                                         const ulong n,
                                         const ulong m,
                                         __local ulong* local_iw) {
    // Préchargement des inverse twiddle factors dans la mémoire locale.
    const ulong twiddle_offset = 3 * 2 * m;
    const ulong T = 3 * m;
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    for (ulong i = lid; i < T; i += lsize) {
        local_iw[i] = wi[twiddle_offset + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const ulong k = get_global_id(0);
    if (k >= n / 4) return;

    if (m == 1) {
        ulong4 coeff = vload4(0, x + 4 * k);
        const ulong iw2  = local_iw[0];
        const ulong iw1  = local_iw[1];
        const ulong iw12 = local_iw[2];

        ulong u0 = coeff.s0;
        ulong u1 = modMul(coeff.s1, iw1);
        ulong u2 = modMul(coeff.s2, iw2);
        ulong u3 = modMul(coeff.s3, iw12);

        ulong v0 = Add(u0, u1);
        ulong v1 = Sub(u0, u1);
        ulong v2 = Add(u2, u3);
        ulong v3 = modMuli(Sub(u3, u2));

        ulong4 result = (ulong4)(Add(v0, v2), Add(v1, v3),
                                 Sub(v0, v2), Sub(v1, v3));
        vstore4(result, 0, x + 4 * k);
    } else {
        const ulong j = k & (m - 1);
        const ulong base = 4 * (k - j) + j;

        ulong coeff0 = x[base + 0 * m];
        ulong coeff1 = x[base + 1 * m];
        ulong coeff2 = x[base + 2 * m];
        ulong coeff3 = x[base + 3 * m];

        const ulong iw2  = local_iw[3 * j + 0];
        const ulong iw1  = local_iw[3 * j + 1];
        const ulong iw12 = local_iw[3 * j + 2];

        ulong u0 = coeff0;
        ulong u1 = modMul(coeff1, iw1);
        ulong u2 = modMul(coeff2, iw2);
        ulong u3 = modMul(coeff3, iw12);

        ulong v0 = Add(u0, u1);
        ulong v1 = Sub(u0, u1);
        ulong v2 = Add(u2, u3);
        ulong v3 = modMuli(Sub(u3, u2));

        x[base + 0 * m] = Add(v0, v2);
        x[base + 1 * m] = Add(v1, v3);
        x[base + 2 * m] = Sub(v0, v2);
        x[base + 3 * m] = Sub(v1, v3);
    }
}

__kernel void kernel_inverse_ntt_radix4_alt(__global ulong* restrict x,
                                             __global ulong* restrict wi,
                                             const ulong n,
                                             const ulong m) {
    const ulong k = get_global_id(0);
    if (k >= n / 4) return;

    // Base offset for inverse twiddle factors.
    const ulong twiddle_offset = 3 * 2 * m;
    
    if (m == 1) {
        ulong4 coeff = vload4(0, x + 4 * k);
        const __global ulong* invwm = wi + twiddle_offset;
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

        ulong4 result = (ulong4)(Add(v0, v2), Add(v1, v3),
                                 Sub(v0, v2), Sub(v1, v3));
        vstore4(result, 0, x + 4 * k);
    } else {
        const ulong j = k & (m - 1);
        const ulong base = 4 * (k - j) + j;
        const __global ulong* invwm = wi + twiddle_offset;

        ulong coeff0 = x[base + 0 * m];
        ulong coeff1 = x[base + 1 * m];
        ulong coeff2 = x[base + 2 * m];
        ulong coeff3 = x[base + 3 * m];

        const ulong iw2  = invwm[3 * j + 0];
        const ulong iw1  = invwm[3 * j + 1];
        const ulong iw12 = invwm[3 * j + 2];

        ulong u0 = coeff0;
        ulong u1 = modMul(coeff1, iw1);
        ulong u2 = modMul(coeff2, iw2);
        ulong u3 = modMul(coeff3, iw12);

        ulong v0 = Add(u0, u1);
        ulong v1 = Sub(u0, u1);
        ulong v2 = Add(u2, u3);
        ulong v3 = modMuli(Sub(u3, u2));

        x[base + 0 * m] = Add(v0, v2);
        x[base + 1 * m] = Add(v1, v3);
        x[base + 2 * m] = Sub(v0, v2);
        x[base + 3 * m] = Sub(v1, v3);
    }
}

__kernel void kernel_ntt_radix4_last(__global ulong* restrict x,
                                     __global ulong* restrict w,
                                     const ulong n,
                                     const ulong m,
                                     __local ulong* local_w) {
    // Preload twiddle factors into local memory.
    const ulong twiddle_offset = 3 * 2 * m;
    const ulong T = 3 * m;
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    for (ulong i = lid; i < T; i += lsize) {
        local_w[i] = w[twiddle_offset + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const ulong k = get_global_id(0);
    if (k >= n / 4) return;

    if (m == 1) {
        // Contiguous case: vector load/store.
        ulong4 coeff = vload4(0, x + 4 * k);
        const ulong w2  = local_w[0];
        const ulong w1  = local_w[1];
        const ulong w12 = local_w[2];

        ulong u0 = coeff.s0;
        ulong u1 = coeff.s1;
        ulong u2 = coeff.s2;
        ulong u3 = coeff.s3;

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        ulong4 result = (ulong4)( Add(v0, v1),
                                  modMul(Sub(v0, v1), w1),
                                  modMul(Add(v2, v3), w2),
                                  modMul(Sub(v2, v3), w12) );
        // Fused square: square each component.
        result.s0 = modMul(result.s0, result.s0);
        result.s1 = modMul(result.s1, result.s1);
        result.s2 = modMul(result.s2, result.s2);
        result.s3 = modMul(result.s3, result.s3);
        vstore4(result, 0, x + 4 * k);
    } else {
        // Non-contiguous case.
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;
        // Load twiddle factors from local memory.
        const ulong w2  = local_w[3 * j + 0];
        const ulong w1  = local_w[3 * j + 1];
        const ulong w12 = local_w[3 * j + 2];

        ulong u0 = x[i + 0 * m];
        ulong u1 = x[i + 1 * m];
        ulong u2 = x[i + 2 * m];
        ulong u3 = x[i + 3 * m];

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        // Compute results and fuse with square.
        ulong r0 = Add(v0, v1);
        ulong r1 = modMul(Sub(v0, v1), w1);
        ulong r2 = modMul(Add(v2, v3), w2);
        ulong r3 = modMul(Sub(v2, v3), w12);

        // Square each result before storing.
        x[i + 0 * m] = modMul(r0, r0);
        x[i + 1 * m] = modMul(r1, r1);
        x[i + 2 * m] = modMul(r2, r2);
        x[i + 3 * m] = modMul(r3, r3);
    }
}

__kernel void kernel_ntt_radix4_last_alt(__global ulong* restrict x,
                                         __global ulong* restrict w,
                                         const ulong n,
                                         const ulong m) {
    const ulong k = get_global_id(0);
    if (k >= n / 4) return;

    // Base offset for twiddle factors in global memory.
    const ulong twiddle_offset = 3 * 2 * m;
    
    if (m == 1) {
        ulong4 coeff = vload4(0, x + 4 * k);
        const __global ulong* wm = w + twiddle_offset;
        const ulong w2  = wm[0];
        const ulong w1  = wm[1];
        const ulong w12 = wm[2];

        ulong u0 = coeff.s0;
        ulong u1 = coeff.s1;
        ulong u2 = coeff.s2;
        ulong u3 = coeff.s3;

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        ulong4 result = (ulong4)( Add(v0, v1),
                                  modMul(Sub(v0, v1), w1),
                                  modMul(Add(v2, v3), w2),
                                  modMul(Sub(v2, v3), w12) );
        // Fused square.
        result.s0 = modMul(result.s0, result.s0);
        result.s1 = modMul(result.s1, result.s1);
        result.s2 = modMul(result.s2, result.s2);
        result.s3 = modMul(result.s3, result.s3);
        vstore4(result, 0, x + 4 * k);
    } else {
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;
        const __global ulong* wm = w + twiddle_offset;

        const ulong w2  = wm[3 * j + 0];
        const ulong w1  = wm[3 * j + 1];
        const ulong w12 = wm[3 * j + 2];

        ulong u0 = x[i + 0 * m];
        ulong u1 = x[i + 1 * m];
        ulong u2 = x[i + 2 * m];
        ulong u3 = x[i + 3 * m];

        ulong v0 = Add(u0, u2);
        ulong v1 = Add(u1, u3);
        ulong v2 = Sub(u0, u2);
        ulong v3 = modMuli(Sub(u1, u3));

        ulong r0 = Add(v0, v1);
        ulong r1 = modMul(Sub(v0, v1), w1);
        ulong r2 = modMul(Add(v2, v3), w2);
        ulong r3 = modMul(Sub(v2, v3), w12);

        // Fused square: square each computed digit.
        x[i + 0 * m] = modMul(r0, r0);
        x[i + 1 * m] = modMul(r1, r1);
        x[i + 2 * m] = modMul(r2, r2);
        x[i + 3 * m] = modMul(r3, r3);
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
