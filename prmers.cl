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
#define MOD_P_COMP 0xffffffffU          // 2^64 - p = 2^32 - 1

inline ulong Add(const ulong lhs, const ulong rhs)
{
	const uint c = (lhs >= MOD_P  - rhs) ? MOD_P_COMP  : 0;
	return lhs + rhs + c;
}

inline ulong Sub(const ulong lhs, const ulong rhs)
{
	const uint c = (lhs < rhs) ? MOD_P_COMP  : 0;
	return lhs - rhs - c;
}

inline ulong Reduce(const ulong lo, const ulong hi)
{
	// hih * 2^96 + hil * 2^64 + lo = lo + hil * 2^32 - (hih + hil)
	const uint c = (lo >= MOD_P ) ? MOD_P_COMP  : 0;
	ulong r = lo + c;
	r = Add(r, hi << 32);				// lhs * rhs < p^2 => hi * 2^32 < p^2 / 2^32 < p.
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

__kernel void kernel_sub2(__global ulong* x,
                          __global int* digit_width,
                          const ulong n)
{
    if (get_global_id(0) == 0)
    {
        uint c = 2U;
        while(c != 0U){
            for(uint i = 0; i < n; i++){
                ulong val = x[i];
                ulong b   = (1UL << digit_width[i]);

                if (val >= c) {
                    x[i] = Sub(val, c);  // Utiliser une soustraction modulaire
                    c = 0U;
                    break;
                } else {
                    x[i] = Add(Sub(val, c), b);  // Correction de l'ajout
                    c = 1U;
                }
            }
        }
    }
}

#ifndef LOCAL_PROPAGATION_DEPTH
#define LOCAL_PROPAGATION_DEPTH 8
#endif

__kernel void kernel_carry(__global ulong* x,
                                    __global ulong* carry_array,
                                     __global int* digit_width,
                                     const ulong n,
                                     __global int* flag
                                     ) 
{


        ulong start = get_global_id(0)*LOCAL_PROPAGATION_DEPTH;
        ulong carry = 0UL;
        for (ulong i = start; i < (start + LOCAL_PROPAGATION_DEPTH); i++) {
            x[i] = digit_adc(x[i], digit_width[i], &carry);
        }
        carry_array[get_global_id(0)] = carry;
}


#ifndef CARRY_WORKER
#define CARRY_WORKER 1
#endif

__kernel void kernel_carry_2(__global ulong* x,
                                    __global ulong* carry_array,
                                    __global ulong* carry_array_out,
                                     __global int* digit_width,
                                     const ulong n,
                                     __global int* flag
                                     )
{
        ulong start = get_global_id(0)*LOCAL_PROPAGATION_DEPTH;
//        ulong carry = carry_array[start == 0 ? CARRY_WORKER -1 : (get_global_id(0) - 1) ];
        ulong carry = carry_array[start == 0 ? get_global_size(0) -1 : (get_global_id(0) - 1)];
        for (ulong i = start; i < (start + LOCAL_PROPAGATION_DEPTH -1); i++) {
            x[i] = digit_adc(x[i], digit_width[i], &carry);
        }
        x[(start + LOCAL_PROPAGATION_DEPTH -1)] += carry;
}

__kernel void kernel_carry_3(__global ulong* x,
                            __global ulong* carry_array,
                            __global int* flag)
{

        //printf("Thread %u read carry = %u",get_global_id(0), carry_array[get_global_id(0)]);
        ulong start = get_global_id(0)*LOCAL_PROPAGATION_DEPTH;
        ulong carry = carry_array[start == 0 ? CARRY_WORKER -1 : (get_global_id(0) - 1) ];
        x[start] += carry;
}


__kernel void kernel_ntt_radix4(__global ulong* x, __global ulong* w, const ulong n, const ulong m) {
   
    ulong k = get_global_id(0);
    
    if (k >= n / 4) return;  // Ne pas dépasser les limites

    ulong j = k & (m - 1);
    ulong i = 4 * (k - j) + j;

    __global ulong* wm = &w[3 * 2 * m];

    ulong w2  = wm[3 * j + 0];
    ulong w1  = wm[3 * j + 1];
    ulong w12 = wm[3 * j + 2];

    ulong u0 = x[i + 0 * m], u1 = x[i + 1 * m], u2 = x[i + 2 * m], u3 = x[i + 3 * m];
    ulong v0 = Add(u0, u2);
    ulong v1 = Add(u1, u3);
    ulong v2 = Sub(u0, u2);
    ulong v3 = modMuli(Sub(u1, u3));
    
    x[i + 0 * m] = Add(v0, v1);
    x[i + 1 * m] = modMul(Sub(v0, v1), w1);
    x[i + 2 * m] = modMul(Add(v2, v3), w2);
    x[i + 3 * m] = modMul(Sub(v2, v3), w12);
}


__kernel void kernel_inverse_ntt_radix4(__global ulong* x, __global ulong* wi, const ulong n, const ulong m) {
    ulong k = get_global_id(0);
    if (k >= n / 4) return;
    __global const ulong* invwm = wi + (3 * 2 * m);
    const ulong j = k & (m - 1);
    const ulong i = 4 * (k - j) + j;
    
    ulong coeff[4];
    coeff[0] = x[i + 0 * m];
    coeff[1] = x[i + 1 * m];
    coeff[2] = x[i + 2 * m];
    coeff[3] = x[i + 3 * m];
    
    const ulong iw2  = invwm[3 * j + 0];
    const ulong iw1  = invwm[3 * j + 1];
    const ulong iw12 = invwm[3 * j + 2];
    
    ulong u0 = coeff[0];
    ulong u1 = modMul(coeff[1], iw1);
    ulong u2 = modMul(coeff[2], iw2);
    ulong u3 = modMul(coeff[3], iw12);
    
    ulong v0 = Add(u0, u1);
    ulong v1 = Sub(u0, u1);
    ulong v2 = Add(u2, u3);
    ulong v3 = modMuli(Sub(u3, u2));
    
    coeff[0] = Add(v0, v2);
    coeff[1] = Add(v1, v3);
    coeff[2] = Sub(v0, v2);
    coeff[3] = Sub(v1, v3);
    
    x[i + 0 * m] = coeff[0];
    x[i + 1 * m] = coeff[1];
    x[i + 2 * m] = coeff[2];
    x[i + 3 * m] = coeff[3];
}


__kernel void kernel_square(__global ulong* x, const ulong n) {
    size_t i = get_global_id(0);
    while (i < n) {
        x[i] = modMul(x[i], x[i]);
        i += get_global_size(0);
    }
}

__kernel void kernel_precomp(__global ulong* x, __global ulong* digit_weight, const ulong n) {
    size_t i = get_global_id(0);
    x[i] = modMul(x[i], digit_weight[i]);
}

__kernel void kernel_postcomp(__global ulong* x, __global ulong* digit_invweight, const ulong n) {
    size_t i = get_global_id(0);
    x[i] = modMul(x[i], digit_invweight[i]);
}

