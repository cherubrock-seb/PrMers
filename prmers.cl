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

//constant ulong MOD_P = (((1UL << 32) - 1UL) << 32) + 1UL;  // p = 2^64 - 2^32 + 1

//constant uint32_t MOD_P_COMP = 0xffffffffu;  // 2^64 - p = 2^32 - 1


#define MOD_P 0xffffffff00000001UL  // p = 2^64 - 2^32 + 1
#define MOD_P_COMP 0xffffffffU          // 2^64 - p = 2^32 - 1

inline long GetLong(const ulong lhs) { return (lhs > MOD_P  / 2) ? (long)(lhs + MOD_P_COMP ) : (long)lhs; }

inline ulong ToZp(const long i) { return (i >= 0) ? (ulong)i : MOD_P  + i; }

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

// Modular addition modulo MOD_P.
inline ulong modAdd(ulong a, ulong b) {
    ulong s = a + b;
    return (s >= MOD_P) ? s - MOD_P : s;
}


// Modular subtraction modulo MOD_P.
inline ulong modSub(ulong a, ulong b) {
    return (a >= b) ? a - b : a + MOD_P - b;
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

// Modular exponentiation: computes base^exp modulo MOD_P.
inline ulong modExp(ulong base, ulong exp) {
    ulong result = 1UL;
    while(exp > 0) {
        if(exp & 1UL) {
            result = modMul(result, base);
        }
        base = modMul(base, base);
        exp >>= 1;
    }
    return result;
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


__kernel void kernel_precomp(__global ulong* x,
                             __global ulong* digit_weight,
                             const ulong n)
{
    size_t i = get_global_id(0);
    while (i < n) {
        x[i] = modMul(x[i], digit_weight[i]);
        i += get_global_size(0); // Chaque thread traite plusieurs éléments
    }
}

__kernel void kernel_postcomp(__global ulong* x,
                             __global ulong* digit_invweight,
                             const ulong n)
{
    size_t i = get_global_id(0);
    while (i < n) {
        x[i] = modMul(x[i], digit_invweight[i]);
        i += get_global_size(0); // Chaque thread traite plusieurs éléments
    }
}


__kernel void kernel_square(__global ulong* x, const ulong n)
{
    size_t i = get_global_id(0);
    while (i < n) {
        ulong val = x[i];
        x[i] = modMul(val, val);
        i += get_global_size(0); // Chaque thread traite plusieurs éléments
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
                if(val >= c){
                    x[i] = val - c;
                    c = 0U;
                    break;
                } else {
                    x[i] = val - c + b;
                    c = 1U;
                }
            }
        }
    }
}

__kernel void kernel_forward_ntt(__global ulong *x, const ulong n) {
    __local ulong d;

    ulong bigBase   = 7UL;
    ulong exponent  = (MOD_P - 1UL) / n;
    ulong root      = modExp(bigBase, exponent);

    for (ulong m = n >> 1, s = 1; m >= 1; m >>= 1, s <<= 1) {
        if (get_local_id(0) == 0) {
            d = modExp(root, s);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) {
            if ((i % (2UL * m)) < m) {
                ulong i1 = i;
                ulong i2 = i + m;
                if (i2 < n) {
                    ulong u = x[i1];
                    ulong v = x[i2];

                    ulong inBlockIndex = i % m;
                    ulong d_j = modExp(d, inBlockIndex); 

                    ulong sum  = modAdd_correct(u, v);
                    ulong diff = modSub_correct(u, v);

                    x[i1] = sum;
                    x[i2] = modMul(diff, d_j);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void kernel_inverse_ntt(__global ulong* x, const ulong n)
{
    // Shared variable for the twiddle factor (one per work-group)
    __local ulong twiddle_factor;

    // Compute the base root of unity: root = 7^((MOD_P-1)/n) mod p
    ulong root = modExp(7UL, (MOD_P - 1UL) / n);

    // Compute its modular inverse: root_inv = root^(p-2) mod p
    ulong root_inv = modExp(root, MOD_P - 2UL);

    // Iteratively perform the inverse NTT
    for (ulong m = 1, s = (n >> 1); m <= (n >> 1); m <<= 1, s >>= 1)
    {
        // The first thread in each work-group computes twiddle_factor = root_inv^s
        if (get_local_id(0) == 0) {
            twiddle_factor = modExp(root_inv, s);
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Synchronize all threads before using twiddle_factor

        // Perform the butterfly operation in parallel
        for (ulong i = get_global_id(0); i < n; i += get_global_size(0))
        {
            if ((i % (2UL * m)) < m) // Ensures each butterfly pair is processed correctly
            {
                ulong i1 = i;
                ulong i2 = i + m;
                if (i2 < n) {
                    ulong u = x[i1];
                    ulong v = x[i2];

                    // Compute the twiddle factor for this index
                    ulong twiddle = modExp(twiddle_factor, i % m);

                    // Apply butterfly transformation
                    v = modMul(v, twiddle);
                    x[i1] = modAdd_correct(u, v);
                    x[i2] = modSub_correct(u, v);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Synchronize threads before the next stage
    }
}

__kernel void kernel_carry(__global ulong* x,
                           __global int* digit_width,
                           const ulong n)
{
    // We can limit execution to get_global_id(0)==0 and run the loop sequentially
    if (get_global_id(0) == 0)
    {
        ulong c = 0UL;
        for (ulong i = 0; i < n; i++){
            x[i] = digit_adc(x[i], digit_width[i], &c);
        }
        // While c != 0, repeat...
        while(c != 0UL) {
            for (ulong i = 0; i < n; i++){
                x[i] = digit_adc(x[i], digit_width[i], &c);
                if(c == 0UL) break;
            }
        }
    }
    
}

__kernel void kernel_fusionne(__global ulong* x,
                                       __global ulong* digit_weight,
                                       __global ulong* digit_invweight,
                                       const ulong n)
{
    // Each work-item processes its range of indices.
    size_t id = get_global_id(0);
    size_t total = get_global_size(0);

    // --- Pre-multiplication: x[i] = modMul(x[i], digit_weight[i]) ---
    for (ulong i = id; i < n; i += total) {
        x[i] = modMul(x[i], digit_weight[i]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    __local ulong d;

    ulong bigBase   = 7UL;
    ulong exponent  = (MOD_P - 1UL) / n;
    ulong root      = modExp(bigBase, exponent);

    for (ulong m = n >> 1, s = 1; m >= 1; m >>= 1, s <<= 1) {
        if (get_local_id(0) == 0) {
            d = modExp(root, s);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) {
            if ((i % (2UL * m)) < m) {
                ulong i1 = i;
                ulong i2 = i + m;
                if (i2 < n) {
                    ulong u = x[i1];
                    ulong v = x[i2];

                    ulong inBlockIndex = i % m;
                    ulong d_j = modExp(d, inBlockIndex); 

                    ulong sum  = modAdd_correct(u, v);
                    ulong diff = modSub_correct(u, v);

                    x[i1] = sum;
                    x[i2] = modMul(diff, d_j);
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    // --- Pointwise squaring ---
    for (ulong i = id; i < n; i += total) {
        x[i] = modMul(x[i], x[i]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- Inverse NTT ---
    // Shared variable for the twiddle factor (one per work-group)
    __local ulong twiddle_factor;

    // Compute the base root of unity: root = 7^((MOD_P-1)/n) mod p
    root = modExp(7UL, (MOD_P - 1UL) / n);

    // Compute its modular inverse: root_inv = root^(p-2) mod p
    ulong root_inv = modExp(root, MOD_P - 2UL);

    // Iteratively perform the inverse NTT
    for (ulong m = 1, s = (n >> 1); m <= (n >> 1); m <<= 1, s >>= 1)
    {
        // The first thread in each work-group computes twiddle_factor = root_inv^s
        if (get_local_id(0) == 0) {
            twiddle_factor = modExp(root_inv, s);
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // Synchronize all threads before using twiddle_factor

        // Perform the butterfly operation in parallel
        for (ulong i = get_global_id(0); i < n; i += get_global_size(0))
        {
            if ((i % (2UL * m)) < m) // Ensures each butterfly pair is processed correctly
            {
                ulong i1 = i;
                ulong i2 = i + m;
                if (i2 < n) {
                    ulong u = x[i1];
                    ulong v = x[i2];

                    // Compute the twiddle factor for this index
                    ulong twiddle = modExp(twiddle_factor, i % m);

                    // Apply butterfly transformation
                    v = modMul(v, twiddle);
                    x[i1] = modAdd_correct(u, v);
                    x[i2] = modSub_correct(u, v);
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // Synchronize threads before the next stage
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    // --- Post-multiplication: x[i] = modMul(x[i], digit_invweight[i]) ---
    for (ulong i = id; i < n; i += total) {
        x[i] = modMul(x[i], digit_invweight[i]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}


#define WG_SIZE 256  // Maximum allocation size for local memory; actual group size = n/4
__kernel void kernel_fusionne_radix4(__global ulong* x,
                                      __global ulong* digit_weight,
                                      __global ulong* digit_invweight,
                                      __global ulong* w,    // Facteurs de twiddle pour la NTT directe
                                      __global ulong* wi,   // Facteurs de twiddle pour la NTT inverse
                                      const ulong n)
{
    // n_4 est le nombre de papillons.
    const ulong n_4 = n / 4;
    // Chaque work–item, identifié par k, traite un papillon.
    const ulong k = get_global_id(0);
    
    // --- Pré–multiplication ---
    // Chaque work–item traite les indices avec un pas de n_4.
    for (ulong i = k; i < n; i += n_4) {
        x[i] = modMul(x[i], digit_weight[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // --- NTT Direct (Radix-4) ---
    for (ulong m = n_4; m >= 1; m /= 4) {
        __global const ulong* wm = w + (3 * 2 * m);
        const ulong j = k & (m - 1);
        const ulong i = 4 * (k - j) + j;
        
        // Utilisation d'un tableau privé pour stocker les 4 coefficients du papillon.
        ulong coeff[4];
        coeff[0] = x[i + 0 * m];
        coeff[1] = x[i + 1 * m];
        coeff[2] = x[i + 2 * m];
        coeff[3] = x[i + 3 * m];
        
        // Récupérer les twiddles pour ce papillon.
        const ulong w2  = wm[3 * j + 0];
        const ulong w1  = wm[3 * j + 1];
        const ulong w12 = wm[3 * j + 2];
        
        // Calcul du papillon radix-4.
        ulong v0 = modAdd_correct(coeff[0], coeff[2]);
        ulong v1 = modAdd_correct(coeff[1], coeff[3]);
        ulong v2 = modSub_correct(coeff[0], coeff[2]);
        ulong v3 = modMuli(modSub_correct(coeff[1], coeff[3]));
        
        coeff[0] = modAdd_correct(v0, v1);
        coeff[1] = modMul(modSub_correct(v0, v1), w1);
        coeff[2] = modMul(modAdd_correct(v2, v3), w2);
        coeff[3] = modMul(modSub_correct(v2, v3), w12);
        
        // Écriture des résultats dans la mémoire globale.
        x[i + 0 * m] = coeff[0];
        x[i + 1 * m] = coeff[1];
        x[i + 2 * m] = coeff[2];
        x[i + 3 * m] = coeff[3];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // --- Carré point–à–point ---
    for (ulong i = k; i < n; i += n_4) {
        x[i] = modMul(x[i], x[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // --- NTT Inverse (Radix-4) ---
    for (ulong m = 1; m <= n_4; m *= 4) {
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
        
        ulong v0 = modAdd_correct(u0, u1);
        ulong v1 = modSub_correct(u0, u1);
        ulong v2 = modAdd_correct(u2, u3);
        ulong v3 = modMuli(modSub_correct(u3, u2));
        
        coeff[0] = modAdd_correct(v0, v2);
        coeff[1] = modAdd_correct(v1, v3);
        coeff[2] = modSub_correct(v0, v2);
        coeff[3] = modSub_correct(v1, v3);
        
        x[i + 0 * m] = coeff[0];
        x[i + 1 * m] = coeff[1];
        x[i + 2 * m] = coeff[2];
        x[i + 3 * m] = coeff[3];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // --- Post–multiplication ---
    for (ulong i = k; i < n; i += n_4) {
        x[i] = modMul(x[i], digit_invweight[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
