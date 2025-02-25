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

constant ulong MOD_P = (((1UL << 32) - 1UL) << 32) + 1UL;  // p = 2^64 - 2^32 + 1

// Compute the 128-bit product of a and b as high:low.
inline void mul128(ulong a, ulong b, __private ulong *hi, __private ulong *lo) {
    uint a0 = (uint)(a & 0xFFFFFFFFUL);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b & 0xFFFFFFFFUL);
    uint b1 = (uint)(b >> 32);
    ulong p0 = (ulong)a0 * (ulong)b0;
    ulong p1 = (ulong)a0 * (ulong)b1;
    ulong p2 = (ulong)a1 * (ulong)b0;
    ulong p3 = (ulong)a1 * (ulong)b1;
    ulong mid = p1 + p2;
    ulong carry = (mid < p1) ? (1UL << 32) : 0UL;
    *hi = p3 + (mid >> 32) + carry;
    *lo = (mid << 32) + p0;
    if(*lo < p0)
        (*hi)++;
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

// Modular multiplication using the 128-bit multiplication.
inline ulong modMul(ulong a, ulong b) {
    ulong hi, lo;
    mul128(a, b, &hi, &lo);
    
    // Decompose hi into high and low 32-bit parts.
    uint A = (uint)(hi >> 32);
    uint B = (uint)(hi & 0xffffffffUL);

    // Step 1: r = lo + (B << 32)
    ulong r = lo;
    ulong old_r = r;
    r += ((ulong)B << 32);

    // Correct for overflow.
    if(r < old_r) {
        r += ((1UL << 32) - 1UL);
    }

    // Step 2: Subtract (A + B) modulo MOD_P.
    ulong sub_val = (ulong)A + (ulong)B;
    if(r >= sub_val)
        r -= sub_val;
    else
        r = r + MOD_P - sub_val;

    // Step 3: Final reduction.
    if(r >= MOD_P)
        r -= MOD_P;

    return r;
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
    if (i < n) {
        x[i] = modMul(x[i], digit_weight[i]);
    }
}

__kernel void kernel_postcomp(__global ulong* x,
                             __global ulong* digit_invweight,
                             const ulong n)
{
    size_t i = get_global_id(0);
    if (i < n) {
        x[i] = modMul(x[i], digit_invweight[i]);
    }
}

__kernel void kernel_square(__global ulong* x, const ulong n)
{
    size_t i = get_global_id(0);
    if (i < n) {
        ulong val = x[i];
        x[i] = modMul(val, val);
    }
}
__kernel void kernel_forward_ntt(__global ulong *x,
                                 const ulong n)
{
    // Un seul work-group => on synchronise tout le monde dans ce groupe
    // via barrier(CLK_LOCAL_MEM_FENCE) ou barrier(CLK_GLOBAL_MEM_FENCE).
    // On suppose get_global_size(0) == n.

    // On pré-calcule la racine de base : root = 7^((MOD_P-1)/n) mod p
    // (Pour info, MOD_P - 1 est divisible par n)
    ulong bigBase   = 7UL;
    ulong exponent  = (MOD_P - 1UL) / n;  // On suppose n divise p-1
    ulong root      = modExp(bigBase, exponent);  // la "racine de l'unité"

    // d_global sera stocké en local pour être partagé
    __local ulong d_global;

    // On itère sur m = n/2 jusqu'à 1 (log2(n) passes)
    // s est l'exposant pour la racine : root^s
    for (ulong m = n >> 1, s = 1; m >= 1; m >>= 1, s <<= 1) 
    {
        // Le work-item 0 calcule d_global = root^s
        if (get_local_id(0) == 0) {
            d_global = modExp(root, s);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase de butterfly parallèle
        for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) {
            ulong group = i / m;       // pour voir si pair/impair dans bloc
            ulong inBlockIndex = i % m;
            // On ne traite que la "moitié" du bloc => si i < m dans le bloc
            if ((group % 2) == 0) {
                // group est pair => i1 = i, i2 = i + m
                // Mais on doit s'assurer que i2 < n
                ulong i1 = i;
                ulong i2 = i + m;
                if (i2 < n) {
                    // butterfly
                    ulong u = x[i1];
                    ulong v = x[i2];

                    // On calcule d_j = (d_global)^inBlockIndex
                    ulong d_j = modExp(d_global, inBlockIndex);

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

/**
 * kernel_sub2
 * Soustrait 2 à x en base mixte. 
 */
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


__kernel void kernel_inverse_ntt(__global ulong* x,
                                 const ulong n)
{
    // Un seul work-group => on synchronise tout le monde avec barrier().
    // On suppose get_global_size(0) == n, ou un multiple qui >= n.

    __local ulong d_global;

    // 1) Calcul de la racine "root = 7^((p-1)/n) mod p"
    ulong root     = modExp((ulong)7, (MOD_P - 1UL)/n);

    // 2) Inverse de root => root_inv = root^(p-2) mod p
    //    (car x^(p-2) mod p = x^(-1) mod p)
    ulong root_inv = modExp(root, MOD_P - 2UL);

    // 3) Boucle de la NTT inverse
    for (ulong m = 1, s = (n >> 1); m <= (n >> 1); m <<= 1, s >>= 1)
    {
        // Work-item 0 calcule d_global = root_inv^s
        if (get_local_id(0) == 0) {
            d_global = modExp(root_inv, s);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Butterfly parallèle
        for (ulong i = get_global_id(0); i < n; i += get_global_size(0))
        {
            // Comme pour la forward-NTT, on peut faire un test sur (i / m) % 2
            // ou directement if((i % (2*m)) < m). Ici, on illustre la 2e forme :
            if ( (i % (2UL * m)) < m )
            {
                ulong i1 = i;
                ulong i2 = i + m;
                if (i2 < n) {
                    ulong u = x[i1];
                    ulong v = x[i2];

                    // inBlockIndex = i % m => exponent pour d_global^inBlockIndex
                    ulong inBlockIndex = i % m;
                    ulong d_j = modExp(d_global, inBlockIndex);

                    // v *= d_j (mod p)
                    v = modMul(v, d_j);

                    // new_u = u + v
                    // new_v = u - v
                    ulong new_u = modAdd_correct(u, v);
                    ulong new_v = modSub_correct(u, v);

                    x[i1] = new_u;
                    x[i2] = new_v;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/**
 * kernel_carry
 * Propagation de retenues dans x, selon digit_width[i].
 * 
 * Ici, on peut le faire avec un seul work-item (séquentiel)
 * ou avec un algo parallèle. La version la plus simple
 * (comme dans vos codes) est la version séquentielle.
 */
__kernel void kernel_carry(__global ulong* x,
                           __global int* digit_width,
                           const ulong n)
{
    // On peut se limiter à get_global_id(0)==0
    // et exécuter la boucle en séquentiel
    if (get_global_id(0) == 0)
    {
        ulong c = 0UL;
        for (ulong i = 0; i < n; i++){
            x[i] = digit_adc(x[i], digit_width[i], &c);
        }
        // Tant que c != 0, on refait...
        while(c != 0UL) {
            for (ulong i = 0; i < n; i++){
                x[i] = digit_adc(x[i], digit_width[i], &c);
                if(c == 0UL) break;
            }
        }
    }
    
}

// Main kernel for the Lucas–Lehmer Mersenne test.
__kernel void lucas_lehmer_mersenne_test(
    const uint p_min,
    const uint candidate_count,
    __global uint *results,

    // Precomputed buffers.
    __global ulong *digit_weight,
    __global ulong *digit_invweight,
    __global int   *digit_width,
    __global ulong *x,
    const ulong n_size
)
{

    // Each work-group handles one candidate.
    uint candidate_index = get_group_id(0);
    if (candidate_index >= candidate_count)
        return;

    // Compute candidate p = p_min + 2 * candidate_index.
    uint p = p_min + 2 * candidate_index;

    bool isprime = true;
    uint64_t n;
    int log_n, w;


    n = n_size;
    w = p / n;
    if(get_local_id(0)==0) {
        // Initialize x: set x[0] = 4 and the rest = 0.
        x[0] = 4UL;
        for(uint i = 1; i < n; i++)
            x[i] = 0UL;
        
    }

    barrier(CLK_LOCAL_MEM_FENCE);  // synchronize work-items

    // Broadcast n to all work-items in the group using local memory.
    __local ulong n_local;
    if (get_local_id(0) == 0)
        n_local = n;
    barrier(CLK_LOCAL_MEM_FENCE);
    n = n_local;
    barrier(CLK_GLOBAL_MEM_FENCE);  // Ensure initialization is complete

    __local ulong d_global;
    __local ulong d_inv_global;


    
    // Determine progress update interval for Lucas–Lehmer iterations.
    uint total_iters = p - 2;
    uint progress_interval = total_iters / 100;
    if(progress_interval == 0)
        progress_interval = 1;
    
    // Lucas–Lehmer loop: perform (p - 2) iterations.
    for(uint iter = 0; iter < (p - 2); iter++){
        // Step 1: Pre-compensation.
        for(uint i = get_local_id(0); i < n; i += get_local_size(0))
            x[i] = modMul(x[i], digit_weight[i]);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Step 2: Forward NTT on x.
        for(uint64_t m = n/2, s = 1; m >= 1; m /= 2, s *= 2){
            if(get_local_id(0)==0)
                d_global = modExp(modExp(7UL, (MOD_P - 1UL)/n), s);
            barrier(CLK_LOCAL_MEM_FENCE);
            for(uint i = get_local_id(0); i < n; i += get_local_size(0)){
                if((i % (2*m)) < m){
                    uint i1 = i;
                    uint i2 = i + m;
                    ulong u = x[i1];
                    ulong v = x[i2];
                    uint j = i % m;
                    ulong d_j = modExp(d_global, j);
                    x[i1] = modAdd_correct(u, v);
                    {
                        ulong diff = modSub_correct(u, v);
                        x[i2] = modMul(diff, d_j);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Step 3: Square each element: x[i] = x[i]^2.
        for(uint i = get_local_id(0); i < n; i += get_local_size(0))
            x[i] = modMul(x[i], x[i]);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Step 4: Inverse NTT on x.
        for(uint64_t m = 1, s = n/2; m <= n/2; m *= 2, s /= 2) {
            if(get_local_id(0) == 0) {
                d_inv_global = modExp(modExp(modExp(7UL, (MOD_P - 1UL)/n), MOD_P - 2UL), s);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            for(uint i = get_local_id(0); i < n; i += get_local_size(0)) {
                if((i % (2*m)) < m) {
                    uint i1 = i;
                    uint i2 = i + m;
                    ulong u = x[i1];
                    uint j = i % m;
                    ulong d_inv_j = modExp(d_inv_global, j);
                    ulong v = modMul(x[i2], d_inv_j);
                    ulong new_u = modAdd_correct(u, v);
                    ulong new_v = modSub_correct(u, v);
                    x[i1] = new_u;
                    x[i2] = new_v;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Step 5: Post-compensation.
        for(uint i = get_local_id(0); i < n; i += get_local_size(0))
            x[i] = modMul(x[i], digit_invweight[i]);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Step 6: Carry propagation.
        if(get_local_id(0)==0) {
            ulong c = 0UL;
            for(uint i = 0; i < n; i++){
                x[i] = digit_adc(x[i], digit_width[i], &c);
            }
            while(c != 0UL) {
                for(uint i = 0; i < n; i++){
                    x[i] = digit_adc(x[i], digit_width[i], &c);
                    if(c == 0UL)
                        break;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Step 7: Subtract 2.
        if(get_local_id(0)==0) {
            uint c = 2U;
            while(c != 0U){
                for(uint i = 0; i < n; i++){
                    ulong val = x[i];
                    ulong b = (1UL << digit_width[i]);
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
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    // Final reduction: check if x equals 0.
    uint local_zero = 1;
    for(uint i = get_local_id(0); i < n; i += get_local_size(0))
        if(x[i] != 0UL) local_zero = 0;
    __local uint group_zero;
    if(get_local_id(0)==0)
        group_zero = local_zero;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(get_local_id(0)==0) {
        results[candidate_index] = (group_zero ? p : 0);
    }
}
