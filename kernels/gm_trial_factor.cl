// Gaussian-Mersenne / Gaussian-quotient trial factoring.
//
// For odd prime p, m=(p+1)/2 and eps=(2/p):
//   GM(p)     = 2^p - eps*2^m + 1
//   5*GQ(p)   = 2^p + eps*2^m + 1
//
// Every non-5 prime divisor has q = 4*k*p + 1. One exponentiation computes
// t=2^m mod q; 2^p follows from t^2/2, so GM and GQ are classified together.

#ifndef MAX_TF_FACTORS
#define MAX_TF_FACTORS 64
#endif

typedef struct {
    ulong factor;
    uint family_bits; // 1=GM, 2=GQ
    uint reserved;
} gm_tf_factor;

inline ulong add_mod_u64(ulong a, ulong b, ulong n) {
    return a >= n - b ? a - (n - b) : a + b;
}

inline ulong sub_mod_u64(ulong a, ulong b, ulong n) {
    return a >= b ? a - b : n - (b - a);
}

inline ulong half_mod_u64(ulong a, ulong n) {
    return (a >> 1) + ((a & 1UL) ? ((n >> 1) + 1UL) : 0UL);
}

inline ulong montgomery_nprime(ulong n) {
    ulong inverse = 1UL;
    inverse *= 2UL - n * inverse;
    inverse *= 2UL - n * inverse;
    inverse *= 2UL - n * inverse;
    inverse *= 2UL - n * inverse;
    inverse *= 2UL - n * inverse;
    inverse *= 2UL - n * inverse;
    return 0UL - inverse;
}

inline ulong mont_mul_u64(ulong a, ulong b, ulong n, ulong nprime) {
    const ulong lo = a * b;
    const ulong hi = mul_hi(a, b);
    const ulong m = lo * nprime;
    const ulong mn_lo = m * n;
    const ulong mn_hi = mul_hi(m, n);
    const ulong sum_lo = lo + mn_lo;
    const ulong carry = sum_lo < lo;
    const ulong partial = hi + mn_hi;
    const int overflow_partial = partial < hi;
    ulong value = partial + carry;
    const int overflow_carry = value < partial;
    if (overflow_partial || overflow_carry) {
        // The true REDC value contains an extra 2^64; subtract n without
        // losing that carry in the 64-bit representation.
        value += 0UL - n;
    } else if (value >= n) {
        value -= n;
    }
    return value;
}

inline ulong pow2_mont_u64(ulong exponent, ulong n, ulong nprime, ulong one_mont) {
    ulong result = one_mont;
    const ulong two_mont = add_mod_u64(one_mont, one_mont, n);
    ulong base = two_mont;
    while (exponent != 0UL) {
        if (exponent & 1UL) result = mont_mul_u64(result, base, n, nprime);
        exponent >>= 1;
        if (exponent != 0UL) base = mont_mul_u64(base, base, n, nprime);
    }
    return result;
}

__kernel void gm_trial_factor(
    __global const ulong* k_values,
    const ulong candidate_count,
    const ulong step,
    const ulong middle,
    const int epsilon,
    const uint requested_family_bits,
    volatile __global uint* factor_count,
    __global gm_tf_factor* factors)
{
    const ulong index = (ulong)get_global_id(0);
    if (index >= candidate_count) return;

    const ulong k = k_values[index];
    const ulong q = step * k + 1UL;
    if (q <= 5UL || (q & 1UL) == 0UL) return;

    const ulong nprime = montgomery_nprime(q);
    const ulong one_mont = (0UL - q) % q; // 2^64 mod q
    const ulong t_mont = pow2_mont_u64(middle, q, nprime, one_mont);
    const ulong t2_mont = mont_mul_u64(t_mont, t_mont, q, nprime);
    const ulong a_mont = half_mod_u64(t2_mont, q); // 2^p in Montgomery form

    ulong gm_residue;
    ulong gq_residue;
    if (epsilon > 0) {
        gm_residue = add_mod_u64(sub_mod_u64(a_mont, t_mont, q), one_mont, q);
        gq_residue = add_mod_u64(add_mod_u64(a_mont, t_mont, q), one_mont, q);
    } else {
        gm_residue = add_mod_u64(add_mod_u64(a_mont, t_mont, q), one_mont, q);
        gq_residue = add_mod_u64(sub_mod_u64(a_mont, t_mont, q), one_mont, q);
    }

    uint family_bits = 0U;
    if ((requested_family_bits & 1U) && gm_residue == 0UL) family_bits |= 1U;
    if ((requested_family_bits & 2U) && gq_residue == 0UL) family_bits |= 2U;
    if (family_bits == 0U) return;

    const uint slot = atomic_inc(factor_count);
    if (slot < MAX_TF_FACTORS) {
        factors[slot].factor = q;
        factors[slot].family_bits = family_bits;
        factors[slot].reserved = 0U;
    }
}
