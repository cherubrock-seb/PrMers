/*
PrMers BananaNTT Split OpenCL kernels

Copyright 2026, Sébastien "Cherubrock"
OpenCL kernels for mixed CRT/PFA odd-radix half-real NTT.

Project:
https://github.com/cherubrock-seb/PrMers/tree/main/docs/prmers-bananantt-split

CPU prototype:
https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast

Original reference code by Yves Gallot:
https://github.com/galloty/mersenne2

The transform is split as odd * 2^m. The 2^m axis keeps the
half-real GF(p^2) packing, while the odd axis is handled with CRT/PFA
indexing.

The code includes paths for:
  - GF(M61^2)
  - GF(M31^2)
  - GF(M61^2) x GF(M31^2) CRT/Garner
  - odd radix 3 and 9
  - cooperative tile kernels for odd-radix head and tail steps

This is not production code. It is intended for validation, benchmarking,
and discussion of the mixed-radix CRT/PFA GPU layout.
*/
typedef ulong u64;
typedef uint  u32;
typedef uchar u8;

#ifndef FIELD_BITS
#define FIELD_BITS 61
#endif

#if FIELD_BITS == 31
typedef uint2 GF;
typedef uint FLIMB;
#define GF_ZERO ((GF)(0u, 0u))
#else
typedef ulong2 GF;
typedef ulong FLIMB;
#define GF_ZERO ((GF)(0ul, 0ul))
#endif

#define P61 (((u64)1 << FIELD_BITS) - 1)

inline u64 fold61_lazy(u64 x) {
    return (x & P61) + (x >> FIELD_BITS);
}

inline u64 norm61(u64 x) {
    x = (x & P61) + (x >> FIELD_BITS);
    x = (x & P61) + (x >> FIELD_BITS);
    x -= ((u64)-(x >= P61)) & P61;
    return x;
}

inline u64 add61_lazy(u64 a, u64 b) {
    u64 s = a + b;
    s = (s & P61) + (s >> FIELD_BITS);
    return s;
}

inline u64 add61(u64 a, u64 b) { return norm61(a + b); }

inline u64 sub61_lazy(u64 a, u64 b) {
    u64 d = a - b;
    d += ((u64)-(a < b)) & P61;
    return d;
}

inline u64 sub61(u64 a, u64 b) { return sub61_lazy(a, b); }

inline u64 dbl61(u64 a) {
    u64 x = a << 1;
    x = (x & P61) + (x >> FIELD_BITS);
    x -= ((u64)-(x >= P61)) & P61;
    return x;
}

#if FIELD_BITS == 31


inline u64 mul61(u64 a, u64 b) {
    const u32 aa = (u32)a;
    const u32 bb = (u32)b;
    const u32 lo = aa * bb;
    const u32 hi = mul_hi(aa, bb);
    u32 r = (lo & (u32)P61) + (lo >> 31) + (hi << 1);
    r = (r & (u32)P61) + (r >> 31);
    r -= ((u32)-(r >= (u32)P61)) & (u32)P61;
    return (u64)r;
}
inline u64 mul61_dbl(u64 a, u64 b) { return dbl61(mul61(a, b)); }
#else
inline u64 mul61(u64 a, u64 b) {
    const u32 a0 = (u32)a;
    const u32 a1 = (u32)(a >> 32);
    const u32 b0 = (u32)b;
    const u32 b1 = (u32)(b >> 32);
    u64 t = (u64)a0 * (u64)b0;
    u64 r = (t & P61) + (t >> 61);
    t = (u64)a0 * (u64)b1 + (u64)a1 * (u64)b0;
    r += ((t & 0x1ffffffful) << 32);
    r += (t >> 29);
    t = (u64)a1 * (u64)b1;
    r += (t << 3);
    r = (r & P61) + (r >> 61);
    r = (r & P61) + (r >> 61);
    return r;
}
inline u64 mul61_dbl(u64 a, u64 b) {
    const u32 a0 = (u32)a;
    const u32 a1 = (u32)(a >> 32);
    const u32 b0 = (u32)b;
    const u32 b1 = (u32)(b >> 32);
    u64 t = (u64)a0 * (u64)b0;
    u64 r = (t & P61) + (t >> 61);
    t = (u64)a0 * (u64)b1 + (u64)a1 * (u64)b0;
    r += ((t & 0x1ffffffful) << 32);
    r += (t >> 29);
    t = (u64)a1 * (u64)b1;
    r += (t << 3);
    r <<= 1;
    r = (r & P61) + (r >> 61);
    r = (r & P61) + (r >> 61);
    return r;
}
#endif

inline GF gf_add(GF x, GF y) { return (GF)(add61_lazy(x.s0, y.s0), add61_lazy(x.s1, y.s1)); }
inline GF gf_sub(GF x, GF y) { return (GF)(sub61_lazy(x.s0, y.s0), sub61_lazy(x.s1, y.s1)); }


#ifndef CRT_GF61_KARATSUBA
#define CRT_GF61_KARATSUBA 1
#endif
inline GF gf_mul(GF x, GF y) {
#if CRT_GF61_KARATSUBA
    const u64 ac = mul61(x.s0, y.s0);
    const u64 bd = mul61(x.s1, y.s1);
    const u64 sx = add61_lazy(x.s0, x.s1);
    const u64 sy = add61_lazy(y.s0, y.s1);
    const u64 abcd = mul61(sx, sy);
    return (GF)(sub61(ac, bd), sub61(sub61(abcd, ac), bd));
#else
    const u64 ac = mul61(x.s0, y.s0);
    const u64 bd = mul61(x.s1, y.s1);
    const u64 ad = mul61(x.s0, y.s1);
    const u64 bc = mul61(x.s1, y.s0);
    return (GF)(sub61(ac, bd), add61(ad, bc));
#endif
}
inline GF gf_sqr(GF x) {
    return (GF)(mul61(add61_lazy(x.s0, x.s1), sub61_lazy(x.s0, x.s1)), mul61_dbl(x.s0, x.s1));
}

inline GF gf_dbl_pair(GF x) {
    return (GF)(dbl61(x.s0), dbl61(x.s1));
}


#define GF_ADD_SUB(A, B, S, D) do { \
    const GF _gf_a = (A); \
    const GF _gf_b = (B); \
    (S) = (GF)(add61_lazy(_gf_a.s0, _gf_b.s0), add61_lazy(_gf_a.s1, _gf_b.s1)); \
    (D) = (GF)(sub61_lazy(_gf_a.s0, _gf_b.s0), sub61_lazy(_gf_a.s1, _gf_b.s1)); \
} while (0)

#define GF_DIF_MUL(A, B, W, S, D) do { \
    GF _gf_d; \
    GF_ADD_SUB((A), (B), (S), _gf_d); \
    (D) = gf_mul(_gf_d, (W)); \
} while (0)

#define GF_DIT_MUL(A, B, W, S, D) do { \
    const GF _gf_v = gf_mul((B), (W)); \
    GF_ADD_SUB((A), _gf_v, (S), (D)); \
} while (0)

inline u64 lshift61(u64 x, u32 s) {
    
    
    x = (x & P61) + (x >> (u32)FIELD_BITS);
    x = (x & P61) + (x >> (u32)FIELD_BITS);
    if (x == P61) x = 0ul;
    s %= (u32)FIELD_BITS;
    if (s == 0u) return x;
    return ((x << s) & P61) | (x >> ((u32)FIELD_BITS - s));
}
inline u64 rshift61(u64 x, u32 s) {
    s %= (u32)FIELD_BITS;
    if (s == 0u) return x;
    return ((x >> s) | ((x << ((u32)FIELD_BITS - s)) & P61));
}


inline GF gf_mul_i_fast(GF z) {
    return (GF)(sub61(0ul, z.s1), z.s0);
}
inline GF gf_mul_minus_i_fast(GF z) {
    return (GF)(z.s1, sub61(0ul, z.s0));
}
inline GF gf_mul_w8_fast(GF z) {
    const u32 sh = ((u32)FIELD_BITS - 1u) >> 1;
#if FIELD_BITS == 31
    
    return (GF)(lshift61(add61_lazy(z.s0, z.s1), sh),
                lshift61(sub61(z.s1, z.s0), sh));
#else
    
    return (GF)(lshift61(sub61(z.s1, z.s0), sh),
                lshift61(sub61(0ul, add61_lazy(z.s0, z.s1)), sh));
#endif
}
inline GF gf_mul_w8_3_fast(GF z) {
    const u32 sh = ((u32)FIELD_BITS - 1u) >> 1;
    const u64 sum = add61_lazy(z.s0, z.s1);
#if FIELD_BITS == 31
    
    return (GF)(lshift61(sub61(z.s1, z.s0), sh),
                lshift61(sub61(0ul, sum), sh));
#else
    
    return (GF)(lshift61(sum, sh),
                lshift61(sub61(z.s1, z.s0), sh));
#endif
}
inline GF gf_mul_w8_inv_fast(GF z) {
    const u32 sh = ((u32)FIELD_BITS - 1u) >> 1;
#if FIELD_BITS == 31
    
    return (GF)(lshift61(sub61(z.s0, z.s1), sh),
                lshift61(add61_lazy(z.s0, z.s1), sh));
#else
    
    return (GF)(lshift61(sub61(0ul, add61_lazy(z.s0, z.s1)), sh),
                lshift61(sub61(z.s0, z.s1), sh));
#endif
}
inline GF gf_mul_w8_inv3_fast(GF z) {
    const u32 sh = ((u32)FIELD_BITS - 1u) >> 1;
    const u64 sum = add61_lazy(z.s0, z.s1);
#if FIELD_BITS == 31
    
    return (GF)(lshift61(sub61(0ul, sum), sh),
                lshift61(sub61(z.s0, z.s1), sh));
#else
    
    return (GF)(lshift61(sub61(z.s0, z.s1), sh),
                lshift61(sum, sh));
#endif
}

inline u32 shift_from_r_on_the_fly(u32 r, const u32 lr2) {
    if (r == 0u) return 0u;
    u32 x = (r * lr2) % (u32)FIELD_BITS;
    x = ((u32)(FIELD_BITS + 1) - x) % (u32)FIELD_BITS;
    return x;
}

inline u32 shift_from_index_on_the_fly(const u32 idx, const u32 n_mask, const u32 p, const u32 lr2) {
    return shift_from_r_on_the_fly((p * idx) & n_mask, lr2);
}

__kernel void gf61_weight_first_stage_dif(__global const u64* digits,
                                          const u32 p,
                                          const u32 lr2,
                                          __global GF* a,
                                          __global const GF* twiddles,
                                          const u32 tw_offset,
                                          const u32 len,
                                          const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const u32 n_mask = len - 1u;

    const GF u = (GF)(lshift61(digits[i0], shift_from_index_on_the_fly(i0, n_mask, p, lr2)), 0ul);
    const GF v = (GF)(lshift61(digits[i1], shift_from_index_on_the_fly(i1, n_mask, p, lr2)), 0ul);

    a[i0] = gf_add(u, v);
    a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
}

__kernel void gf61_last_stage_dit_unweight(__global GF* a,
                                           __global const GF* twiddles,
                                           const u32 p,
                                           const u32 lr2,
                                           __global u64* digits,
                                           const u32 log_n,
                                           const u32 tw_offset,
                                           const u32 len,
                                           const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const u32 n_mask = (1u << log_n) - 1u;

    const GF u = a[i0];
    const GF v = gf_mul(a[i1], twiddles[tw_offset + j]);
    const GF z0 = gf_add(u, v);
    const GF z1 = gf_sub(u, v);

    digits[i0] = rshift61(norm61(z0.s0), shift_from_index_on_the_fly(i0, n_mask, p, lr2) + log_n);
    digits[i1] = rshift61(norm61(z1.s0), shift_from_index_on_the_fly(i1, n_mask, p, lr2) + log_n);
}

#define DECL_WEIGHT_FIRST_STAGE(NAME, WG) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global const u64* digits, \
          const u32 p, \
          const u32 lr2, \
          __global GF* a, \
          __global const GF* twiddles, \
          const u32 tw_offset, \
          const u32 len, \
          const u32 half_len) { \
    const u32 gid = (u32)get_global_id(0); \
    const u32 block = gid / half_len; \
    const u32 j = gid - block * half_len; \
    const u32 base = block * len; \
    const u32 i0 = base + j; \
    const u32 i1 = i0 + half_len; \
    const u32 n_mask = len - 1u; \
    const u32 r0 = (p * i0) & n_mask; \
    const u32 r1 = (p * i1) & n_mask; \
    const GF u = (GF)(lshift61(digits[i0], shift_from_r_on_the_fly(r0, lr2)), 0ul); \
    const GF v = (GF)(lshift61(digits[i1], shift_from_r_on_the_fly(r1, lr2)), 0ul); \
    a[i0] = gf_add(u, v); \
    a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]); \
}

#define DECL_LAST_STAGE_UNWEIGHT(NAME, WG) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global GF* a, \
          __global const GF* twiddles, \
          const u32 p, \
          const u32 lr2, \
          __global u64* digits, \
          const u32 log_n, \
          const u32 tw_offset, \
          const u32 len, \
          const u32 half_len) { \
    const u32 gid = (u32)get_global_id(0); \
    const u32 block = gid / half_len; \
    const u32 j = gid - block * half_len; \
    const u32 base = block * len; \
    const u32 i0 = base + j; \
    const u32 i1 = i0 + half_len; \
    const u32 n_mask = (1u << log_n) - 1u; \
    const u32 r0 = (p * i0) & n_mask; \
    const u32 r1 = (p * i1) & n_mask; \
    const GF u = a[i0]; \
    const GF v = gf_mul(a[i1], twiddles[tw_offset + j]); \
    const GF z0 = gf_add(u, v); \
    const GF z1 = gf_sub(u, v); \
    digits[i0] = rshift61(norm61(z0.s0), shift_from_r_on_the_fly(r0, lr2) + log_n); \
    digits[i1] = rshift61(norm61(z1.s0), shift_from_r_on_the_fly(r1, lr2) + log_n); \
}

DECL_WEIGHT_FIRST_STAGE(gf61_weight_first_stage_dif_wg16, 16u)
DECL_WEIGHT_FIRST_STAGE(gf61_weight_first_stage_dif_wg64, 64u)
DECL_LAST_STAGE_UNWEIGHT(gf61_last_stage_dit_unweight_wg16, 16u)
DECL_LAST_STAGE_UNWEIGHT(gf61_last_stage_dit_unweight_wg64, 64u)
#define GF61_WEIGHT_LOAD(DIGITS, IDX, R, LR2) \
    ((GF)(lshift61((DIGITS)[(IDX)], shift_from_r_on_the_fly((R), (LR2))), 0ul))

#define GF61_STORE_UNWEIGHT(DIGITS, IDX, X, R, LR2, LOGN) \
    do { \
        (DIGITS)[(IDX)] = rshift61(norm61((X).s0), shift_from_r_on_the_fly((R), (LR2)) + (LOGN)); \
    } while (0)

inline u64 gf61_unweight_digit_from_r(GF x, u32 r, u32 lr2, u32 log_n) {
    return rshift61(norm61(x.s0), shift_from_r_on_the_fly(r, lr2) + log_n);
}


#define DECL_WEIGHT_FIRST_STAGE_RADIX4_LOCAL(NAME, WG) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global const u64* digits, \
          const u32 p, \
          const u32 lr2, \
          __global GF* a, \
          __global const GF* tw_stage0, \
          __global const GF* tw_stage1, \
          __global const GF* tw_stage2, \
          const u32 tw_offset0, \
          const u32 tw_offset1, \
          const u32 tw_offset2, \
          const u32 len) { \
    const u32 gid = (u32)get_global_id(0); \
    const u32 q = len >> 3; \
    const u32 q2 = q << 1; \
    const u32 q3 = q2 + q; \
    const u32 j = gid & (q - 1u); \
    const u32 base = (gid - j) << 3; \
    const u32 i0 = base + j; \
    const u32 i1 = i0 + q; \
    const u32 i2 = i0 + q2; \
    const u32 i3 = i0 + q3; \
    const u32 i4 = i0 + (q << 2); \
    const u32 i5 = i4 + q; \
    const u32 i6 = i4 + q2; \
    const u32 i7 = i4 + q3; \
    const u32 n_mask = len - 1u; \
    const u32 r_step = (p * q) & n_mask; \
    const u32 r_step2 = (r_step << 1) & n_mask; \
    const u32 r_step4 = (r_step << 2) & n_mask; \
    const u32 r0 = (p * i0) & n_mask; \
    const GF tw10 = tw_stage1[tw_offset1 + j]; \
    const GF tw11 = tw_stage1[tw_offset1 + j + q]; \
    const GF tw20 = tw_stage2[tw_offset2 + j]; \
    GF A, B, C, D, E, F, G, H, T; \
    \
    A = GF61_WEIGHT_LOAD(digits, i0, r0, lr2); \
    B = GF61_WEIGHT_LOAD(digits, i4, (r0 + r_step4) & n_mask, lr2); \
    C = gf_add(A, B); \
    D = gf_mul(gf_sub(A, B), tw_stage0[tw_offset0 + j]); \
    \
    A = GF61_WEIGHT_LOAD(digits, i2, (r0 + r_step2) & n_mask, lr2); \
    B = GF61_WEIGHT_LOAD(digits, i6, (r0 + r_step2 + r_step4) & n_mask, lr2); \
    E = gf_add(A, B); \
    F = gf_mul(gf_sub(A, B), tw_stage0[tw_offset0 + j + q2]); \
    \
    A = gf_add(C, E); \
    B = gf_mul(gf_sub(C, E), tw10); \
    C = gf_add(D, F); \
    D = gf_mul(gf_sub(D, F), tw10); \
    \
    E = GF61_WEIGHT_LOAD(digits, i1, (r0 + r_step) & n_mask, lr2); \
    F = GF61_WEIGHT_LOAD(digits, i5, (r0 + r_step + r_step4) & n_mask, lr2); \
    G = gf_add(E, F); \
    H = gf_mul(gf_sub(E, F), tw_stage0[tw_offset0 + j + q]); \
    \
    E = GF61_WEIGHT_LOAD(digits, i3, (r0 + r_step + r_step2) & n_mask, lr2); \
    F = GF61_WEIGHT_LOAD(digits, i7, (r0 + r_step + r_step2 + r_step4) & n_mask, lr2); \
    T = gf_add(E, F); \
    F = gf_mul(gf_sub(E, F), tw_stage0[tw_offset0 + j + q3]); \
    E = T; \
    \
    T = gf_mul(gf_sub(G, E), tw11); \
    E = gf_add(G, E); \
    G = gf_add(H, F); \
    F = gf_mul(gf_sub(H, F), tw11); \
    \
    a[i0] = gf_add(A, E); \
    a[i1] = gf_mul(gf_sub(A, E), tw20); \
    a[i2] = gf_add(B, T); \
    a[i3] = gf_mul(gf_sub(B, T), tw20); \
    a[i4] = gf_add(C, G); \
    a[i5] = gf_mul(gf_sub(C, G), tw20); \
    a[i6] = gf_add(D, F); \
    a[i7] = gf_mul(gf_sub(D, F), tw20); \
}

#define DECL_LAST_STAGE_RADIX4_UNWEIGHT_LOCAL(NAME, WG) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global GF* a, \
          __global const GF* tw_stage1, \
          __global const GF* tw_stage2, \
          __global const GF* tw_stage3, \
          const u32 p, \
          const u32 lr2, \
          __global u64* digits, \
          const u32 log_n, \
          const u32 tw_offset1, \
          const u32 tw_offset2, \
          const u32 tw_offset3, \
          const u32 len) { \
    const u32 gid = (u32)get_global_id(0); \
    const u32 q = len >> 1; \
    const u32 j = gid & (q - 1u); \
    const u32 base = (gid - j) << 3; \
    const u32 i0 = base + j; \
    const u32 i1 = i0 + q; \
    const u32 i2 = i0 + len; \
    const u32 i3 = i2 + q; \
    const u32 i4 = i0 + (len << 1); \
    const u32 i5 = i4 + q; \
    const u32 i6 = i4 + len; \
    const u32 i7 = i6 + q; \
    const GF tw1 = tw_stage1[tw_offset1 + j]; \
    const GF tw20 = tw_stage2[tw_offset2 + j]; \
    const GF tw21 = tw_stage2[tw_offset2 + j + q]; \
    GF A, B, C, D, E, F, G, H, T, Z; \
    \
    A = a[i0]; \
    B = gf_mul(a[i1], tw1); \
    C = a[i2]; \
    D = gf_mul(a[i3], tw1); \
    E = gf_add(A, B); \
    B = gf_sub(A, B); \
    A = E; \
    E = gf_add(C, D); \
    D = gf_sub(C, D); \
    C = E; \
    E = gf_mul(C, tw20); \
    F = gf_mul(D, tw21); \
    C = gf_sub(A, E); \
    A = gf_add(A, E); \
    D = gf_sub(B, F); \
    B = gf_add(B, F); \
    \
    E = a[i4]; \
    F = gf_mul(a[i5], tw1); \
    G = a[i6]; \
    H = gf_mul(a[i7], tw1); \
    T = gf_add(E, F); \
    F = gf_sub(E, F); \
    E = T; \
    T = gf_add(G, H); \
    H = gf_sub(G, H); \
    G = T; \
    T = gf_mul(G, tw20); \
    G = gf_sub(E, T); \
    E = gf_add(E, T); \
    T = gf_mul(H, tw21); \
    H = gf_sub(F, T); \
    F = gf_add(F, T); \
    \
    const u32 n_mask = (1u << log_n) - 1u; \
    const u32 r_step = (p * q) & n_mask; \
    const u32 r_step2 = (r_step << 1) & n_mask; \
    const u32 r_step4 = (r_step << 2) & n_mask; \
    const u32 r0 = (p * i0) & n_mask; \
    \
    T = gf_mul(E, tw_stage3[tw_offset3 + j]); \
    Z = gf_add(A, T); \
    GF61_STORE_UNWEIGHT(digits, i0, Z, r0, lr2, log_n); \
    Z = gf_sub(A, T); \
    GF61_STORE_UNWEIGHT(digits, i4, Z, (r0 + r_step4) & n_mask, lr2, log_n); \
    \
    T = gf_mul(F, tw_stage3[tw_offset3 + j + q]); \
    Z = gf_add(B, T); \
    GF61_STORE_UNWEIGHT(digits, i1, Z, (r0 + r_step) & n_mask, lr2, log_n); \
    Z = gf_sub(B, T); \
    GF61_STORE_UNWEIGHT(digits, i5, Z, (r0 + r_step + r_step4) & n_mask, lr2, log_n); \
    \
    T = gf_mul(G, tw_stage3[tw_offset3 + j + len]); \
    Z = gf_add(C, T); \
    GF61_STORE_UNWEIGHT(digits, i2, Z, (r0 + r_step2) & n_mask, lr2, log_n); \
    Z = gf_sub(C, T); \
    GF61_STORE_UNWEIGHT(digits, i6, Z, (r0 + r_step2 + r_step4) & n_mask, lr2, log_n); \
    \
    T = gf_mul(H, tw_stage3[tw_offset3 + j + len + q]); \
    Z = gf_add(D, T); \
    GF61_STORE_UNWEIGHT(digits, i3, Z, (r0 + r_step + r_step2) & n_mask, lr2, log_n); \
    Z = gf_sub(D, T); \
    GF61_STORE_UNWEIGHT(digits, i7, Z, (r0 + r_step + r_step2 + r_step4) & n_mask, lr2, log_n); \
}

DECL_WEIGHT_FIRST_STAGE_RADIX4_LOCAL(gf61_weight_first_stage_dif_radix4_wg64, 64u)
DECL_WEIGHT_FIRST_STAGE_RADIX4_LOCAL(gf61_weight_first_stage_dif_radix4_wg128, 128u)
DECL_LAST_STAGE_RADIX4_UNWEIGHT_LOCAL(gf61_last_stage_dit_radix4_unweight_wg64, 64u)
DECL_LAST_STAGE_RADIX4_UNWEIGHT_LOCAL(gf61_last_stage_dit_radix4_unweight_wg128, 128u)

__kernel void gf61_ntt_stage_dif(__global GF* a,
                                 __global const GF* twiddles,
                                 const u32 tw_offset,
                                 const u32 len,
                                 const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const GF u = a[i0];
    const GF v = a[i1];
    a[i0] = gf_add(u, v);
    a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gf61_ntt_stage_dif_len2048(__global GF* a,
                                __global const GF* twiddles,
                                const u32 tw_offset) {
    const u32 block = (u32)get_group_id(0);
    const u32 lid = (u32)get_local_id(0);
    const u32 base = block << 11;

    for (u32 r = 0; r < 4u; ++r) {
        const u32 j = lid + (r << 8);
        const u32 i0 = base + j;
        const u32 i1 = i0 + 1024u;
        const GF u = a[i0];
        const GF v = a[i1];
        a[i0] = gf_add(u, v);
        a[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
    }
}

__kernel void gf61_ntt_stage_dit(__global GF* a,
                                 __global const GF* twiddles,
                                 const u32 tw_offset,
                                 const u32 len,
                                 const u32 half_len) {
    const u32 gid = (u32)get_global_id(0);
    const u32 block = gid / half_len;
    const u32 j = gid - block * half_len;
    const u32 base = block * len;
    const u32 i0 = base + j;
    const u32 i1 = i0 + half_len;
    const GF u = a[i0];
    const GF v = gf_mul(a[i1], twiddles[tw_offset + j]);
    a[i0] = gf_add(u, v);
    a[i1] = gf_sub(u, v);
}

inline GF gf61_mul_sub_tw(const GF a, const GF b, const GF w) {
    const u64 x0 = sub61_lazy(a.s0, b.s0);
    const u64 x1 = sub61_lazy(a.s1, b.s1);

    return (GF)(
        sub61_lazy(mul61(x0, w.s0), mul61(x1, w.s1)),
        add61_lazy(mul61(x0, w.s1), mul61(x1, w.s0))
    );
}

#define DECL_GF61_NTT_STAGE_DIF_RADIX4_OPT(NAME, WG)                    \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))                \
void NAME(__global GF* restrict a,                                      \
          __global const GF* restrict tw_stage1,                        \
          __global const GF* restrict tw_stage2,                        \
          const u32 tw_offset1,                                         \
          const u32 tw_offset2,                                         \
          const u32 len) {                                              \
    const u32 gid = (u32)get_global_id(0);                              \
                                                                        \
    const u32 q = len >> 2;                                             \
    const u32 qmask = q - 1u;                                           \
                                                                        \
    const u32 j = gid & qmask;                                          \
    const u32 i0 = ((gid & ~qmask) << 2) + j;                           \
                                                                        \
    const u32 q2 = q << 1;                                              \
    const u32 q3 = q2 + q;                                              \
                                                                        \
    __global GF* restrict ap = a + i0;                                  \
                                                                        \
    const GF w10 = tw_stage1[tw_offset1 + j];                           \
    const GF w11 = tw_stage1[tw_offset1 + j + q];                       \
    const GF w20 = tw_stage2[tw_offset2 + j];                           \
                                                                        \
    const GF x0 = ap[0];                                                \
    const GF x2 = ap[q2];                                               \
                                                                        \
    const GF u0 = gf_add(x0, x2);                                       \
    const GF u2 = gf61_mul_sub_tw(x0, x2, w10);                         \
                                                                        \
    const GF x1 = ap[q];                                                \
    const GF x3 = ap[q3];                                               \
                                                                        \
    const GF u1 = gf_add(x1, x3);                                       \
    const GF u3 = gf61_mul_sub_tw(x1, x3, w11);                         \
                                                                        \
    ap[0]  = gf_add(u0, u1);                                            \
    ap[q]  = gf61_mul_sub_tw(u0, u1, w20);                              \
    ap[q2] = gf_add(u2, u3);                                            \
    ap[q3] = gf61_mul_sub_tw(u2, u3, w20);                              \
}

DECL_GF61_NTT_STAGE_DIF_RADIX4_OPT(gf61_ntt_stage_dif_radix4, 64u)
DECL_GF61_NTT_STAGE_DIF_RADIX4_OPT(gf61_ntt_stage_dif_radix4_wg128, 128u)

#define DECL_GF61_NTT_STAGE_DIT_RADIX4_OPT(NAME, WG)                    \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))                \
void NAME(__global GF* restrict a,                                      \
          __global const GF* restrict tw_stage1,                        \
          __global const GF* restrict tw_stage2,                        \
          const u32 tw_offset1,                                         \
          const u32 tw_offset2,                                         \
          const u32 len) {                                              \
    const u32 gid = (u32)get_global_id(0);                              \
                                                                        \
    const u32 q = len >> 1;                                             \
    const u32 qmask = q - 1u;                                           \
                                                                        \
    const u32 j = gid & qmask;                                          \
    const u32 i0 = ((gid & ~qmask) << 1) + j;                           \
                                                                        \
    const u32 q2 = q << 1;                                              \
    const u32 q3 = q2 + q;                                              \
                                                                        \
    __global GF* restrict ap = a + i0;                                  \
                                                                        \
    const GF tw1  = tw_stage1[tw_offset1 + j];                          \
    const GF tw20 = tw_stage2[tw_offset2 + j];                          \
    const GF tw21 = tw_stage2[tw_offset2 + j + q];                      \
                                                                        \
    const GF x0 = ap[0];                                                \
    const GF x1 = ap[q];                                                \
    const GF x2 = ap[q2];                                               \
    const GF x3 = ap[q3];                                               \
                                                                        \
    const GF y1 = gf_mul(x1, tw1);                                      \
    const GF y3 = gf_mul(x3, tw1);                                      \
                                                                        \
    const GF u0 = gf_add(x0, y1);                                       \
    const GF u1 = gf_sub(x0, y1);                                       \
    const GF u2 = gf_add(x2, y3);                                       \
    const GF u3 = gf_sub(x2, y3);                                       \
                                                                        \
    const GF v2 = gf_mul(u2, tw20);                                     \
    const GF v3 = gf_mul(u3, tw21);                                     \
                                                                        \
    ap[0]  = gf_add(u0, v2);                                            \
    ap[q2] = gf_sub(u0, v2);                                            \
    ap[q]  = gf_add(u1, v3);                                            \
    ap[q3] = gf_sub(u1, v3);                                            \
}

DECL_GF61_NTT_STAGE_DIT_RADIX4_OPT(gf61_ntt_stage_dit_radix4, 64u)
DECL_GF61_NTT_STAGE_DIT_RADIX4_OPT(gf61_ntt_stage_dit_radix4_wg128, 128u)


#define GF61_DIF4_PRIV(A0,A1,A2,A3,W10,W11,W20) do {                  \
    const GF _x0 = (A0);                                               \
    const GF _x1 = (A1);                                               \
    const GF _x2 = (A2);                                               \
    const GF _x3 = (A3);                                               \
    const GF _u0 = gf_add(_x0, _x2);                                   \
    const GF _u2 = gf61_mul_sub_tw(_x0, _x2, (W10));                   \
    const GF _u1 = gf_add(_x1, _x3);                                   \
    const GF _u3 = gf61_mul_sub_tw(_x1, _x3, (W11));                   \
    (A0) = gf_add(_u0, _u1);                                           \
    (A1) = gf61_mul_sub_tw(_u0, _u1, (W20));                           \
    (A2) = gf_add(_u2, _u3);                                           \
    (A3) = gf61_mul_sub_tw(_u2, _u3, (W20));                           \
} while (0)

#define GF61_DIT4_PRIV(A0,A1,A2,A3,W1,W20,W21) do {                   \
    const GF _x0 = (A0);                                               \
    const GF _x1 = (A1);                                               \
    const GF _x2 = (A2);                                               \
    const GF _x3 = (A3);                                               \
    const GF _y1 = gf_mul(_x1, (W1));                                  \
    const GF _y3 = gf_mul(_x3, (W1));                                  \
    const GF _u0 = gf_add(_x0, _y1);                                   \
    const GF _u1 = gf_sub(_x0, _y1);                                   \
    const GF _u2 = gf_add(_x2, _y3);                                   \
    const GF _u3 = gf_sub(_x2, _y3);                                   \
    const GF _v2 = gf_mul(_u2, (W20));                                 \
    const GF _v3 = gf_mul(_u3, (W21));                                 \
    (A0) = gf_add(_u0, _v2);                                           \
    (A2) = gf_sub(_u0, _v2);                                           \
    (A1) = gf_add(_u1, _v3);                                           \
    (A3) = gf_sub(_u1, _v3);                                           \
} while (0)

#define DECL_GF61_NTT_STAGE_DIF_RADIX4X2_OPT(NAME, WG)                 \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))                \
void NAME(__global GF* restrict a,                                      \
          __global const GF* restrict tw_stage1,                        \
          __global const GF* restrict tw_stage2,                        \
          __global const GF* restrict tw_stage3,                        \
          __global const GF* restrict tw_stage4,                        \
          const u32 tw_offset1,                                         \
          const u32 tw_offset2,                                         \
          const u32 tw_offset3,                                         \
          const u32 tw_offset4,                                         \
          const u32 len) {                                              \
    const u32 gid = (u32)get_global_id(0);                              \
    const u32 q = len >> 4;                                             \
    const u32 qmask = q - 1u;                                           \
    const u32 j = gid & qmask;                                          \
    const u32 i0 = ((gid & ~qmask) << 4) + j;                           \
    const u32 q2 = q << 1;                                              \
    const u32 q3 = q2 + q;                                              \
    const u32 q4 = q << 2;                                              \
    const u32 q8 = q << 3;                                              \
    __global GF* restrict ap = a + i0;                                  \
    GF x0  = ap[0u  * q];                                               \
    GF x1  = ap[1u  * q];                                               \
    GF x2  = ap[2u  * q];                                               \
    GF x3  = ap[3u  * q];                                               \
    GF x4  = ap[4u  * q];                                               \
    GF x5  = ap[5u  * q];                                               \
    GF x6  = ap[6u  * q];                                               \
    GF x7  = ap[7u  * q];                                               \
    GF x8  = ap[8u  * q];                                               \
    GF x9  = ap[9u  * q];                                               \
    GF x10 = ap[10u * q];                                               \
    GF x11 = ap[11u * q];                                               \
    GF x12 = ap[12u * q];                                               \
    GF x13 = ap[13u * q];                                               \
    GF x14 = ap[14u * q];                                               \
    GF x15 = ap[15u * q];                                               \
    GF61_DIF4_PRIV(x0, x4, x8,  x12, tw_stage1[tw_offset1 + j],      tw_stage1[tw_offset1 + j + q4],      tw_stage2[tw_offset2 + j]);      \
    GF61_DIF4_PRIV(x1, x5, x9,  x13, tw_stage1[tw_offset1 + j + q],  tw_stage1[tw_offset1 + j + q + q4],  tw_stage2[tw_offset2 + j + q]);  \
    GF61_DIF4_PRIV(x2, x6, x10, x14, tw_stage1[tw_offset1 + j + q2], tw_stage1[tw_offset1 + j + q2 + q4], tw_stage2[tw_offset2 + j + q2]); \
    GF61_DIF4_PRIV(x3, x7, x11, x15, tw_stage1[tw_offset1 + j + q3], tw_stage1[tw_offset1 + j + q3 + q4], tw_stage2[tw_offset2 + j + q3]); \
    const GF w30 = tw_stage3[tw_offset3 + j];                          \
    const GF w31 = tw_stage3[tw_offset3 + j + q];                      \
    const GF w40 = tw_stage4[tw_offset4 + j];                          \
    GF61_DIF4_PRIV(x0,  x1,  x2,  x3,  w30, w31, w40);                 \
    GF61_DIF4_PRIV(x4,  x5,  x6,  x7,  w30, w31, w40);                 \
    GF61_DIF4_PRIV(x8,  x9,  x10, x11, w30, w31, w40);                 \
    GF61_DIF4_PRIV(x12, x13, x14, x15, w30, w31, w40);                 \
    ap[0u  * q] = x0;   ap[1u  * q] = x1;   ap[2u  * q] = x2;   ap[3u  * q] = x3;  \
    ap[4u  * q] = x4;   ap[5u  * q] = x5;   ap[6u  * q] = x6;   ap[7u  * q] = x7;  \
    ap[8u  * q] = x8;   ap[9u  * q] = x9;   ap[10u * q] = x10;  ap[11u * q] = x11; \
    ap[12u * q] = x12;  ap[13u * q] = x13;  ap[14u * q] = x14;  ap[15u * q] = x15; \
}

DECL_GF61_NTT_STAGE_DIF_RADIX4X2_OPT(gf61_ntt_stage_dif_radix4x2, 64u)
DECL_GF61_NTT_STAGE_DIF_RADIX4X2_OPT(gf61_ntt_stage_dif_radix4x2_wg128, 128u)

#define DECL_GF61_NTT_STAGE_DIT_RADIX4X2_OPT(NAME, WG)                 \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))                \
void NAME(__global GF* restrict a,                                      \
          __global const GF* restrict tw_stage1,                        \
          __global const GF* restrict tw_stage2,                        \
          __global const GF* restrict tw_stage3,                        \
          __global const GF* restrict tw_stage4,                        \
          const u32 tw_offset1,                                         \
          const u32 tw_offset2,                                         \
          const u32 tw_offset3,                                         \
          const u32 tw_offset4,                                         \
          const u32 len) {                                              \
    const u32 gid = (u32)get_global_id(0);                              \
    const u32 q = len >> 1;                                             \
    const u32 qmask = q - 1u;                                           \
    const u32 j = gid & qmask;                                          \
    const u32 i0 = ((gid & ~qmask) << 4) + j;                           \
    const u32 q2 = q << 1;                                              \
    const u32 q3 = q2 + q;                                              \
    const u32 q4 = q << 2;                                              \
    __global GF* restrict ap = a + i0;                                  \
    GF x0  = ap[0u  * q];                                               \
    GF x1  = ap[1u  * q];                                               \
    GF x2  = ap[2u  * q];                                               \
    GF x3  = ap[3u  * q];                                               \
    GF x4  = ap[4u  * q];                                               \
    GF x5  = ap[5u  * q];                                               \
    GF x6  = ap[6u  * q];                                               \
    GF x7  = ap[7u  * q];                                               \
    GF x8  = ap[8u  * q];                                               \
    GF x9  = ap[9u  * q];                                               \
    GF x10 = ap[10u * q];                                               \
    GF x11 = ap[11u * q];                                               \
    GF x12 = ap[12u * q];                                               \
    GF x13 = ap[13u * q];                                               \
    GF x14 = ap[14u * q];                                               \
    GF x15 = ap[15u * q];                                               \
    const GF w10 = tw_stage1[tw_offset1 + j];                           \
    const GF w20 = tw_stage2[tw_offset2 + j];                           \
    const GF w21 = tw_stage2[tw_offset2 + j + q];                       \
    GF61_DIT4_PRIV(x0,  x1,  x2,  x3,  w10, w20, w21);                 \
    GF61_DIT4_PRIV(x4,  x5,  x6,  x7,  w10, w20, w21);                 \
    GF61_DIT4_PRIV(x8,  x9,  x10, x11, w10, w20, w21);                 \
    GF61_DIT4_PRIV(x12, x13, x14, x15, w10, w20, w21);                 \
    GF61_DIT4_PRIV(x0, x4, x8,  x12, tw_stage3[tw_offset3 + j],      tw_stage4[tw_offset4 + j],      tw_stage4[tw_offset4 + j + q4]);      \
    GF61_DIT4_PRIV(x1, x5, x9,  x13, tw_stage3[tw_offset3 + j + q],  tw_stage4[tw_offset4 + j + q],  tw_stage4[tw_offset4 + j + q + q4]);  \
    GF61_DIT4_PRIV(x2, x6, x10, x14, tw_stage3[tw_offset3 + j + q2], tw_stage4[tw_offset4 + j + q2], tw_stage4[tw_offset4 + j + q2 + q4]); \
    GF61_DIT4_PRIV(x3, x7, x11, x15, tw_stage3[tw_offset3 + j + q3], tw_stage4[tw_offset4 + j + q3], tw_stage4[tw_offset4 + j + q3 + q4]); \
    ap[0u  * q] = x0;   ap[1u  * q] = x1;   ap[2u  * q] = x2;   ap[3u  * q] = x3;  \
    ap[4u  * q] = x4;   ap[5u  * q] = x5;   ap[6u  * q] = x6;   ap[7u  * q] = x7;  \
    ap[8u  * q] = x8;   ap[9u  * q] = x9;   ap[10u * q] = x10;  ap[11u * q] = x11; \
    ap[12u * q] = x12;  ap[13u * q] = x13;  ap[14u * q] = x14;  ap[15u * q] = x15; \
}

DECL_GF61_NTT_STAGE_DIT_RADIX4X2_OPT(gf61_ntt_stage_dit_radix4x2, 64u)
DECL_GF61_NTT_STAGE_DIT_RADIX4X2_OPT(gf61_ntt_stage_dit_radix4x2_wg128, 128u)
__kernel void gf61_pointwise_sqr(__global GF* a, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) a[gid] = gf_sqr(a[gid]);
}

__kernel void gf61_mul_small_digits(__global u64* digits, const u32 k, const u32 n) {
    const u32 gid = (u32)get_global_id(0);
    if (gid < n) digits[gid] *= (u64)k;
}

inline void local_stage_dif_pow2(__local GF* x,
                                 __global const GF* twiddles,
                                 const u32 chunk,
                                 const u32 len,
                                 const u32 lid,
                                 const u32 lsize) {
    const u32 half_len_local = len >> 1;
    const u32 tw_offset = half_len_local - 1u;
    const u32 butterflies = chunk >> 1;

    for (u32 t = lid; t < butterflies; t += lsize) {
        const u32 block = t / half_len_local;
        const u32 j = t - block * half_len_local;
        const u32 i0 = block * len + j;
        const u32 i1 = i0 + half_len_local;
        const GF u = x[i0];
        const GF v = x[i1];
        x[i0] = gf_add(u, v);
        x[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
    }
}

inline void local_stage_dit_pow2(__local GF* x,
                                 __global const GF* twiddles,
                                 const u32 chunk,
                                 const u32 len,
                                 const u32 lid,
                                 const u32 lsize) {
    const u32 half_len_local = len >> 1;
    const u32 tw_offset = half_len_local - 1u;
    const u32 butterflies = chunk >> 1;

    for (u32 t = lid; t < butterflies; t += lsize) {
        const u32 block = t / half_len_local;
        const u32 j = t - block * half_len_local;
        const u32 i0 = block * len + j;
        const u32 i1 = i0 + half_len_local;
        const GF u = x[i0];
        const GF v = gf_mul(x[i1], twiddles[tw_offset + j]);
        x[i0] = gf_add(u, v);
        x[i1] = gf_sub(u, v);
    }
}

inline void local_stage_dif_radix4_pow2(__local GF* x,
                                        __global const GF* twiddles,
                                        const u32 n,
                                        const u32 len,
                                        const u32 lid,
                                        const u32 wg) {
    const u32 quarter = len >> 2;
    const u32 total = n >> 2;
    const u32 tw_offset1 = (len >> 1) - 1u;
    const u32 tw_offset2 = quarter - 1u;

    for (u32 t = lid; t < total; t += wg) {
        const u32 block = t / quarter;
        const u32 j = t - block * quarter;
        const u32 base = block * len;

        const u32 i0 = base + j;
        const u32 i1 = i0 + quarter;
        const u32 i2 = i1 + quarter;
        const u32 i3 = i2 + quarter;

        const GF a0 = x[i0];
        const GF a1 = x[i1];
        const GF a2 = x[i2];
        const GF a3 = x[i3];

        const GF b0 = gf_add(a0, a2);
        const GF b2 = gf_mul(gf_sub(a0, a2), twiddles[tw_offset1 + j]);
        const GF b1 = gf_add(a1, a3);
        const GF b3 = gf_mul(gf_sub(a1, a3), twiddles[tw_offset1 + j + quarter]);

        x[i0] = gf_add(b0, b1);
        x[i1] = gf_mul(gf_sub(b0, b1), twiddles[tw_offset2 + j]);
        x[i2] = gf_add(b2, b3);
        x[i3] = gf_mul(gf_sub(b2, b3), twiddles[tw_offset2 + j]);
    }
}

inline void local_stage_dit_radix4_pow2(__local GF* x,
                                        __global const GF* twiddles,
                                        const u32 n,
                                        const u32 len,
                                        const u32 lid,
                                        const u32 wg) {
    const u32 half_len_local = len >> 1;
    const u32 total = n >> 2;
    const u32 tw_offset1 = half_len_local - 1u;
    const u32 tw_offset2 = len - 1u;

    for (u32 t = lid; t < total; t += wg) {
        const u32 block = t / half_len_local;
        const u32 j = t - block * half_len_local;
        const u32 base = block * (len << 1);

        const u32 i0 = base + j;
        const u32 i1 = i0 + half_len_local;
        const u32 i2 = i0 + len;
        const u32 i3 = i2 + half_len_local;

        const GF x0 = x[i0];
        const GF x1 = x[i1];
        const GF x2 = x[i2];
        const GF x3 = x[i3];

        const GF y0 = gf_add(x0, gf_mul(x1, twiddles[tw_offset1 + j]));
        const GF y1 = gf_sub(x0, gf_mul(x1, twiddles[tw_offset1 + j]));
        const GF y2 = gf_add(x2, gf_mul(x3, twiddles[tw_offset1 + j]));
        const GF y3 = gf_sub(x2, gf_mul(x3, twiddles[tw_offset1 + j]));

        x[i0] = gf_add(y0, gf_mul(y2, twiddles[tw_offset2 + j]));
        x[i2] = gf_sub(y0, gf_mul(y2, twiddles[tw_offset2 + j]));
        x[i1] = gf_add(y1, gf_mul(y3, twiddles[tw_offset2 + j + half_len_local]));
        x[i3] = gf_sub(y1, gf_mul(y3, twiddles[tw_offset2 + j + half_len_local]));
    }
}


static inline __attribute__((always_inline)) void local_stage_dif_radix8_pow2(__local GF* x,
                                        __global const GF* tw,
                                        const u32 chunk,
                                        const u32 len,
                                        const u32 lid,
                                        const u32 wg) {
    const u32 q = len >> 3;
    const u32 total = chunk >> 3;
    const u32 o1 = (len >> 1) - 1u;
    const u32 o2 = (len >> 2) - 1u;
    const u32 o3 = q - 1u;
    for (u32 t = lid; t < total; t += wg) {
        const u32 block = t / q;
        const u32 j = t - block * q;
        const u32 base = block * len;
        const u32 i0 = base + j;
        const u32 i1 = i0 + q;
        const u32 i2 = i1 + q;
        const u32 i3 = i2 + q;
        const u32 i4 = i3 + q;
        const u32 i5 = i4 + q;
        const u32 i6 = i5 + q;
        const u32 i7 = i6 + q;

        GF x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
        GF x4 = x[i4], x5 = x[i5], x6 = x[i6], x7 = x[i7];

        if (len == 8u) {
            
            
            GF t0 = gf_add(x0, x4); GF t4 = gf_sub(x0, x4);
            GF t1 = gf_add(x1, x5); GF t5 = gf_mul_w8_fast(gf_sub(x1, x5));
#if FIELD_BITS == 31
            GF t2 = gf_add(x2, x6); GF t6 = gf_mul_minus_i_fast(gf_sub(x2, x6));
#else
            GF t2 = gf_add(x2, x6); GF t6 = gf_mul_i_fast(gf_sub(x2, x6));
#endif
            GF t3 = gf_add(x3, x7); GF t7 = gf_mul_w8_3_fast(gf_sub(x3, x7));

            x0 = gf_add(t0, t2); x2 = gf_sub(t0, t2);
#if FIELD_BITS == 31
            x1 = gf_add(t1, t3); x3 = gf_mul_minus_i_fast(gf_sub(t1, t3));
#else
            x1 = gf_add(t1, t3); x3 = gf_mul_i_fast(gf_sub(t1, t3));
#endif
            x4 = gf_add(t4, t6); x6 = gf_sub(t4, t6);
#if FIELD_BITS == 31
            x5 = gf_add(t5, t7); x7 = gf_mul_minus_i_fast(gf_sub(t5, t7));
#else
            x5 = gf_add(t5, t7); x7 = gf_mul_i_fast(gf_sub(t5, t7));
#endif

            t0 = gf_add(x0, x1); t1 = gf_sub(x0, x1);
            t2 = gf_add(x2, x3); t3 = gf_sub(x2, x3);
            t4 = gf_add(x4, x5); t5 = gf_sub(x4, x5);
            t6 = gf_add(x6, x7); t7 = gf_sub(x6, x7);

            x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
            x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
            continue;
        }

        GF t0 = gf_add(x0, x4); GF t4 = gf_mul(gf_sub(x0, x4), tw[o1 + j]);
        GF t1 = gf_add(x1, x5); GF t5 = gf_mul(gf_sub(x1, x5), tw[o1 + j + q]);
        GF t2 = gf_add(x2, x6); GF t6 = gf_mul(gf_sub(x2, x6), tw[o1 + j + (q << 1)]);
        GF t3 = gf_add(x3, x7); GF t7 = gf_mul(gf_sub(x3, x7), tw[o1 + j + (q * 3u)]);

        x0 = gf_add(t0, t2); x2 = gf_mul(gf_sub(t0, t2), tw[o2 + j]);
        x1 = gf_add(t1, t3); x3 = gf_mul(gf_sub(t1, t3), tw[o2 + j + q]);
        x4 = gf_add(t4, t6); x6 = gf_mul(gf_sub(t4, t6), tw[o2 + j]);
        x5 = gf_add(t5, t7); x7 = gf_mul(gf_sub(t5, t7), tw[o2 + j + q]);

        t0 = gf_add(x0, x1); t1 = gf_mul(gf_sub(x0, x1), tw[o3 + j]);
        t2 = gf_add(x2, x3); t3 = gf_mul(gf_sub(x2, x3), tw[o3 + j]);
        t4 = gf_add(x4, x5); t5 = gf_mul(gf_sub(x4, x5), tw[o3 + j]);
        t6 = gf_add(x6, x7); t7 = gf_mul(gf_sub(x6, x7), tw[o3 + j]);

        x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
        x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
    }
}

static inline __attribute__((always_inline))
void local_stage_dif_radix8_len8_512_wg64(__local GF* x, const u32 lid)
{
    const u32 base = lid << 3;

    GF x0 = x[base + 0u];
    GF x1 = x[base + 1u];
    GF x2 = x[base + 2u];
    GF x3 = x[base + 3u];
    GF x4 = x[base + 4u];
    GF x5 = x[base + 5u];
    GF x6 = x[base + 6u];
    GF x7 = x[base + 7u];

    GF a, b;

    a = gf_add(x0, x4); b = gf_sub(x0, x4); x0 = a; x4 = b;
    a = gf_add(x1, x5); b = gf_sub(x1, x5); x1 = a; x5 = gf_mul_w8_fast(b);

    a = gf_add(x2, x6); b = gf_sub(x2, x6); x2 = a;
#if FIELD_BITS == 31
    x6 = gf_mul_minus_i_fast(b);
#else
    x6 = gf_mul_i_fast(b);
#endif

    a = gf_add(x3, x7); b = gf_sub(x3, x7); x3 = a; x7 = gf_mul_w8_3_fast(b);

    a = gf_add(x0, x2); b = gf_sub(x0, x2); x0 = a; x2 = b;

    a = gf_add(x1, x3); b = gf_sub(x1, x3); x1 = a;
#if FIELD_BITS == 31
    x3 = gf_mul_minus_i_fast(b);
#else
    x3 = gf_mul_i_fast(b);
#endif

    a = gf_add(x4, x6); b = gf_sub(x4, x6); x4 = a; x6 = b;

    a = gf_add(x5, x7); b = gf_sub(x5, x7); x5 = a;
#if FIELD_BITS == 31
    x7 = gf_mul_minus_i_fast(b);
#else
    x7 = gf_mul_i_fast(b);
#endif

    a = gf_add(x0, x1); b = gf_sub(x0, x1); x[base + 0u] = a; x[base + 1u] = b;
    a = gf_add(x2, x3); b = gf_sub(x2, x3); x[base + 2u] = a; x[base + 3u] = b;
    a = gf_add(x4, x5); b = gf_sub(x4, x5); x[base + 4u] = a; x[base + 5u] = b;
    a = gf_add(x6, x7); b = gf_sub(x6, x7); x[base + 6u] = a; x[base + 7u] = b;
}

static inline __attribute__((always_inline)) void local_stage_dit_radix8_pow2(__local GF* x,
                                        __global const GF* tw,
                                        const u32 chunk,
                                        const u32 len,
                                        const u32 lid,
                                        const u32 wg) {
    const u32 q = len >> 1;
    const u32 total = chunk >> 3;
    const u32 o1 = q - 1u;
    const u32 o2 = len - 1u;
    const u32 o3 = (len << 1) - 1u;
    for (u32 t = lid; t < total; t += wg) {
        const u32 block = t / q;
        const u32 j = t - block * q;
        const u32 base = block * (len << 2);
        const u32 i0 = base + j;
        const u32 i1 = i0 + q;
        const u32 i2 = i1 + q;
        const u32 i3 = i2 + q;
        const u32 i4 = i3 + q;
        const u32 i5 = i4 + q;
        const u32 i6 = i5 + q;
        const u32 i7 = i6 + q;

        GF x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
        GF x4 = x[i4], x5 = x[i5], x6 = x[i6], x7 = x[i7];

        if (len == 2u) {
            
            GF y0 = x1;
            GF y1 = x3;
            GF y2 = x5;
            GF y3 = x7;
            x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
            x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
            x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
            x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);

            y0 = x2;
#if FIELD_BITS == 31
            y1 = gf_mul_i_fast(x3);
#else
            y1 = gf_mul_minus_i_fast(x3);
#endif
            GF y4 = x6;
#if FIELD_BITS == 31
            GF y5 = gf_mul_i_fast(x7);
#else
            GF y5 = gf_mul_minus_i_fast(x7);
#endif
            x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
            x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
            x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
            x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);

            y0 = x4;
            y1 = gf_mul_w8_inv_fast(x5);
#if FIELD_BITS == 31
            y2 = gf_mul_i_fast(x6);
#else
            y2 = gf_mul_minus_i_fast(x6);
#endif
            y3 = gf_mul_w8_inv3_fast(x7);
            x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
            x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
            x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
            x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);

            x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
            x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
            continue;
        }

        GF y0 = gf_mul(x1, tw[o1 + j]);
        GF y1 = gf_mul(x3, tw[o1 + j]);
        GF y2 = gf_mul(x5, tw[o1 + j]);
        GF y3 = gf_mul(x7, tw[o1 + j]);
        x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
        x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
        x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
        x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);

        y0 = gf_mul(x2, tw[o2 + j]);
        y1 = gf_mul(x3, tw[o2 + j + q]);
        GF y4 = gf_mul(x6, tw[o2 + j]);
        GF y5 = gf_mul(x7, tw[o2 + j + q]);
        x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
        x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
        x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
        x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);

        y0 = gf_mul(x4, tw[o3 + j]);
        y1 = gf_mul(x5, tw[o3 + j + q]);
        y2 = gf_mul(x6, tw[o3 + j + (q << 1)]);
        y3 = gf_mul(x7, tw[o3 + j + (q * 3u)]);
        x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
        x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
        x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
        x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);

        x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
        x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
    }
}

static inline __attribute__((always_inline))
void local_stage_dif8_sqr_dit8_512_wg64(__local GF* restrict x,
                                           __local GF* restrict twl,
                                           __global const GF* restrict twi,
                                           const u32 lid)
{
    if (lid < 56u) twl[lid] = twi[7u + lid];
    const u32 base = lid << 3;

    GF x0 = x[base + 0u]; GF x1 = x[base + 1u]; GF x2 = x[base + 2u]; GF x3 = x[base + 3u];
    GF x4 = x[base + 4u]; GF x5 = x[base + 5u]; GF x6 = x[base + 6u]; GF x7 = x[base + 7u];
    GF b;

    GF_ADD_SUB(x0, x4, x0, x4);
    GF_ADD_SUB(x1, x5, x1, b); x5 = gf_mul_w8_fast(b);
    GF_ADD_SUB(x2, x6, x2, b);
#if FIELD_BITS == 31
    x6 = gf_mul_minus_i_fast(b);
#else
    x6 = gf_mul_i_fast(b);
#endif
    GF_ADD_SUB(x3, x7, x3, b); x7 = gf_mul_w8_3_fast(b);

    GF_ADD_SUB(x0, x2, x0, x2);
    GF_ADD_SUB(x1, x3, x1, b);
#if FIELD_BITS == 31
    x3 = gf_mul_minus_i_fast(b);
#else
    x3 = gf_mul_i_fast(b);
#endif
    GF_ADD_SUB(x4, x6, x4, x6);
    GF_ADD_SUB(x5, x7, x5, b);
#if FIELD_BITS == 31
    x7 = gf_mul_minus_i_fast(b);
#else
    x7 = gf_mul_i_fast(b);
#endif

    GF_ADD_SUB(x0, x1, x0, x1);
    GF_ADD_SUB(x2, x3, x2, x3);
    GF_ADD_SUB(x4, x5, x4, x5);
    GF_ADD_SUB(x6, x7, x6, x7);

    x0 = gf_sqr(x0); x1 = gf_sqr(x1); x2 = gf_sqr(x2); x3 = gf_sqr(x3);
    x4 = gf_sqr(x4); x5 = gf_sqr(x5); x6 = gf_sqr(x6); x7 = gf_sqr(x7);

    GF y0 = x1; GF y1 = x3; GF y2 = x5; GF y3 = x7;
    GF_ADD_SUB(x0, y0, x0, x1);
    GF_ADD_SUB(x2, y1, x2, x3);
    GF_ADD_SUB(x4, y2, x4, x5);
    GF_ADD_SUB(x6, y3, x6, x7);

    y0 = x2;
#if FIELD_BITS == 31
    y1 = gf_mul_i_fast(x3);
#else
    y1 = gf_mul_minus_i_fast(x3);
#endif
    GF y4 = x6;
#if FIELD_BITS == 31
    GF y5 = gf_mul_i_fast(x7);
#else
    GF y5 = gf_mul_minus_i_fast(x7);
#endif
    GF_ADD_SUB(x0, y0, x0, x2);
    GF_ADD_SUB(x1, y1, x1, x3);
    GF_ADD_SUB(x4, y4, x4, x6);
    GF_ADD_SUB(x5, y5, x5, x7);

    y0 = x4;
    y1 = gf_mul_w8_inv_fast(x5);
#if FIELD_BITS == 31
    y2 = gf_mul_i_fast(x6);
#else
    y2 = gf_mul_minus_i_fast(x6);
#endif
    y3 = gf_mul_w8_inv3_fast(x7);
    GF_ADD_SUB(x0, y0, x0, x4);
    GF_ADD_SUB(x1, y1, x1, x5);
    GF_ADD_SUB(x2, y2, x2, x6);
    GF_ADD_SUB(x3, y3, x3, x7);

    x[base + 0u] = x0; x[base + 1u] = x1; x[base + 2u] = x2; x[base + 3u] = x3;
    x[base + 4u] = x4; x[base + 5u] = x5; x[base + 6u] = x6; x[base + 7u] = x7;
}


static inline __attribute__((always_inline))
void local_stage_dif_radix8_len512_gload_512_wg64(__global const GF* restrict a,
                                                  __local GF* restrict x,
                                                  __local GF* restrict twl,
                                                  __global const GF* restrict tw,
                                                  const u32 gbase,
                                                  const u32 lid)
{
    if (lid < 56u) twl[lid] = tw[7u + lid];
    const u32 j = lid;
    const u32 i0 = j;
    const u32 i1 = i0 +  64u;
    const u32 i2 = i0 + 128u;
    const u32 i3 = i0 + 192u;
    const u32 i4 = i0 + 256u;
    const u32 i5 = i0 + 320u;
    const u32 i6 = i0 + 384u;
    const u32 i7 = i0 + 448u;

    GF x0 = a[gbase + i0];
    GF x1 = a[gbase + i1];
    GF x2 = a[gbase + i2];
    GF x3 = a[gbase + i3];
    GF x4 = a[gbase + i4];
    GF x5 = a[gbase + i5];
    GF x6 = a[gbase + i6];
    GF x7 = a[gbase + i7];

    GF t0, t1, t2, t3, t4, t5, t6, t7;
    const GF w255_0 = tw[255u + j];
    const GF w255_1 = tw[255u + j +  64u];
    const GF w255_2 = tw[255u + j + 128u];
    const GF w255_3 = tw[255u + j + 192u];
    const GF w127_0 = tw[127u + j];
    const GF w127_1 = tw[127u + j +  64u];
    const GF w63    = tw[ 63u + j];

    GF_DIF_MUL(x0, x4, w255_0, t0, t4);
    GF_DIF_MUL(x1, x5, w255_1, t1, t5);
    GF_DIF_MUL(x2, x6, w255_2, t2, t6);
    GF_DIF_MUL(x3, x7, w255_3, t3, t7);

    GF_DIF_MUL(t0, t2, w127_0, x0, x2);
    GF_DIF_MUL(t1, t3, w127_1, x1, x3);
    GF_DIF_MUL(t4, t6, w127_0, x4, x6);
    GF_DIF_MUL(t5, t7, w127_1, x5, x7);

    GF_DIF_MUL(x0, x1, w63, t0, t1);
    GF_DIF_MUL(x2, x3, w63, t2, t3);
    GF_DIF_MUL(x4, x5, w63, t4, t5);
    GF_DIF_MUL(x6, x7, w63, t6, t7);

    x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
    x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
}


static inline __attribute__((always_inline))
void local_stage_dit_radix8_len128_gstore_512_wg64(__local GF* restrict x,
                                                   __global GF* restrict a,
                                                   __global const GF* restrict tw,
                                                   const u32 gbase,
                                                   const u32 lid)
{
    const u32 j = lid;

    const u32 i0 = j;
    const u32 i1 = i0 +  64u;
    const u32 i2 = i0 + 128u;
    const u32 i3 = i0 + 192u;
    const u32 i4 = i0 + 256u;
    const u32 i5 = i0 + 320u;
    const u32 i6 = i0 + 384u;
    const u32 i7 = i0 + 448u;

    GF x0 = x[i0]; GF x1 = x[i1]; GF x2 = x[i2]; GF x3 = x[i3];
    GF x4 = x[i4]; GF x5 = x[i5]; GF x6 = x[i6]; GF x7 = x[i7];

    const GF w63    = tw[ 63u + j];
    const GF w127_0 = tw[127u + j];
    const GF w127_1 = tw[127u + j +  64u];
    const GF w255_0 = tw[255u + j];
    const GF w255_1 = tw[255u + j +  64u];
    const GF w255_2 = tw[255u + j + 128u];
    const GF w255_3 = tw[255u + j + 192u];

    GF_DIT_MUL(x0, x1, w63, x0, x1);
    GF_DIT_MUL(x2, x3, w63, x2, x3);
    GF_DIT_MUL(x4, x5, w63, x4, x5);
    GF_DIT_MUL(x6, x7, w63, x6, x7);

    GF_DIT_MUL(x0, x2, w127_0, x0, x2);
    GF_DIT_MUL(x1, x3, w127_1, x1, x3);
    GF_DIT_MUL(x4, x6, w127_0, x4, x6);
    GF_DIT_MUL(x5, x7, w127_1, x5, x7);

    GF_DIT_MUL(x0, x4, w255_0, x0, x4);
    GF_DIT_MUL(x1, x5, w255_1, x1, x5);
    GF_DIT_MUL(x2, x6, w255_2, x2, x6);
    GF_DIT_MUL(x3, x7, w255_3, x3, x7);

    a[gbase + i0] = x0; a[gbase + i1] = x1; a[gbase + i2] = x2; a[gbase + i3] = x3;
    a[gbase + i4] = x4; a[gbase + i5] = x5; a[gbase + i6] = x6; a[gbase + i7] = x7;
}


#define DECL_GF61_CENTER_FUSED(NAME, CHUNK, WG) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global GF* a, __global const GF* tw_fwd, __global const GF* tw_inv) { \
    const u32 lid = (u32)get_local_id(0); \
    const u32 group = (u32)get_group_id(0); \
    const u32 base = group * (CHUNK); \
    __local GF x[(CHUNK)]; \
    for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = a[base + i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    u32 f_len = (CHUNK); \
    for (; f_len >= 8u; f_len >>= 3) { \
        local_stage_dif_radix8_pow2(x, tw_fwd, (CHUNK), f_len, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (f_len == 8u) { f_len = 1u; break; } \
    } \
    if (f_len == 4u) { \
        local_stage_dif_radix4_pow2(x, tw_fwd, (CHUNK), 4u, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } else if (f_len == 2u) { \
        local_stage_dif_pow2(x, tw_fwd, (CHUNK), 2u, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = gf_sqr(x[i]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    u32 i_len = 2u; \
    for (; (i_len << 2) <= (CHUNK); i_len <<= 3) { \
        local_stage_dit_radix8_pow2(x, tw_inv, (CHUNK), i_len, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if ((i_len << 1) <= (CHUNK)) { \
        local_stage_dit_radix4_pow2(x, tw_inv, (CHUNK), i_len, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        i_len <<= 2; \
    } \
    if (i_len <= (CHUNK)) { \
        local_stage_dit_pow2(x, tw_inv, (CHUNK), i_len, lid, (WG)); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (u32 i = lid; i < (CHUNK); i += (WG)) a[base + i] = x[i]; \
}

#ifndef ENABLE_GF61_CENTER4096
#define ENABLE_GF61_CENTER4096 0
#endif

DECL_GF61_CENTER_FUSED(gf61_center_fused_8,     8u,   8u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_16,   16u,   8u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_32,   32u,  16u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_64,   64u,  16u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_128, 128u, 32u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_256,  256u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_512,  512u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_1024, 1024u, 64u)
DECL_GF61_CENTER_FUSED(gf61_center_fused_2048, 2048u, 64u)
#if ENABLE_GF61_CENTER4096
DECL_GF61_CENTER_FUSED(gf61_center_fused_4096, 4096u, 64u)
#endif

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_center_fused_256_explicit(__global GF* a,
                                    __global const GF* tw_fwd,
                                    __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 256u;

    __local GF x[256];

    for (u32 i = lid; i < 256u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dif_radix4_pow2(x, tw_fwd, 256u, 256u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(x, tw_fwd, 256u, 64u,  lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(x, tw_fwd, 256u, 16u,  lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(x, tw_fwd, 256u, 4u,   lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);

    for (u32 i = lid; i < 256u; i += 64u) x[i] = gf_sqr(x[i]);
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dit_radix4_pow2(x, tw_inv, 256u, 2u,   lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(x, tw_inv, 256u, 8u,   lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(x, tw_inv, 256u, 32u,  lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(x, tw_inv, 256u, 128u, lid, 64u); barrier(CLK_LOCAL_MEM_FENCE);

    for (u32 i = lid; i < 256u; i += 64u) a[base + i] = x[i];
}

#define DECL_GF61_FORWARD_BRIDGE(NAME, CHUNK, WG, STOP_CHUNK) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global GF* a, __global const GF* tw_fwd) { \
    const u32 lid = (u32)get_local_id(0); \
    const u32 group = (u32)get_group_id(0); \
    const u32 base = group * (CHUNK); \
    __local GF x[(CHUNK)]; \
    for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = a[base + i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (u32 len = (CHUNK); len > (STOP_CHUNK); ) { \
        if ((len >> 2) >= (STOP_CHUNK)) { \
            local_stage_dif_radix4_pow2(x, tw_fwd, (CHUNK), len, lid, (WG)); \
            len >>= 2; \
        } else { \
            local_stage_dif_pow2(x, tw_fwd, (CHUNK), len, lid, (WG)); \
            len >>= 1; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (u32 i = lid; i < (CHUNK); i += (WG)) a[base + i] = x[i]; \
}

#define DECL_GF61_INVERSE_BRIDGE(NAME, CHUNK, WG, START_CHUNK) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global GF* a, __global const GF* tw_inv) { \
    const u32 lid = (u32)get_local_id(0); \
    const u32 group = (u32)get_group_id(0); \
    const u32 base = group * (CHUNK); \
    __local GF x[(CHUNK)]; \
    for (u32 i = lid; i < (CHUNK); i += (WG)) x[i] = a[base + i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (u32 len = ((START_CHUNK) << 1); len <= (CHUNK); ) { \
        if ((len << 1) <= (CHUNK)) { \
            local_stage_dit_radix4_pow2(x, tw_inv, (CHUNK), len, lid, (WG)); \
            len <<= 2; \
        } else { \
            local_stage_dit_pow2(x, tw_inv, (CHUNK), len, lid, (WG)); \
            len <<= 1; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (u32 i = lid; i < (CHUNK); i += (WG)) a[base + i] = x[i]; \
}

DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_64_to_16, 64u, 16u, 16u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_16_to_64, 64u, 16u, 16u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_256_to_64, 256u, 64u, 64u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_64_to_256, 256u, 64u, 64u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_512_to_256, 512u, 64u, 256u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_256_to_512, 512u, 64u, 256u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_1024_to_512, 1024u, 64u, 512u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_512_to_1024, 1024u, 64u, 512u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_1024_to_256, 1024u, 64u, 256u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_256_to_1024, 1024u, 64u, 256u)
DECL_GF61_FORWARD_BRIDGE(gf61_forward_bridge_2048_to_256, 2048u, 64u, 256u)
DECL_GF61_INVERSE_BRIDGE(gf61_inverse_bridge_256_to_2048, 2048u, 64u, 256u)

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_forward_ext_1024_to_256_explicit(__global GF* a, __global const GF* tw_fwd) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 1024u;
    __local GF x[1024u];

    for (u32 i = lid; i < 1024u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(x, tw_fwd, 1024u, 1024u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 1024u; i += 64u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gf61_forward_ext_1024_to_256_explicit_wg128(__global GF* a, __global const GF* tw_fwd) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 1024u;
    __local GF x[1024u];

    for (u32 i = lid; i < 1024u; i += 128u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(x, tw_fwd, 1024u, 1024u, lid, 128u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 1024u; i += 128u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_inverse_ext_256_to_1024_explicit(__global GF* a, __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 1024u;
    __local GF x[1024u];

    for (u32 i = lid; i < 1024u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(x, tw_inv, 1024u, 512u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 1024u; i += 64u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gf61_inverse_ext_256_to_1024_explicit_wg128(__global GF* a, __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 1024u;
    __local GF x[1024u];

    for (u32 i = lid; i < 1024u; i += 128u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(x, tw_inv, 1024u, 512u, lid, 128u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 1024u; i += 128u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_forward_ext_2048_to_256_explicit(__global GF* a, __global const GF* tw_fwd) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 2048u;
    __local GF x[2048u];

    for (u32 i = lid; i < 2048u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_pow2(x, tw_fwd, 2048u, 2048u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(x, tw_fwd, 2048u, 1024u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 2048u; i += 64u) a[base + i] = x[i];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gf61_inverse_ext_256_to_2048_explicit(__global GF* a, __global const GF* tw_inv) {
    const u32 lid = (u32)get_local_id(0);
    const u32 group = (u32)get_group_id(0);
    const u32 base = group * 2048u;
    __local GF x[2048u];

    for (u32 i = lid; i < 2048u; i += 64u) x[i] = a[base + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(x, tw_inv, 2048u, 512u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_pow2(x, tw_inv, 2048u, 2048u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < 2048u; i += 64u) a[base + i] = x[i];
}

inline u64 digit_mask(const u32 w) {
    return (w >= 64u) ? ~0ul : ((((u64)1) << w) - 1ul);
}

__kernel void gf61_carry_first_pass(__global u64* digits,
                                    __global const u8* widths,
                                    __global u64* carry_out,
                                    __global u32* pending,
                                    const u32 n) {
    const u32 i = (u32)get_global_id(0);
    if (i >= n) return;

    if (i == 0u) pending[0] = 0u;

    const u32 w = (u32)widths[i];
    const u64 total = digits[i];

    digits[i] = total & digit_mask(w);
    carry_out[(i + 1u) & (n - 1u)] = total >> w;
}

__kernel void gf61_carry_clear_pending(__global u32* pending) {
    if (get_global_id(0) == 0) pending[0] = 0u;
}

__kernel void gf61_carry_pass(__global u64* digits,
                              __global const u8* widths,
                              __global const u64* carry_in,
                              __global u64* carry_out,
                              __global u32* pending,
                              const u32 n,
                              const u32 set_pending) {
    const u32 i = (u32)get_global_id(0);
    if (i >= n) return;

    const u32 w = (u32)widths[i];
    const u64 total = digits[i] + carry_in[i];
    digits[i] = total & digit_mask(w);

    const u64 c = total >> w;
    carry_out[(i + 1u) & (n - 1u)] = c;

    if (set_pending && c != 0ul) {
        atomic_or((volatile __global unsigned int*)pending, 1u);
    }
}

__kernel void gf61_carry_cleanup_serial(__global u64* digits,
                                        __global const u8* widths,
                                        __global u64* carry_in,
                                        __global u32* pending,
                                        __global u32* stats,
                                        const u32 n) {
    if (get_global_id(0) != 0) return;
    if (pending[0] == 0u) return;

    u32 again = 1u;
    u32 rounds = 0u;

    while (again) {
        again = 0u;
        u64 incoming = 0ul;

        for (u32 i = 0u; i < n; ++i) {
            const u32 w = (u32)widths[i];
            const u64 total = digits[i] + carry_in[i] + incoming;
            carry_in[i] = 0ul;
            digits[i] = total & digit_mask(w);
            incoming = total >> w;
        }

        if (incoming != 0ul) {
            carry_in[0] = incoming;
            again = 1u;
        }

        ++rounds;
    }

    stats[3] = rounds;
    atomic_inc((volatile __global unsigned int*)&stats[0]);
    atomic_add((volatile __global unsigned int*)&stats[1], rounds);
    atomic_max((volatile __global unsigned int*)&stats[2], rounds);
    pending[0] = 0u;
}

__kernel void gf61_carry_segment_first(__global u64* digits,
                                       __global const u8* widths,
                                       __global u64* carry_out,
                                       __global u32* pending,
                                       const u32 digit_n,
                                       const u32 segments,
                                       const u32 items_per_segment) {
    const u32 seg = (u32)get_global_id(0);
    if (seg >= segments) return;

    if (seg == 0u) pending[0] = 0u;

    const u32 begin = seg * items_per_segment;
    u32 end = begin + items_per_segment;
    if (end > digit_n) end = digit_n;

    u64 carry = 0ul;
    for (u32 i = begin; i < end; ++i) {
        const u32 w = (u32)widths[i];
        const u64 total = digits[i] + carry;
        digits[i] = total & digit_mask(w);
        carry = total >> w;
    }
    carry_out[(seg + 1u == segments) ? 0u : (seg + 1u)] = carry;
}

__kernel void gf61_carry_segment_pass(__global u64* digits,
                                      __global const u8* widths,
                                      __global const u64* carry_in,
                                      __global u64* carry_out,
                                      __global u32* pending,
                                      const u32 digit_n,
                                      const u32 segments,
                                      const u32 items_per_segment,
                                      const u32 set_pending) {
    const u32 seg = (u32)get_global_id(0);
    if (seg >= segments) return;

    u64 carry = carry_in[seg];
    const u32 next = (seg + 1u == segments) ? 0u : (seg + 1u);

    if (carry == 0ul) {
        carry_out[next] = 0ul;
        return;
    }

    const u32 begin = seg * items_per_segment;
    u32 end = begin + items_per_segment;
    if (end > digit_n) end = digit_n;

    for (u32 i = begin; i < end && carry != 0ul; ++i) {
        const u32 w = (u32)widths[i];
        const u64 total = digits[i] + carry;
        digits[i] = total & digit_mask(w);
        carry = total >> w;
    }

    carry_out[next] = carry;
    if (set_pending && carry != 0ul) {
        atomic_or((volatile __global unsigned int*)pending, 1u);
    }
}

__kernel void gf61_carry_cleanup_serial_segments(__global u64* digits,
                                                 __global const u8* widths,
                                                 __global u64* carry_in,
                                                 __global u32* pending,
                                                 __global u32* stats,
                                                 const u32 digit_n,
                                                 const u32 segments,
                                                 const u32 items_per_segment) {
    if (get_global_id(0) != 0) return;
    if (pending[0] == 0u) return;

    u32 rounds = 0u;
    u32 again = 1u;

    while (again) {
        again = 0u;
        u64 incoming = 0ul;

        for (u32 seg = 0u; seg < segments; ++seg) {
            u64 carry = carry_in[seg] + incoming;
            carry_in[seg] = 0ul;

            const u32 begin = seg * items_per_segment;
            u32 end = begin + items_per_segment;
            if (end > digit_n) end = digit_n;

            for (u32 i = begin; i < end && carry != 0ul; ++i) {
                const u32 w = (u32)widths[i];
                const u64 total = digits[i] + carry;
                digits[i] = total & digit_mask(w);
                carry = total >> w;
            }
            incoming = carry;
        }

        if (incoming != 0ul) {
            carry_in[0] = incoming;
            again = 1u;
        }
        ++rounds;
    }

    stats[3] = rounds;
    atomic_inc((volatile __global unsigned int*)&stats[0]);
    atomic_add((volatile __global unsigned int*)&stats[1], rounds);
    atomic_max((volatile __global unsigned int*)&stats[2], rounds);
    pending[0] = 0u;
}


#define CRT_M31 ((u64)2147483647ul)
#define CRT_M61 ((u64)2305843009213693951ul)

inline u64 crt_add_carry_lo(u64 a, u64 b, __private u64* carry)
{
    u64 r = a + b;
    *carry = (r < a) ? 1ul : 0ul;
    return r;
}

inline void crt_coeff_from_residues(u64 a61, u64 a31, __private u64* lo, __private u64* hi)
{
    
    
    u64 a61m31 = (a61 & CRT_M31) + (a61 >> 31);
    a61m31 = (a61m31 & CRT_M31) + (a61m31 >> 31);
    if (a61m31 >= CRT_M31) a61m31 -= CRT_M31;
    u64 diff = (a31 >= a61m31) ? (a31 - a61m31) : (a31 + CRT_M31 - a61m31);
    u64 twice = diff << 1;
    if (twice >= CRT_M31) twice -= CRT_M31;
    u64 t = (twice == 0ul) ? 0ul : (CRT_M31 - twice);

    
    u64 prod_lo = t << 61;
    u64 prod_hi = t >> 3;
    u64 old_lo = prod_lo;
    prod_lo -= t;
    prod_hi -= (old_lo < t) ? 1ul : 0ul;
    u64 c = 0ul;
    u64 sum_lo = crt_add_carry_lo(prod_lo, a61, &c);
    *lo = sum_lo;
    *hi = prod_hi + c;
}

inline void crt_add_u128(u64 alo, u64 ahi, u64 blo, u64 bhi, __private u64* rlo, __private u64* rhi)
{
    u64 c = 0ul;
    u64 lo = crt_add_carry_lo(alo, blo, &c);
    *rlo = lo;
    *rhi = ahi + bhi + c;
}

inline void crt_shr_u128(u64 lo, u64 hi, u32 s, __private u64* rlo, __private u64* rhi)
{
    if (s == 0u) {
        *rlo = lo;
        *rhi = hi;
    } else if (s < 64u) {
        *rlo = (lo >> s) | (hi << (64u - s));
        *rhi = (hi >> s);
    } else {
        *rlo = (hi >> (s - 64u));
        *rhi = 0ul;
    }
}

inline u32 crt_reduce_m31_ulong(u64 x)
{
    u64 y = (x & 0x7fffffffu) + (x >> 31);
    y = (y & 0x7fffffffu) + (y >> 31);
    return (u32)((y >= 2147483647UL) ? (y - 2147483647UL) : y);
}

inline u64 crt_low_mask(u32 w)
{
    return (w >= 64u) ? 0xfffffffffffffffful : ((1ul << w) - 1ul);
}

inline u64 crt_take_digit_and_shift_u128(u64 total_lo, u64 total_hi, u32 w,
                                         __private u64* carry_lo,
                                         __private u64* carry_hi)
{
    
    
    if (w == 31u) {
        *carry_lo = (total_lo >> 31u) | (total_hi << 33u);
        *carry_hi = (total_hi >> 31u);
        return total_lo & 0x7ffffffful;
    }
    if (w == 32u) {
        *carry_lo = (total_lo >> 32u) | (total_hi << 32u);
        *carry_hi = (total_hi >> 32u);
        return total_lo & 0xfffffffful;
    }
    if (w == 33u) {
        *carry_lo = (total_lo >> 33u) | (total_hi << 31u);
        *carry_hi = (total_hi >> 33u);
        return total_lo & 0x1fffffffful;
    }
    if (w == 34u) {
        *carry_lo = (total_lo >> 34u) | (total_hi << 30u);
        *carry_hi = (total_hi >> 34u);
        return total_lo & 0x3fffffffful;
    }
    const u64 digit = total_lo & crt_low_mask(w);
    crt_shr_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    return digit;
}

inline void crt_scan_digit_with_coeff(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    u32 i,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 xlo, xhi;
    crt_coeff_from_residues(digits61[i], digits31[i], &xlo, &xhi);
    u64 total_lo, total_hi;
    crt_add_u128(xlo, xhi, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u32 w = (u32)widths[i];
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
    digits31[i] = crt_reduce_m31_ulong(digit);
}

inline void crt_scan_digit_add_carry(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    u32 i,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 total_lo, total_hi;
    crt_add_u128(digits61[i], 0ul, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u32 w = (u32)widths[i];
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
    digits31[i] = crt_reduce_m31_ulong(digit);
}

__kernel void gf61_crt_garner_segment_first(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    __global u64* carry_lo_out,
    __global u64* carry_hi_out,
    __global u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;
    const u32 start = seg * items_per_segment;
    u32 end = start + items_per_segment;
    if (end > digit_n) end = digit_n;

    u64 clo = 0ul, chi = 0ul;
    for (u32 i = start; i < end; ++i) {
        crt_scan_digit_with_coeff(digits61, digits31, widths, i, &clo, &chi);
    }
    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_carry_segment_pass(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    __global const u64* carry_lo_in,
    __global const u64* carry_hi_in,
    __global u64* carry_lo_out,
    __global u64* carry_hi_out,
    __global u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;
    u64 clo = carry_lo_in[seg];
    u64 chi = carry_hi_in[seg];
    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    if ((clo | chi) == 0ul) {
        carry_lo_out[next] = 0ul;
        carry_hi_out[next] = 0ul;
        
        return;
    }
    const u32 start = seg * items_per_segment;
    u32 end = start + items_per_segment;
    if (end > digit_n) end = digit_n;
    for (u32 i = start; i < end; ++i) {
        crt_scan_digit_add_carry(digits61, digits31, widths, i, &clo, &chi);
        if ((clo | chi) == 0ul) break;
    }
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_carry_cleanup_serial(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    __global u64* carry_lo,
    __global u64* carry_hi,
    __global const u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    if (pending[0] == 0u) return;
    u64 clo = 0ul, chi = 0ul;
    for (u32 seg = 0u; seg < segments; ++seg) {
        u64 tlo, thi;
        crt_add_u128(clo, chi, carry_lo[seg], carry_hi[seg], &tlo, &thi);
        clo = tlo;
        chi = thi;
        const u32 start = seg * items_per_segment;
        u32 end = start + items_per_segment;
        if (end > digit_n) end = digit_n;
        for (u32 i = start; i < end; ++i) {
            if ((clo | chi) == 0ul) break;
            crt_scan_digit_add_carry(digits61, digits31, widths, i, &clo, &chi);
        }
    }
    
    for (u32 iter = 0u; iter < 4u && ((clo | chi) != 0ul); ++iter) {
        for (u32 i = 0u; i < digit_n && ((clo | chi) != 0ul); ++i) {
            crt_scan_digit_add_carry(digits61, digits31, widths, i, &clo, &chi);
        }
    }
}


inline void crt_scan_digit_with_coeff_oneout(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    u32 i,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 xlo, xhi;
    crt_coeff_from_residues(digits61[i], digits31[i], &xlo, &xhi);
    u64 total_lo, total_hi;
    crt_add_u128(xlo, xhi, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u32 w = (u32)widths[i];
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
}

inline void crt_scan_residue_values_oneout(
    __global u64* digits61,
    __global const u8* widths,
    u32 i,
    u64 d61,
    u32 d31,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 xlo, xhi;
    crt_coeff_from_residues(d61, d31, &xlo, &xhi);
    u64 total_lo, total_hi;
    crt_add_u128(xlo, xhi, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u32 w = (u32)widths[i];
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
}

inline void crt_scan_residue_values_oneout_w(
    __global u64* digits61,
    u32 i,
    u64 d61,
    u32 d31,
    u32 w,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 xlo, xhi;
    crt_coeff_from_residues(d61, d31, &xlo, &xhi);
    u64 total_lo, total_hi;
    crt_add_u128(xlo, xhi, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
}

inline void crt_scan_digit_add_carry_oneout(
    __global u64* digits61,
    __global const u8* widths,
    u32 i,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 total_lo, total_hi;
    crt_add_u128(digits61[i], 0ul, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u32 w = (u32)widths[i];
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
}


__kernel void gf61_crt_garner_segment_first_oneout(
    __global u64* digits61,
    __global u32* digits31,
    __global const u8* widths,
    __global u64* carry_lo_out,
    __global u64* carry_hi_out,
    __global u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;
    const u32 start = seg * items_per_segment;
    u32 end = start + items_per_segment;
    if (end > digit_n) end = digit_n;

    u64 clo = 0ul, chi = 0ul;
    for (u32 i = start; i < end; ++i) {
        crt_scan_digit_with_coeff_oneout(digits61, digits31, widths, i, &clo, &chi);
    }
    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

inline void crt_scan_digit_with_coeff_oneout_w(
    __global u64* digits61,
    __global u32* digits31,
    u32 i,
    u32 w,
    __private u64* carry_lo,
    __private u64* carry_hi)
{
    u64 xlo, xhi;
    crt_coeff_from_residues(digits61[i], digits31[i], &xlo, &xhi);
    u64 total_lo, total_hi;
    crt_add_u128(xlo, xhi, *carry_lo, *carry_hi, &total_lo, &total_hi);
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, carry_lo, carry_hi);
    digits61[i] = digit;
}

__kernel void gf61_crt_garner_segment_first_oneout_mask32(
    __global u64* restrict digits61,
    __global u32* restrict digits31,
    __global const u32* restrict width_mask32,
    const u32 width_base,
    __global u64* restrict carry_lo_out,
    __global u64* restrict carry_hi_out,
    __global u32* restrict pending,
    const u32 digit_n,
    const u32 segments)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 start = seg << 5;
    const u32 mask = width_mask32[seg];

    u64 clo = 0ul, chi = 0ul;

    
    if (start + 31u < digit_n) {
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + j;
            const u32 w = width_base + ((mask >> j) & 1u);
            crt_scan_digit_with_coeff_oneout_w(digits61, digits31, i, w, &clo, &chi);
        }
    } else {
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + j;
            if (i >= digit_n) break;
            const u32 w = width_base + ((mask >> j) & 1u);
            crt_scan_digit_with_coeff_oneout_w(digits61, digits31, i, w, &clo, &chi);
        }
    }
    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}


static inline __attribute__((always_inline)) void crt_scan_digit_oneout_base32_fast(
    __global u64* restrict digits61,
    __global u32* restrict digits31,
    const u32 i,
    const u32 width_bit,
    __private u64* clo,
    __private u64* chi)
{
    const u64 a61 = digits61[i];
    const u64 a31 = (u64)digits31[i];

    
    u64 a61m31 = (a61 & CRT_M31) + (a61 >> 31);
    a61m31 = (a61m31 & CRT_M31) + (a61m31 >> 31);
    if (a61m31 >= CRT_M31) a61m31 -= CRT_M31;

    const u64 diff = (a31 >= a61m31) ? (a31 - a61m31) : (a31 + CRT_M31 - a61m31);
    u64 twice = diff << 1;
    if (twice >= CRT_M31) twice -= CRT_M31;
    const u64 t = (twice == 0ul) ? 0ul : (CRT_M31 - twice);

    
    u64 prod_lo = t << 61;
    u64 prod_hi = t >> 3;
    const u64 old_lo = prod_lo;
    prod_lo -= t;
    prod_hi -= (old_lo < t) ? 1ul : 0ul;

    u64 c0 = 0ul;
    const u64 xlo = crt_add_carry_lo(prod_lo, a61, &c0);
    const u64 xhi = prod_hi + c0;

    u64 c1 = 0ul;
    const u64 total_lo = crt_add_carry_lo(xlo, *clo, &c1);
    const u64 total_hi = xhi + *chi + c1;

    
    const u32 sh = 32u + width_bit;
    const u64 digit_mask = 0xfffffffful | (((u64)width_bit) << 32);
    digits61[i] = total_lo & digit_mask;
    *clo = (total_lo >> sh) | (total_hi << (32u - width_bit));
    *chi = (total_hi >> sh);
}


static inline __attribute__((always_inline)) void crt_scan_digit_oneout_base32_u32lean_select(
    __global u64* restrict digits61,
    __global u32* restrict digits31,
    const u32 i,
    const u32 width_bit,
    __private u64* clo,
    __private u64* chi)
{
    const u64 a61 = digits61[i];
    const u32 a31 = digits31[i];

    
    u32 a61m31 = (u32)(a61 & 0x7ffffffful) + (u32)(a61 >> 31u);
    a61m31 = (a61m31 & 0x7fffffffu) + (a61m31 >> 31u);
    if (a61m31 >= 0x7fffffffu) a61m31 -= 0x7fffffffu;

    const u32 diff = (a31 >= a61m31) ? (a31 - a61m31) : (a31 + 0x7fffffffu - a61m31);
    u32 twice = diff << 1u;
    if (twice >= 0x7fffffffu) twice -= 0x7fffffffu;
    const u32 t32 = (twice == 0u) ? 0u : (0x7fffffffu - twice);
    const u64 t = (u64)t32;

    
    u64 prod_lo = t << 61u;
    u64 prod_hi = (u64)(t32 >> 3u);
    const u64 old_lo = prod_lo;
    prod_lo -= t;
    prod_hi -= (old_lo < t) ? 1ul : 0ul;

    u64 c0 = 0ul;
    const u64 xlo = crt_add_carry_lo(prod_lo, a61, &c0);
    const u64 xhi = prod_hi + c0;

    u64 c1 = 0ul;
    const u64 total_lo = crt_add_carry_lo(xlo, *clo, &c1);
    const u64 total_hi = xhi + *chi + c1;

    
    if (width_bit == 0u) {
        digits61[i] = total_lo & 0xfffffffful;
        *clo = (total_lo >> 32u) | (total_hi << 32u);
        *chi = (total_hi >> 32u);
    } else {
        digits61[i] = total_lo & 0x1fffffffful;
        *clo = (total_lo >> 33u) | (total_hi << 31u);
        *chi = (total_hi >> 33u);
    }
}

__kernel void gf61_crt_garner_segment_first_oneout_mask32_base32_u32lean(
    __global u64* restrict digits61,
    __global u32* restrict digits31,
    __global const u32* restrict width_mask32,
    const u32 width_base,
    __global u64* restrict carry_lo_out,
    __global u64* restrict carry_hi_out,
    __global u32* restrict pending,
    const u32 digit_n,
    const u32 segments)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    if (width_base != 32u) {
        const u32 start_fallback = seg << 5;
        const u32 mask_fallback = width_mask32[seg];
        u64 clo_fallback = 0ul, chi_fallback = 0ul;
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start_fallback + j;
            if (i >= digit_n) break;
            const u32 w = width_base + ((mask_fallback >> j) & 1u);
            crt_scan_digit_with_coeff_oneout_w(digits61, digits31, i, w, &clo_fallback, &chi_fallback);
        }
        const u32 next_fallback = (seg + 1u < segments) ? (seg + 1u) : 0u;
        carry_lo_out[next_fallback] = clo_fallback;
        carry_hi_out[next_fallback] = chi_fallback;
        if ((clo_fallback | chi_fallback) != 0ul) pending[0] = 1u;
        return;
    }

    const u32 start = seg << 5;
    const u32 mask = width_mask32[seg];
    u64 clo = 0ul, chi = 0ul;

    if (start + 31u < digit_n) {
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_digit_oneout_base32_u32lean_select(digits61, digits31, start + j, (mask >> j) & 1u, &clo, &chi);
        }
    } else {
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + j;
            if (i >= digit_n) break;
            crt_scan_digit_oneout_base32_u32lean_select(digits61, digits31, i, (mask >> j) & 1u, &clo, &chi);
        }
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_garner_segment_first_oneout_mask32_base32_fast(
    __global u64* restrict digits61,
    __global u32* restrict digits31,
    __global const u32* restrict width_mask32,
    const u32 width_base,
    __global u64* restrict carry_lo_out,
    __global u64* restrict carry_hi_out,
    __global u32* restrict pending,
    const u32 digit_n,
    const u32 segments)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    
    if (width_base != 32u) {
        const u32 start_fallback = seg << 5;
        const u32 mask_fallback = width_mask32[seg];
        u64 clo_fallback = 0ul, chi_fallback = 0ul;
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start_fallback + j;
            if (i >= digit_n) break;
            const u32 w = width_base + ((mask_fallback >> j) & 1u);
            crt_scan_digit_with_coeff_oneout_w(digits61, digits31, i, w, &clo_fallback, &chi_fallback);
        }
        const u32 next_fallback = (seg + 1u < segments) ? (seg + 1u) : 0u;
        carry_lo_out[next_fallback] = clo_fallback;
        carry_hi_out[next_fallback] = chi_fallback;
        if ((clo_fallback | chi_fallback) != 0ul) pending[0] = 1u;
        return;
    }

    const u32 start = seg << 5;
    const u32 mask = width_mask32[seg];
    u64 clo = 0ul, chi = 0ul;

    if (start + 31u < digit_n) {
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_digit_oneout_base32_fast(digits61, digits31, start + j, (mask >> j) & 1u, &clo, &chi);
        }
    } else {
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + j;
            if (i >= digit_n) break;
            crt_scan_digit_oneout_base32_fast(digits61, digits31, i, (mask >> j) & 1u, &clo, &chi);
        }
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_garner_segment_first_oneout_mask32_base32_x2(
    __global u64* restrict digits61,
    __global u32* restrict digits31,
    __global const u32* restrict width_mask32,
    const u32 width_base,
    __global u64* restrict carry_lo_out,
    __global u64* restrict carry_hi_out,
    __global u32* restrict pending,
    const u32 digit_n,
    const u32 segments)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 start = seg << 6;
    const u32 mask_index = seg << 1;
    const u32 mask0 = width_mask32[mask_index];
    const u32 mask1 = ((start + 32u) < digit_n) ? width_mask32[mask_index + 1u] : 0u;
    u64 clo = 0ul, chi = 0ul;

    if (width_base == 32u && start + 63u < digit_n) {
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_digit_oneout_base32_fast(digits61, digits31, start + j, (mask0 >> j) & 1u, &clo, &chi);
        }
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_digit_oneout_base32_fast(digits61, digits31, start + 32u + j, (mask1 >> j) & 1u, &clo, &chi);
        }
    } else {
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + j;
            if (i >= digit_n) break;
            const u32 w = width_base + ((mask0 >> j) & 1u);
            crt_scan_digit_with_coeff_oneout_w(digits61, digits31, i, w, &clo, &chi);
        }
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + 32u + j;
            if (i >= digit_n) break;
            const u32 w = width_base + ((mask1 >> j) & 1u);
            crt_scan_digit_with_coeff_oneout_w(digits61, digits31, i, w, &clo, &chi);
        }
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}


static inline __attribute__((always_inline)) void crt_scan_coeffhi_oneout_w(
    __global u64* restrict digits61,
    __global u32* restrict coeff_hi32,
    const u32 i,
    const u32 w,
    __private u64* clo,
    __private u64* chi)
{
    const u64 xlo = digits61[i];
    const u64 xhi = (u64)coeff_hi32[i];
    u64 total_lo, total_hi;
    crt_add_u128(xlo, xhi, *clo, *chi, &total_lo, &total_hi);
    const u64 digit = crt_take_digit_and_shift_u128(total_lo, total_hi, w, clo, chi);
    digits61[i] = digit;
}

static inline __attribute__((always_inline)) void crt_scan_coeffhi_oneout_base32_select(
    __global u64* restrict digits61,
    __global u32* restrict coeff_hi32,
    const u32 i,
    const u32 width_bit,
    __private u64* clo,
    __private u64* chi)
{
    const u64 xlo = digits61[i];
    const u64 xhi = (u64)coeff_hi32[i];

    u64 c = 0ul;
    const u64 total_lo = crt_add_carry_lo(xlo, *clo, &c);
    const u64 total_hi = xhi + *chi + c;

    if (width_bit == 0u) {
        digits61[i] = total_lo & 0xfffffffful;
        *clo = (total_lo >> 32u) | (total_hi << 32u);
        *chi = (total_hi >> 32u);
    } else {
        digits61[i] = total_lo & 0x1fffffffful;
        *clo = (total_lo >> 33u) | (total_hi << 31u);
        *chi = (total_hi >> 33u);
    }
}

__kernel void gf61_crt_garner_segment_first_oneout_coeffhi_mask32_base32(
    __global u64* restrict digits61,
    __global u32* restrict coeff_hi32,
    __global const u32* restrict width_mask32,
    const u32 width_base,
    __global u64* restrict carry_lo_out,
    __global u64* restrict carry_hi_out,
    __global u32* restrict pending,
    const u32 digit_n,
    const u32 segments)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 start = seg << 5;
    const u32 mask = width_mask32[seg];
    u64 clo = 0ul, chi = 0ul;

    if (width_base == 32u && start + 31u < digit_n) {
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_coeffhi_oneout_base32_select(digits61, coeff_hi32, start + j, (mask >> j) & 1u, &clo, &chi);
        }
    } else {
        for (u32 j = 0u; j < 32u; ++j) {
            const u32 i = start + j;
            if (i >= digit_n) break;
            const u32 w = width_base + ((mask >> j) & 1u);
            crt_scan_coeffhi_oneout_w(digits61, coeff_hi32, i, w, &clo, &chi);
        }
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}


__kernel void gf61_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2(
    __global u64* restrict digits61,
    __global u32* restrict coeff_hi32,
    __global const u32* restrict width_mask32,
    const u32 width_base,
    __global u64* restrict carry_lo_out,
    __global u64* restrict carry_hi_out,
    __global u32* restrict pending,
    const u32 digit_n,
    const u32 segments)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 start = seg << 6;
    const u32 mask0 = width_mask32[seg << 1];
    const u32 mask1 = width_mask32[(seg << 1) + 1u];
    u64 clo = 0ul, chi = 0ul;

    if (width_base == 32u && start + 63u < digit_n) {
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_coeffhi_oneout_base32_select(digits61, coeff_hi32, start + j, (mask0 >> j) & 1u, &clo, &chi);
        }
        #pragma unroll 32
        for (u32 j = 0u; j < 32u; ++j) {
            crt_scan_coeffhi_oneout_base32_select(digits61, coeff_hi32, start + 32u + j, (mask1 >> j) & 1u, &clo, &chi);
        }
    } else {
        for (u32 j = 0u; j < 64u; ++j) {
            const u32 i = start + j;
            if (i >= digit_n) break;
            const u32 mask = (j < 32u) ? mask0 : mask1;
            const u32 bit = (mask >> (j & 31u)) & 1u;
            crt_scan_coeffhi_oneout_w(digits61, coeff_hi32, i, width_base + bit, &clo, &chi);
        }
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_carry_segment_pass_oneout(
    __global u64* digits61,
    __global const u8* widths,
    __global const u64* carry_lo_in,
    __global const u64* carry_hi_in,
    __global u64* carry_lo_out,
    __global u64* carry_hi_out,
    __global u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;
    u64 clo = carry_lo_in[seg];
    u64 chi = carry_hi_in[seg];
    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    if ((clo | chi) == 0ul) {
        carry_lo_out[next] = 0ul;
        carry_hi_out[next] = 0ul;
        return;
    }
    const u32 start = seg * items_per_segment;
    u32 end = start + items_per_segment;
    if (end > digit_n) end = digit_n;
    for (u32 i = start; i < end; ++i) {
        crt_scan_digit_add_carry_oneout(digits61, widths, i, &clo, &chi);
        if ((clo | chi) == 0ul) break;
    }
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_carry_cleanup_serial_oneout(
    __global u64* digits61,
    __global const u8* widths,
    __global u64* carry_lo,
    __global u64* carry_hi,
    __global const u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    if (pending[0] == 0u) return;
    u64 clo = 0ul, chi = 0ul;
    for (u32 seg = 0u; seg < segments; ++seg) {
        u64 tlo, thi;
        crt_add_u128(clo, chi, carry_lo[seg], carry_hi[seg], &tlo, &thi);
        clo = tlo;
        chi = thi;
        const u32 start = seg * items_per_segment;
        u32 end = start + items_per_segment;
        if (end > digit_n) end = digit_n;
        for (u32 i = start; i < end; ++i) {
            if ((clo | chi) == 0ul) break;
            crt_scan_digit_add_carry_oneout(digits61, widths, i, &clo, &chi);
        }
    }
    for (u32 iter = 0u; iter < 4u && ((clo | chi) != 0ul); ++iter) {
        for (u32 i = 0u; i < digit_n && ((clo | chi) != 0ul); ++i) {
            crt_scan_digit_add_carry_oneout(digits61, widths, i, &clo, &chi);
        }
    }
}

__kernel void gf61_crt_carry_cleanup_parallel_oneout(
    __global u64* digits61,
    __global const u8* widths,
    __global const u64* carry_lo_in,
    __global const u64* carry_hi_in,
    __global u64* carry_lo_out,
    __global u64* carry_hi_out,
    __global u32* pending,
    const u32 digit_n,
    const u32 segments,
    const u32 items_per_segment)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = 0ul;
    carry_hi_out[next] = 0ul;

    u64 clo = carry_lo_in[seg];
    u64 chi = carry_hi_in[seg];
    if ((clo | chi) == 0ul) return;

    const u32 start = seg * items_per_segment;
    u32 end = start + items_per_segment;
    if (end > digit_n) end = digit_n;

    
    for (u32 i = start; i < end; ++i) {
        crt_scan_digit_add_carry_oneout(digits61, widths, i, &clo, &chi);
        if ((clo | chi) == 0ul) break;
    }

    if ((clo | chi) != 0ul) {
        carry_lo_out[next] = clo;
        carry_hi_out[next] = chi;
        pending[0] = 1u;
    }
}

typedef uint2 GF31;
#define CRT_P31 0x7fffffffu

inline uint f31_reduce_uint(uint x) {
    uint r = (x & CRT_P31) + (x >> 31);
    r = (r & CRT_P31) + (r >> 31);
    return (r == CRT_P31) ? 0u : r;
}

inline uint f31_reduce_ulong(ulong x) {
    uint r = (uint)((x & 0x7ffffffful) + (x >> 31));
    r = (r & CRT_P31) + (r >> 31);
    return (r == CRT_P31) ? 0u : r;
}

inline uint f31_add_scalar(uint a, uint b) {
    uint s = a + b;
    return (s & CRT_P31) + (s >> 31);
}

inline uint f31_sub_scalar(uint a, uint b) {
    uint d = a - b;
    d += ((uint)-(a < b)) & CRT_P31;
    return d;
}

inline uint f31_neg_lazy_scalar(uint a) {
    return CRT_P31 - a;
}

inline uint f31_double_scalar(uint a) {
    uint s = a << 1;
    return (s & CRT_P31) + (s >> 31);
}

static inline __attribute__((always_inline)) uint f31_mul_scalar(uint a, uint b) {
    const uint lo = a * b;
    const uint hi = mul_hi(a, b);
    uint r = (lo & CRT_P31) + (lo >> 31) + (hi << 1);
    return (r & CRT_P31) + (r >> 31);
}

#ifndef CRT_GF31_KARATSUBA
#define CRT_GF31_KARATSUBA 1
#endif

inline GF31 f31_add(GF31 x, GF31 y) {
    return (GF31)(f31_add_scalar(x.s0, y.s0),
                  f31_add_scalar(x.s1, y.s1));
}

inline GF31 f31_sub(GF31 x, GF31 y) {
    return (GF31)(f31_sub_scalar(x.s0, y.s0),
                  f31_sub_scalar(x.s1, y.s1));
}

static inline __attribute__((always_inline)) GF31 f31_mul(GF31 x, GF31 y) {
#if CRT_GF31_KARATSUBA
    // GF31 complex multiply in GF(p^2), i^2=-1.
    // Karatsuba saves one 32x32 mul_hi path versus the old 4-mul formula.
    const uint ac = f31_mul_scalar(x.s0, y.s0);
    const uint bd = f31_mul_scalar(x.s1, y.s1);
    const uint sx = f31_add_scalar(x.s0, x.s1);
    const uint sy = f31_add_scalar(y.s0, y.s1);
    const uint abcd = f31_mul_scalar(sx, sy);
    return (GF31)(f31_sub_scalar(ac, bd),
                  f31_sub_scalar(f31_sub_scalar(abcd, ac), bd));
#else
    const uint ac = f31_mul_scalar(x.s0, y.s0);
    const uint bd = f31_mul_scalar(x.s1, y.s1);
    const uint ad = f31_mul_scalar(x.s0, y.s1);
    const uint bc = f31_mul_scalar(x.s1, y.s0);
    return (GF31)(f31_sub_scalar(ac, bd),
                  f31_add_scalar(ad, bc));
#endif
}

inline GF31 f31_sqr(GF31 x) {
    const uint apb = f31_add_scalar(x.s0, x.s1);
    const uint amb = f31_sub_scalar(x.s0, x.s1);
    const uint real = f31_mul_scalar(apb, amb);
    const uint imag = f31_double_scalar(f31_mul_scalar(x.s0, x.s1));
    return (GF31)(real, imag);
}

inline GF31 f31_dbl_pair(GF31 x) {
    return (GF31)(f31_double_scalar(x.s0), f31_double_scalar(x.s1));
}

#define F31_ADD_SUB(A, B, S, D) do { \
    const GF31 _f31_a = (A); \
    const GF31 _f31_b = (B); \
    (S) = (GF31)(f31_add_scalar(_f31_a.s0, _f31_b.s0), f31_add_scalar(_f31_a.s1, _f31_b.s1)); \
    (D) = (GF31)(f31_sub_scalar(_f31_a.s0, _f31_b.s0), f31_sub_scalar(_f31_a.s1, _f31_b.s1)); \
} while (0)

#define F31_DIF_MUL(A, B, W, S, D) do { \
    GF31 _f31_d; \
    F31_ADD_SUB((A), (B), (S), _f31_d); \
    (D) = f31_mul(_f31_d, (W)); \
} while (0)

#define F31_DIT_MUL(A, B, W, S, D) do { \
    const GF31 _f31_v = f31_mul((B), (W)); \
    F31_ADD_SUB((A), _f31_v, (S), (D)); \
} while (0)

inline uint f31_mod31_small(uint s) {
    s = (s & 31u) + (s >> 5);
    return s - (31u & (0u - (uint)(s >= 31u)));
}

static inline __attribute__((always_inline)) uint f31_lshift_scalar_norm(uint x, uint s) {
    /* M31: multiplying by 2^s is a rotate in 31 bits.
       The mask on the right shift makes s=0 and s=31 valid and branchless. */
    return ((x << s) & CRT_P31) | (x >> ((31u - s) & 31u));
}

static inline __attribute__((always_inline)) uint f31_lshift_scalar(uint x, uint s) {
    return f31_lshift_scalar_norm(x, f31_mod31_small(s));
}

inline uint crt_mod31_u32_fast(uint x) {
    x = (x & 31u) + (x >> 5);
    x = (x & 31u) + (x >> 5);
    x = (x & 31u) + (x >> 5);
    x = (x & 31u) + (x >> 5);
    return x - (31u & (0u - (uint)(x >= 31u)));
}

inline uint crt_add31_fast(uint a, uint b) {
    uint s = a + b;
    return s - (31u & (0u - (uint)(s >= 31u)));
}

inline uint crt_sub31_fast(uint a, uint b) {
    const uint lt = (uint)(a < b);
    return a + (31u & (0u - lt)) - b;
}

inline uint crt_sub31_if(uint a, uint b, uint cond01) {
    const uint bsel = b & (0u - cond01);
    const uint lt = (uint)(a < bsel);
    return a + (31u & (0u - lt)) - bsel;
}

inline uint f31_lshift_scalar_nomod(uint x, uint s) {
    return f31_lshift_scalar_norm(x, s);
}

inline uint f31_rshift_scalar_nomod(uint x, uint s) {
    return f31_lshift_scalar_norm(x, 31u - s);
}

inline uint shift_from_r31_residue_mod(uint r, uint r31, uint lr2_mod31) {
    const uint x = crt_mod31_u32_fast(r31 * lr2_mod31);
    const uint s = f31_mod31_small(32u - x);
    return s & (0u - (uint)(r != 0u));
}
inline uint shift_from_r31_residue(uint r, uint r31, uint lr2) {
    return shift_from_r31_residue_mod(r, r31, f31_mod31_small(lr2));
}

inline void crt_advance_r31(uint* r, uint* r31, uint step, uint step31, uint n, uint n31) {
    uint nr = *r + step;
    const uint wrap = (uint)(nr >= n);
    nr -= n & (0u - wrap);
    uint nb = crt_add31_fast(*r31, step31);
    nb = crt_sub31_if(nb, n31, wrap);
    *r = nr;
    *r31 = nb;
}

inline uint crt_mod61_small(uint x) {
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    return x - (61u & (0u - (uint)(x >= 61u)));
}

inline uint crt_mod61_u32_fast(uint x) {
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    x = (x & 63u) + 3u * (x >> 6);
    return x - (61u & (0u - (uint)(x >= 61u)));
}

inline uint crt_add61_fast(uint a, uint b) {
    uint s = a + b;
    return s - (61u & (0u - (uint)(s >= 61u)));
}

inline uint crt_sub61_fast(uint a, uint b) {
    const uint lt = (uint)(a < b);
    return a + (61u & (0u - lt)) - b;
}

inline uint crt_sub61_if(uint a, uint b, uint cond01) {
    const uint bsel = b & (0u - cond01);
    const uint lt = (uint)(a < bsel);
    return a + (61u & (0u - lt)) - bsel;
}

inline void crt_advance_r61(uint* r, uint* r61, uint step, uint step61, uint n, uint n61) {
    uint nr = *r + step;
    const uint wrap = (uint)(nr >= n);
    nr -= n & (0u - wrap);
    uint nr61 = crt_add61_fast(*r61, step61);
    nr61 = crt_sub61_if(nr61, n61, wrap);
    *r = nr;
    *r61 = nr61;
}

inline uint shift_from_r61_residue_mod(uint r61, uint lr2_61_mod) {
    const uint prod = crt_mod61_small(r61 * lr2_61_mod);
    uint s = 62u - prod;
    s -= 61u & (0u - (uint)(s >= 61u));
    return s;
}
inline uint shift_from_r61_residue(uint r61, uint lr2_61) {
    return shift_from_r61_residue_mod(r61, crt_mod61_u32_fast(lr2_61));
}

static inline __attribute__((always_inline)) uint f31_lshift15_scalar(uint x) {
    return ((x << 15u) & CRT_P31) | (x >> 16u);
}

static inline __attribute__((always_inline)) uint f31_lshift29_scalar(uint x) {
    return ((x << 29u) & CRT_P31) | (x >> 2u);
}

static inline __attribute__((always_inline)) uint f31_lshift30_scalar(uint x) {
    return ((x << 30u) & CRT_P31) | (x >> 1u);
}

static inline __attribute__((always_inline)) GF31 f31_lshift(GF31 v, uint s) {
    s = f31_mod31_small(s);
    return (GF31)(f31_lshift_scalar_norm(v.s0, s),
                  f31_lshift_scalar_norm(v.s1, s));
}

static inline __attribute__((always_inline)) GF31 f31_rshift(GF31 v, uint s) {
    s = f31_mod31_small(s);
    const uint rs = (31u - s) & 31u;
    return (GF31)(f31_lshift_scalar_norm(v.s0, rs),
                  f31_lshift_scalar_norm(v.s1, rs));
}

inline GF31 f31_mul_i_fast(GF31 z) {
    return (GF31)(f31_neg_lazy_scalar(z.s1), z.s0);
}

inline GF31 f31_mul_minus_i_fast(GF31 z) {
    return (GF31)(z.s1, f31_neg_lazy_scalar(z.s0));
}

inline GF31 f31_mul_w8_fast(GF31 z) {
    return (GF31)(f31_lshift15_scalar(f31_add_scalar(z.s0, z.s1)),
                  f31_lshift15_scalar(f31_sub_scalar(z.s1, z.s0)));
}

inline GF31 f31_mul_w8_3_fast(GF31 z) {
    const uint sum = f31_add_scalar(z.s0, z.s1);
    return (GF31)(f31_lshift15_scalar(f31_sub_scalar(z.s1, z.s0)),
                  f31_lshift15_scalar(f31_neg_lazy_scalar(sum)));
}

inline GF31 f31_mul_w8_inv_fast(GF31 z) {
    return (GF31)(f31_lshift15_scalar(f31_sub_scalar(z.s0, z.s1)),
                  f31_lshift15_scalar(f31_add_scalar(z.s0, z.s1)));
}

inline GF31 f31_mul_w8_inv3_fast(GF31 z) {
    const uint sum = f31_add_scalar(z.s0, z.s1);
    return (GF31)(f31_lshift15_scalar(f31_neg_lazy_scalar(sum)),
                  f31_lshift15_scalar(f31_sub_scalar(z.s0, z.s1)));
}

inline uint shift_from_r31(uint r, uint lr2) {
    if (r == 0u) return 0u;
    uint x = (r * lr2) % 31u;
    return (32u - x) % 31u;
}

inline GF31 f31_weight_digit(ulong digit, uint index, uint p, uint n, uint lr2) {
    uint r = (uint)(((ulong)index * (ulong)p) & (ulong)(n - 1u));
    return f31_lshift((GF31)(f31_reduce_ulong(digit), 0u),
                      shift_from_r31(r, lr2));
}

inline uint f31_unweight_digit_logn(GF31 v, uint r, uint lr2, uint log_n) {
    GF31 y = f31_rshift(v, shift_from_r31(r, lr2) + log_n);
    return f31_reduce_uint(y.s0);
}

__kernel void gf61_crt_last_unweight_garner_segment_first_oneout(__global GF* a61,
                                                                 __global GF31* a31,
                                                                 __global const GF* tw61,
                                                                 __global const GF31* tw31,
                                                                 __global const u8* widths,
                                                                 __global u64* digits61,
                                                                 __global u64* carry_lo_out,
                                                                 __global u64* carry_hi_out,
                                                                 __global u32* pending,
                                                                 const u32 n,
                                                                 const u32 p,
                                                                 const u32 lr2_61,
                                                                 const u32 lr2_31,
                                                                 const u32 log_n,
                                                                 const u32 digit_n,
                                                                 const u32 segments,
                                                                 const u32 items_per_segment)
{
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 q = n >> 2;
    const u32 len = n >> 1;
    const u32 nmask = n - 1u;
    const u32 start = seg * items_per_segment;
    u32 end = start + items_per_segment;
    if (end > digit_n) end = digit_n;

    u64 clo = 0ul, chi = 0ul;
    for (u32 i = start; i < end; ++i) {
        const u32 quad = i / q;
        const u32 j = i - quad * q;
        const u32 r = (u32)(((u64)i * (u64)p) & (u64)nmask);

        const u32 i0 = j;
        const u32 i1 = j + q;
        const u32 i2 = j + 2u * q;
        const u32 i3 = j + 3u * q;

        const GF x0 = a61[i0];
        const GF x1 = a61[i1];
        const GF x2 = a61[i2];
        const GF x3 = a61[i3];
        const GF t1 = tw61[q - 1u + j];
        const GF t2 = tw61[len - 1u + j];
        const GF t3 = tw61[len + q - 1u + j];

        const GF u0 = gf_add(x0, x1);
        const GF u1 = gf_mul(gf_sub(x0, x1), t1);
        const GF u2 = gf_add(x2, x3);
        const GF u3 = gf_mul(gf_sub(x2, x3), t1);
        const GF y0 = gf_add(u0, u2);
        const GF y2 = gf_mul(gf_sub(u0, u2), t2);
        const GF y1 = gf_add(u1, u3);
        const GF y3 = gf_mul(gf_sub(u1, u3), t3);
        const GF y61 = (quad == 0u) ? y0 : ((quad == 1u) ? y1 : ((quad == 2u) ? y2 : y3));
        const u64 d61 = gf61_unweight_digit_from_r(y61, r, lr2_61, log_n);

        const GF31 z0 = a31[i0];
        const GF31 z1 = a31[i1];
        const GF31 z2 = a31[i2];
        const GF31 z3 = a31[i3];
        const GF31 s1 = tw31[q - 1u + j];
        const GF31 s2 = tw31[len - 1u + j];
        const GF31 s3 = tw31[len + q - 1u + j];

        const GF31 r0 = f31_add(z0, z1);
        const GF31 r1 = f31_mul(f31_sub(z0, z1), s1);
        const GF31 r2 = f31_add(z2, z3);
        const GF31 r3 = f31_mul(f31_sub(z2, z3), s1);
        const GF31 w0 = f31_add(r0, r2);
        const GF31 w2 = f31_mul(f31_sub(r0, r2), s2);
        const GF31 w1 = f31_add(r1, r3);
        const GF31 w3 = f31_mul(f31_sub(r1, r3), s3);
        const GF31 y31 = (quad == 0u) ? w0 : ((quad == 1u) ? w1 : ((quad == 2u) ? w2 : w3));
        const u32 d31 = f31_unweight_digit_logn(y31, r, lr2_31, log_n);

        crt_scan_residue_values_oneout(digits61, widths, i, d61, d31, &clo, &chi);
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}

__kernel void gf61_crt_last_unweight_garner_segment_first_oneout_mask32(__global GF* a61,
                                                                        __global GF31* a31,
                                                                        __global const GF* tw61,
                                                                        __global const GF31* tw31,
                                                                        __global const u32* width_mask32,
                                                                        const u32 width_base,
                                                                        __global u64* digits61,
                                                                        __global u64* carry_lo_out,
                                                                        __global u64* carry_hi_out,
                                                                        __global u32* pending,
                                                                        const u32 n,
                                                                        const u32 p,
                                                                        const u32 lr2_61,
                                                                        const u32 lr2_31,
                                                                        const u32 log_n,
                                                                        const u32 digit_n,
                                                                        const u32 segments,
                                                                        const u32 items_per_segment)
{
    (void)items_per_segment;
    const u32 seg = get_global_id(0);
    if (seg >= segments) return;

    const u32 q = n >> 2;
    const u32 len = n >> 1;
    const u32 nmask = n - 1u;
    const u32 start = seg << 5;
    const u32 end0 = start + 32u;
    const u32 end = (end0 > digit_n) ? digit_n : end0;
    const u32 mask = width_mask32[seg];

    u64 clo = 0ul, chi = 0ul;
    for (u32 i = start; i < end; ++i) {
        const u32 bit = i - start;
        const u32 digit_w = width_base + ((mask >> bit) & 1u);
        const u32 quad = i / q;
        const u32 j = i - quad * q;
        const u32 r = (u32)(((u64)i * (u64)p) & (u64)nmask);

        const u32 i0 = j;
        const u32 i1 = j + q;
        const u32 i2 = j + 2u * q;
        const u32 i3 = j + 3u * q;

        const GF x0 = a61[i0];
        const GF x1 = a61[i1];
        const GF x2 = a61[i2];
        const GF x3 = a61[i3];
        const GF t1 = tw61[q - 1u + j];
        const GF t2 = tw61[len - 1u + j];
        const GF t3 = tw61[len + q - 1u + j];

        const GF u0 = gf_add(x0, x1);
        const GF u1 = gf_mul(gf_sub(x0, x1), t1);
        const GF u2 = gf_add(x2, x3);
        const GF u3 = gf_mul(gf_sub(x2, x3), t1);
        const GF y0 = gf_add(u0, u2);
        const GF y2 = gf_mul(gf_sub(u0, u2), t2);
        const GF y1 = gf_add(u1, u3);
        const GF y3 = gf_mul(gf_sub(u1, u3), t3);
        const GF y61 = (quad == 0u) ? y0 : ((quad == 1u) ? y1 : ((quad == 2u) ? y2 : y3));
        const u64 d61 = gf61_unweight_digit_from_r(y61, r, lr2_61, log_n);

        const GF31 z0 = a31[i0];
        const GF31 z1 = a31[i1];
        const GF31 z2 = a31[i2];
        const GF31 z3 = a31[i3];
        const GF31 s1 = tw31[q - 1u + j];
        const GF31 s2 = tw31[len - 1u + j];
        const GF31 s3 = tw31[len + q - 1u + j];

        const GF31 r0 = f31_add(z0, z1);
        const GF31 r1 = f31_mul(f31_sub(z0, z1), s1);
        const GF31 r2 = f31_add(z2, z3);
        const GF31 r3 = f31_mul(f31_sub(z2, z3), s1);
        const GF31 w0 = f31_add(r0, r2);
        const GF31 w2 = f31_mul(f31_sub(r0, r2), s2);
        const GF31 w1 = f31_add(r1, r3);
        const GF31 w3 = f31_mul(f31_sub(r1, r3), s3);
        const GF31 y31 = (quad == 0u) ? w0 : ((quad == 1u) ? w1 : ((quad == 2u) ? w2 : w3));
        const u32 d31 = f31_unweight_digit_logn(y31, r, lr2_31, log_n);

        crt_scan_residue_values_oneout_w(digits61, i, d61, d31, digit_w, &clo, &chi);
    }

    const u32 next = (seg + 1u < segments) ? (seg + 1u) : 0u;
    carry_lo_out[next] = clo;
    carry_hi_out[next] = chi;
    if ((clo | chi) != 0ul) pending[0] = 1u;
}


inline void crt_radix4_dif_61(__global GF* a, __global const GF* tw, uint n, uint len, uint gid) {
    uint q = len >> 2;
    uint h = len >> 1;
    uint groups = n / len;
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + h;
    uint i3 = i2 + q;
    GF A = a[i0], B = a[i2];
    GF C = gf_add(A, B);
    GF D = gf_mul(gf_sub(A, B), tw[((len >> 1) - 1u) + j]);
    A = a[i1]; B = a[i3];
    GF E = gf_add(A, B);
    GF F = gf_mul(gf_sub(A, B), tw[((len >> 1) - 1u) + j + q]);
    GF W = tw[((h >> 1) - 1u) + j];
    a[i0] = gf_add(C, E);
    a[i1] = gf_mul(gf_sub(C, E), W);
    a[i2] = gf_add(D, F);
    a[i3] = gf_mul(gf_sub(D, F), W);
}
inline void crt_radix4_dif_31(__global GF31* a, __global const GF31* tw, uint n, uint len, uint gid) {
    uint q = len >> 2;
    uint h = len >> 1;
    uint groups = n / len;
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + h;
    uint i3 = i2 + q;
    GF31 A = a[i0], B = a[i2];
    GF31 C = f31_add(A, B);
    GF31 D = f31_mul(f31_sub(A, B), tw[((len >> 1) - 1u) + j]);
    A = a[i1]; B = a[i3];
    GF31 E = f31_add(A, B);
    GF31 F = f31_mul(f31_sub(A, B), tw[((len >> 1) - 1u) + j + q]);
    GF31 W = tw[((h >> 1) - 1u) + j];
    a[i0] = f31_add(C, E);
    a[i1] = f31_mul(f31_sub(C, E), W);
    a[i2] = f31_add(D, F);
    a[i3] = f31_mul(f31_sub(D, F), W);
}
inline void crt_radix4_dit_61(__global GF* a, __global const GF* tw, uint n, uint len, uint gid) {
    uint q = len >> 1;
    uint groups = n / (len << 1);
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + len;
    uint i3 = i2 + q;
    GF A = a[i0];
    GF B = gf_mul(a[i1], tw[((len >> 1) - 1u) + j]);
    GF C = gf_add(A, B);
    GF D = gf_sub(A, B);
    GF E = a[i2];
    GF F = gf_mul(a[i3], tw[((len >> 1) - 1u) + j]);
    GF G = gf_add(E, F);
    GF H = gf_sub(E, F);
    

    GF TG = gf_mul(G, tw[(len - 1u) + j]);
    GF TH = gf_mul(H, tw[(len - 1u) + j + q]);
    a[i0] = gf_add(C, TG);
    a[i2] = gf_sub(C, TG);
    a[i1] = gf_add(D, TH);
    a[i3] = gf_sub(D, TH);
}
inline void crt_radix4_dit_31(__global GF31* a, __global const GF31* tw, uint n, uint len, uint gid) {
    uint q = len >> 1;
    uint groups = n / (len << 1);
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + len;
    uint i3 = i2 + q;
    GF31 A = a[i0];
    GF31 B = f31_mul(a[i1], tw[((len >> 1) - 1u) + j]);
    GF31 C = f31_add(A, B);
    GF31 D = f31_sub(A, B);
    GF31 E = a[i2];
    GF31 F = f31_mul(a[i3], tw[((len >> 1) - 1u) + j]);
    GF31 G = f31_add(E, F);
    GF31 H = f31_sub(E, F);
    GF31 TG = f31_mul(G, tw[(len - 1u) + j]);
    GF31 TH = f31_mul(H, tw[(len - 1u) + j + q]);
    a[i0] = f31_add(C, TG);
    a[i2] = f31_sub(C, TG);
    a[i1] = f31_add(D, TH);
    a[i3] = f31_sub(D, TH);
}

inline void crt_local_stage_dif_radix4_31(__local GF31* x,
                                          __global const GF31* twiddles,
                                          const uint n,
                                          const uint len,
                                          const uint lid,
                                          const uint wg) {
    const uint quarter = len >> 2;
    const uint half_len = len >> 1;
    const uint total = n >> 2;
    const uint tw_offset1 = half_len - 1u;
    const uint tw_offset2 = quarter - 1u;

    for (uint t = lid; t < total; t += wg) {
        const uint block = t / quarter;
        const uint j = t - block * quarter;
        const uint base = block * len;

        const uint i0 = base + j;
        const uint i1 = i0 + quarter;
        const uint i2 = i1 + quarter;
        const uint i3 = i2 + quarter;

        const GF31 a0 = x[i0];
        const GF31 a1 = x[i1];
        const GF31 a2 = x[i2];
        const GF31 a3 = x[i3];

        const GF31 b0 = f31_add(a0, a2);
        const GF31 b2 = f31_mul(f31_sub(a0, a2), twiddles[tw_offset1 + j]);
        const GF31 b1 = f31_add(a1, a3);
        const GF31 b3 = f31_mul(f31_sub(a1, a3), twiddles[tw_offset1 + j + quarter]);

        x[i0] = f31_add(b0, b1);
        x[i1] = f31_mul(f31_sub(b0, b1), twiddles[tw_offset2 + j]);
        x[i2] = f31_add(b2, b3);
        x[i3] = f31_mul(f31_sub(b2, b3), twiddles[tw_offset2 + j]);
    }
}

inline void crt_local_stage_dit_radix4_31(__local GF31* x,
                                          __global const GF31* twiddles,
                                          const uint n,
                                          const uint len,
                                          const uint lid,
                                          const uint wg) {
    const uint half_len_local = len >> 1;
    const uint total = n >> 2;
    const uint tw_offset1 = half_len_local - 1u;
    const uint tw_offset2 = len - 1u;

    for (uint t = lid; t < total; t += wg) {
        const uint block = t / half_len_local;
        const uint j = t - block * half_len_local;
        const uint base = block * (len << 1);

        const uint i0 = base + j;
        const uint i1 = i0 + half_len_local;
        const uint i2 = i0 + len;
        const uint i3 = i2 + half_len_local;

        const GF31 x0 = x[i0];
        const GF31 x1 = x[i1];
        const GF31 x2 = x[i2];
        const GF31 x3 = x[i3];

        const GF31 y0 = f31_add(x0, f31_mul(x1, twiddles[tw_offset1 + j]));
        const GF31 y1 = f31_sub(x0, f31_mul(x1, twiddles[tw_offset1 + j]));
        const GF31 y2 = f31_add(x2, f31_mul(x3, twiddles[tw_offset1 + j]));
        const GF31 y3 = f31_sub(x2, f31_mul(x3, twiddles[tw_offset1 + j]));

        x[i0] = f31_add(y0, f31_mul(y2, twiddles[tw_offset2 + j]));
        x[i2] = f31_sub(y0, f31_mul(y2, twiddles[tw_offset2 + j]));
        x[i1] = f31_add(y1, f31_mul(y3, twiddles[tw_offset2 + j + half_len_local]));
        x[i3] = f31_sub(y1, f31_mul(y3, twiddles[tw_offset2 + j + half_len_local]));
    }
}


inline void crt_local_stage_dif_pow2_31(__local GF31* x,
                                        __global const GF31* twiddles,
                                        const uint chunk,
                                        const uint len,
                                        const uint lid,
                                        const uint wg) {
    const uint half_len = len >> 1;
    const uint tw_offset = half_len - 1u;
    const uint butterflies = chunk >> 1;

    for (uint t = lid; t < butterflies; t += wg) {
        const uint block = t / half_len;
        const uint j = t - block * half_len;
        const uint i0 = block * len + j;
        const uint i1 = i0 + half_len;
        const GF31 u = x[i0];
        const GF31 v = x[i1];
        x[i0] = f31_add(u, v);
        x[i1] = f31_mul(f31_sub(u, v), twiddles[tw_offset + j]);
    }
}

inline void crt_local_stage_dit_pow2_31(__local GF31* x,
                                        __global const GF31* twiddles,
                                        const uint chunk,
                                        const uint len,
                                        const uint lid,
                                        const uint wg) {
    const uint half_len = len >> 1;
    const uint tw_offset = half_len - 1u;
    const uint butterflies = chunk >> 1;

    for (uint t = lid; t < butterflies; t += wg) {
        const uint block = t / half_len;
        const uint j = t - block * half_len;
        const uint i0 = block * len + j;
        const uint i1 = i0 + half_len;
        const GF31 u = x[i0];
        const GF31 v = f31_mul(x[i1], twiddles[tw_offset + j]);
        x[i0] = f31_add(u, v);
        x[i1] = f31_sub(u, v);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_first_stage_dif_radix4_wg64(
    __global const ulong* digits,
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_61, uint lr2_31, uint n, uint len)
{
    uint gid = get_global_id(0);
    uint q = len >> 2;
    uint h = len >> 1;
    uint total = n >> 2;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + h;
    uint i3 = i2 + q;
    uint r61_0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    uint r61_1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    uint r61_2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    uint r61_3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));
    GF A = GF61_WEIGHT_LOAD(digits, i0, r61_0, lr2_61);
    GF B = GF61_WEIGHT_LOAD(digits, i2, r61_2, lr2_61);
    GF C = gf_add(A, B);
    GF D = gf_mul(gf_sub(A, B), tw61[((len >> 1) - 1u) + j]);
    A = GF61_WEIGHT_LOAD(digits, i1, r61_1, lr2_61);
    B = GF61_WEIGHT_LOAD(digits, i3, r61_3, lr2_61);
    GF E = gf_add(A, B);
    GF F = gf_mul(gf_sub(A, B), tw61[((len >> 1) - 1u) + j + q]);
    GF W = tw61[((h >> 1) - 1u) + j];
    a61[i0] = gf_add(C, E);
    a61[i1] = gf_mul(gf_sub(C, E), W);
    a61[i2] = gf_add(D, F);
    a61[i3] = gf_mul(gf_sub(D, F), W);

    GF31 A31 = f31_weight_digit(digits[i0], i0, p, n, lr2_31);
    GF31 B31 = f31_weight_digit(digits[i2], i2, p, n, lr2_31);
    GF31 C31 = f31_add(A31, B31);
    GF31 D31 = f31_mul(f31_sub(A31, B31), tw31[((len >> 1) - 1u) + j]);
    A31 = f31_weight_digit(digits[i1], i1, p, n, lr2_31);
    B31 = f31_weight_digit(digits[i3], i3, p, n, lr2_31);
    GF31 E31 = f31_add(A31, B31);
    GF31 F31 = f31_mul(f31_sub(A31, B31), tw31[((len >> 1) - 1u) + j + q]);
    GF31 W31 = tw31[((h >> 1) - 1u) + j];
    a31[i0] = f31_add(C31, E31);
    a31[i1] = f31_mul(f31_sub(C31, E31), W31);
    a31[i2] = f31_add(D31, F31);
    a31[i3] = f31_mul(f31_sub(D31, F31), W31);
}


static inline __attribute__((always_inline)) void crt_radix8_dif_61(__global GF* restrict a, __global const GF* restrict tw, uint n, uint len, uint gid) {
    uint q = len >> 3;
    uint total = n >> 3;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 3;

    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i1 + q;
    uint i3 = i2 + q;
    uint i4 = i3 + q;
    uint i5 = i4 + q;
    uint i6 = i5 + q;
    uint i7 = i6 + q;

    GF x0 = a[i0], x1 = a[i1], x2 = a[i2], x3 = a[i3];
    GF x4 = a[i4], x5 = a[i5], x6 = a[i6], x7 = a[i7];

    const uint o1 = (len >> 1) - 1u;
    const GF w10 = tw[o1 + j];
    const GF w11 = tw[o1 + j + q];
    const GF w12 = tw[o1 + j + (q << 1)];
    const GF w13 = tw[o1 + j + (q * 3u)];
    GF t0 = gf_add(x0, x4); GF t4 = gf_mul(gf_sub(x0, x4), w10);
    GF t1 = gf_add(x1, x5); GF t5 = gf_mul(gf_sub(x1, x5), w11);
    GF t2 = gf_add(x2, x6); GF t6 = gf_mul(gf_sub(x2, x6), w12);
    GF t3 = gf_add(x3, x7); GF t7 = gf_mul(gf_sub(x3, x7), w13);

    const uint o2 = (len >> 2) - 1u;
    const GF w20 = tw[o2 + j];
    const GF w21 = tw[o2 + j + q];
    x0 = gf_add(t0, t2); x2 = gf_mul(gf_sub(t0, t2), w20);
    x1 = gf_add(t1, t3); x3 = gf_mul(gf_sub(t1, t3), w21);
    x4 = gf_add(t4, t6); x6 = gf_mul(gf_sub(t4, t6), w20);
    x5 = gf_add(t5, t7); x7 = gf_mul(gf_sub(t5, t7), w21);

    const uint o3 = q - 1u;
    const GF w30 = tw[o3 + j];
    t0 = gf_add(x0, x1); t1 = gf_mul(gf_sub(x0, x1), w30);
    t2 = gf_add(x2, x3); t3 = gf_mul(gf_sub(x2, x3), w30);
    t4 = gf_add(x4, x5); t5 = gf_mul(gf_sub(x4, x5), w30);
    t6 = gf_add(x6, x7); t7 = gf_mul(gf_sub(x6, x7), w30);

    a[i0] = t0; a[i1] = t1; a[i2] = t2; a[i3] = t3;
    a[i4] = t4; a[i5] = t5; a[i6] = t6; a[i7] = t7;
}

static inline __attribute__((always_inline)) void crt_radix8_dif_31(__global GF31* restrict a, __global const GF31* restrict tw, uint n, uint len, uint gid) {
    uint q = len >> 3;
    uint total = n >> 3;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 3;

    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i1 + q;
    uint i3 = i2 + q;
    uint i4 = i3 + q;
    uint i5 = i4 + q;
    uint i6 = i5 + q;
    uint i7 = i6 + q;

    GF31 x0 = a[i0], x1 = a[i1], x2 = a[i2], x3 = a[i3];
    GF31 x4 = a[i4], x5 = a[i5], x6 = a[i6], x7 = a[i7];

    const uint o1 = (len >> 1) - 1u;
    const GF31 w10 = tw[o1 + j];
    const GF31 w11 = tw[o1 + j + q];
    const GF31 w12 = tw[o1 + j + (q << 1)];
    const GF31 w13 = tw[o1 + j + (q * 3u)];
    GF31 t0 = f31_add(x0, x4); GF31 t4 = f31_mul(f31_sub(x0, x4), w10);
    GF31 t1 = f31_add(x1, x5); GF31 t5 = f31_mul(f31_sub(x1, x5), w11);
    GF31 t2 = f31_add(x2, x6); GF31 t6 = f31_mul(f31_sub(x2, x6), w12);
    GF31 t3 = f31_add(x3, x7); GF31 t7 = f31_mul(f31_sub(x3, x7), w13);

    const uint o2 = (len >> 2) - 1u;
    const GF31 w20 = tw[o2 + j];
    const GF31 w21 = tw[o2 + j + q];
    x0 = f31_add(t0, t2); x2 = f31_mul(f31_sub(t0, t2), w20);
    x1 = f31_add(t1, t3); x3 = f31_mul(f31_sub(t1, t3), w21);
    x4 = f31_add(t4, t6); x6 = f31_mul(f31_sub(t4, t6), w20);
    x5 = f31_add(t5, t7); x7 = f31_mul(f31_sub(t5, t7), w21);

    const uint o3 = q - 1u;
    const GF31 w30 = tw[o3 + j];
    t0 = f31_add(x0, x1); t1 = f31_mul(f31_sub(x0, x1), w30);
    t2 = f31_add(x2, x3); t3 = f31_mul(f31_sub(x2, x3), w30);
    t4 = f31_add(x4, x5); t5 = f31_mul(f31_sub(x4, x5), w30);
    t6 = f31_add(x6, x7); t7 = f31_mul(f31_sub(x6, x7), w30);

    a[i0] = t0; a[i1] = t1; a[i2] = t2; a[i3] = t3;
    a[i4] = t4; a[i5] = t5; a[i6] = t6; a[i7] = t7;
}

static inline __attribute__((always_inline)) void crt_radix8_dit_61(__global GF* restrict a, __global const GF* restrict tw, uint n, uint len, uint gid) {
    uint q = len >> 1;
    uint total = n >> 3;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 3;

    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i1 + q;
    uint i3 = i2 + q;
    uint i4 = i3 + q;
    uint i5 = i4 + q;
    uint i6 = i5 + q;
    uint i7 = i6 + q;

    GF x0 = a[i0], x1 = a[i1], x2 = a[i2], x3 = a[i3];
    GF x4 = a[i4], x5 = a[i5], x6 = a[i6], x7 = a[i7];

    const uint o1 = q - 1u;
    const GF w10 = tw[o1 + j];
    GF y0 = gf_mul(x1, w10);
    GF y1 = gf_mul(x3, w10);
    GF y2 = gf_mul(x5, w10);
    GF y3 = gf_mul(x7, w10);
    x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
    x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
    x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);

    const uint o2 = len - 1u;
    const GF w20 = tw[o2 + j];
    const GF w21 = tw[o2 + j + q];
    y0 = gf_mul(x2, w20);
    y1 = gf_mul(x3, w21);
    GF y4 = gf_mul(x6, w20);
    GF y5 = gf_mul(x7, w21);
    x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
    x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);

    const uint o3 = (len << 1) - 1u;
    const GF w30 = tw[o3 + j];
    const GF w31 = tw[o3 + j + q];
    const GF w32 = tw[o3 + j + (q << 1)];
    const GF w33 = tw[o3 + j + (q * 3u)];
    y0 = gf_mul(x4, w30);
    y1 = gf_mul(x5, w31);
    y2 = gf_mul(x6, w32);
    y3 = gf_mul(x7, w33);
    x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
    x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);

    a[i0] = x0; a[i1] = x1; a[i2] = x2; a[i3] = x3;
    a[i4] = x4; a[i5] = x5; a[i6] = x6; a[i7] = x7;
}

static inline __attribute__((always_inline)) void crt_radix8_dit_31(__global GF31* restrict a, __global const GF31* restrict tw, uint n, uint len, uint gid) {
    uint q = len >> 1;
    uint total = n >> 3;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 3;

    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i1 + q;
    uint i3 = i2 + q;
    uint i4 = i3 + q;
    uint i5 = i4 + q;
    uint i6 = i5 + q;
    uint i7 = i6 + q;

    GF31 x0 = a[i0], x1 = a[i1], x2 = a[i2], x3 = a[i3];
    GF31 x4 = a[i4], x5 = a[i5], x6 = a[i6], x7 = a[i7];

    const uint o1 = q - 1u;
    const GF31 w10 = tw[o1 + j];
    GF31 y0 = f31_mul(x1, w10);
    GF31 y1 = f31_mul(x3, w10);
    GF31 y2 = f31_mul(x5, w10);
    GF31 y3 = f31_mul(x7, w10);
    x1 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
    x3 = f31_sub(x2, y1); x2 = f31_add(x2, y1);
    x5 = f31_sub(x4, y2); x4 = f31_add(x4, y2);
    x7 = f31_sub(x6, y3); x6 = f31_add(x6, y3);

    const uint o2 = len - 1u;
    const GF31 w20 = tw[o2 + j];
    const GF31 w21 = tw[o2 + j + q];
    y0 = f31_mul(x2, w20);
    y1 = f31_mul(x3, w21);
    GF31 y4 = f31_mul(x6, w20);
    GF31 y5 = f31_mul(x7, w21);
    x2 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
    x3 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
    x6 = f31_sub(x4, y4); x4 = f31_add(x4, y4);
    x7 = f31_sub(x5, y5); x5 = f31_add(x5, y5);

    const uint o3 = (len << 1) - 1u;
    const GF31 w30 = tw[o3 + j];
    const GF31 w31 = tw[o3 + j + q];
    const GF31 w32 = tw[o3 + j + (q << 1)];
    const GF31 w33 = tw[o3 + j + (q * 3u)];
    y0 = f31_mul(x4, w30);
    y1 = f31_mul(x5, w31);
    y2 = f31_mul(x6, w32);
    y3 = f31_mul(x7, w33);
    x4 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
    x5 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
    x6 = f31_sub(x2, y2); x2 = f31_add(x2, y2);
    x7 = f31_sub(x3, y3); x3 = f31_add(x3, y3);

    a[i0] = x0; a[i1] = x1; a[i2] = x2; a[i3] = x3;
    a[i4] = x4; a[i5] = x5; a[i6] = x6; a[i7] = x7;
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix2(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len)
{
    uint gid = get_global_id(0);
    uint h = len >> 1;
    uint total = n >> 1;
    if (gid >= total) return;
    uint j = gid & (h - 1u);
    uint base = (gid - j) << 1;
    uint i0 = base + j;
    uint i1 = i0 + h;

    GF u61 = a61[i0];
    GF v61 = a61[i1];
    a61[i0] = gf_add(u61, v61);
    a61[i1] = gf_mul(gf_sub(u61, v61), tw61[(h - 1u) + j]);

    GF31 u31 = a31[i0];
    GF31 v31 = a31[i1];
    a31[i0] = f31_add(u31, v31);
    a31[i1] = f31_mul(f31_sub(u31, v31), tw31[(h - 1u) + j]);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix2(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len)
{
    uint gid = get_global_id(0);
    uint h = len >> 1;
    uint total = n >> 1;
    if (gid >= total) return;
    uint j = gid & (h - 1u);
    uint base = (gid - j) << 1;
    uint i0 = base + j;
    uint i1 = i0 + h;

    GF u61 = a61[i0];
    GF v61 = gf_mul(a61[i1], tw61[(h - 1u) + j]);
    a61[i0] = gf_add(u61, v61);
    a61[i1] = gf_sub(u61, v61);

    GF31 u31 = a31[i0];
    GF31 v31 = f31_mul(a31[i1], tw31[(h - 1u) + j]);
    a31[i0] = f31_add(u31, v31);
    a31[i1] = f31_sub(u31, v31);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix8(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len)
{
    uint gid = get_global_id(0);
    crt_radix8_dif_61(a61, tw61, n, len, gid);
    crt_radix8_dif_31(a31, tw31, n, len, gid);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix8(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len)
{
    uint gid = get_global_id(0);
    crt_radix8_dit_61(a61, tw61, n, len, gid);
    crt_radix8_dit_31(a31, tw31, n, len, gid);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix4(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len)
{
    uint gid = get_global_id(0);
    crt_radix4_dif_61(a61, tw61, n, len, gid);
    crt_radix4_dif_31(a31, tw31, n, len, gid);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix4(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len)
{
    uint gid = get_global_id(0);
    crt_radix4_dit_61(a61, tw61, n, len, gid);
    crt_radix4_dit_31(a31, tw31, n, len, gid);
}

inline void crt_local_dif_full_61(__local GF* x, __global const GF* tw, uint base, uint len) {
    uint lid = get_local_id(0);
    for (uint L = len; L >= 2u; L >>= 1) {
        uint h = L >> 1;
        for (uint t = lid; t < (len >> 1); t += get_local_size(0)) {
            uint group = t / h;
            uint j = t & (h - 1u);
            uint i0 = group * L + j;
            uint i1 = i0 + h;
            GF a = x[i0];
            GF b = x[i1];
            x[i0] = gf_add(a, b);
            x[i1] = gf_mul(gf_sub(a, b), tw[(h - 1u) + j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 2u) break;
    }
}
inline void crt_local_dif_full_31(__local GF31* x, __global const GF31* tw, uint base, uint len) {
    uint lid = get_local_id(0);
    for (uint L = len; L >= 2u; L >>= 1) {
        uint h = L >> 1;
        for (uint t = lid; t < (len >> 1); t += get_local_size(0)) {
            uint group = t / h;
            uint j = t & (h - 1u);
            uint i0 = group * L + j;
            uint i1 = i0 + h;
            GF31 a = x[i0];
            GF31 b = x[i1];
            x[i0] = f31_add(a, b);
            x[i1] = f31_mul(f31_sub(a, b), tw[(h - 1u) + j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 2u) break;
    }
}
inline void crt_local_dit_full_61(__local GF* x, __global const GF* tw, uint base, uint len) {
    uint lid = get_local_id(0);
    for (uint L = 2u; L <= len; L <<= 1) {
        uint h = L >> 1;
        for (uint t = lid; t < (len >> 1); t += get_local_size(0)) {
            uint group = t / h;
            uint j = t & (h - 1u);
            uint i0 = group * L + j;
            uint i1 = i0 + h;
            GF a = x[i0];
            GF b = gf_mul(x[i1], tw[(h - 1u) + j]);
            x[i0] = gf_add(a, b);
            x[i1] = gf_sub(a, b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
inline void crt_local_dit_full_31(__local GF31* x, __global const GF31* tw, uint base, uint len) {
    uint lid = get_local_id(0);
    for (uint L = 2u; L <= len; L <<= 1) {
        uint h = L >> 1;
        for (uint t = lid; t < (len >> 1); t += get_local_size(0)) {
            uint group = t / h;
            uint j = t & (h - 1u);
            uint i0 = group * L + j;
            uint i1 = i0 + h;
            GF31 a = x[i0];
            GF31 b = f31_mul(x[i1], tw[(h - 1u) + j]);
            x[i0] = f31_add(a, b);
            x[i1] = f31_sub(a, b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


static inline __attribute__((always_inline)) void crt_local_stage_dif_radix8_31(__local GF31* x,
                                          __global const GF31* tw,
                                          const uint chunk,
                                          const uint len,
                                          const uint lid,
                                          const uint wg) {
    const uint q = len >> 3;
    const uint total = chunk >> 3;
    const uint o1 = (len >> 1) - 1u;
    const uint o2 = (len >> 2) - 1u;
    const uint o3 = q - 1u;
    for (uint t = lid; t < total; t += wg) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * len;
        const uint i0 = base + j;
        const uint i1 = i0 + q;
        const uint i2 = i1 + q;
        const uint i3 = i2 + q;
        const uint i4 = i3 + q;
        const uint i5 = i4 + q;
        const uint i6 = i5 + q;
        const uint i7 = i6 + q;

        GF31 x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
        GF31 x4 = x[i4], x5 = x[i5], x6 = x[i6], x7 = x[i7];

        if (len == 8u) {
            
            
            GF31 t0 = f31_add(x0, x4); GF31 t4 = f31_sub(x0, x4);
            GF31 t1 = f31_add(x1, x5); GF31 t5 = f31_mul_w8_fast(f31_sub(x1, x5));
            GF31 t2 = f31_add(x2, x6); GF31 t6 = f31_mul_minus_i_fast(f31_sub(x2, x6));
            GF31 t3 = f31_add(x3, x7); GF31 t7 = f31_mul_w8_3_fast(f31_sub(x3, x7));

            x0 = f31_add(t0, t2); x2 = f31_sub(t0, t2);
            x1 = f31_add(t1, t3); x3 = f31_mul_minus_i_fast(f31_sub(t1, t3));
            x4 = f31_add(t4, t6); x6 = f31_sub(t4, t6);
            x5 = f31_add(t5, t7); x7 = f31_mul_minus_i_fast(f31_sub(t5, t7));

            t0 = f31_add(x0, x1); t1 = f31_sub(x0, x1);
            t2 = f31_add(x2, x3); t3 = f31_sub(x2, x3);
            t4 = f31_add(x4, x5); t5 = f31_sub(x4, x5);
            t6 = f31_add(x6, x7); t7 = f31_sub(x6, x7);

            x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
            x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
            continue;
        }

        GF31 t0 = f31_add(x0, x4); GF31 t4 = f31_mul(f31_sub(x0, x4), tw[o1 + j]);
        GF31 t1 = f31_add(x1, x5); GF31 t5 = f31_mul(f31_sub(x1, x5), tw[o1 + j + q]);
        GF31 t2 = f31_add(x2, x6); GF31 t6 = f31_mul(f31_sub(x2, x6), tw[o1 + j + (q << 1)]);
        GF31 t3 = f31_add(x3, x7); GF31 t7 = f31_mul(f31_sub(x3, x7), tw[o1 + j + (q * 3u)]);

        x0 = f31_add(t0, t2); x2 = f31_mul(f31_sub(t0, t2), tw[o2 + j]);
        x1 = f31_add(t1, t3); x3 = f31_mul(f31_sub(t1, t3), tw[o2 + j + q]);
        x4 = f31_add(t4, t6); x6 = f31_mul(f31_sub(t4, t6), tw[o2 + j]);
        x5 = f31_add(t5, t7); x7 = f31_mul(f31_sub(t5, t7), tw[o2 + j + q]);

        t0 = f31_add(x0, x1); t1 = f31_mul(f31_sub(x0, x1), tw[o3 + j]);
        t2 = f31_add(x2, x3); t3 = f31_mul(f31_sub(x2, x3), tw[o3 + j]);
        t4 = f31_add(x4, x5); t5 = f31_mul(f31_sub(x4, x5), tw[o3 + j]);
        t6 = f31_add(x6, x7); t7 = f31_mul(f31_sub(x6, x7), tw[o3 + j]);

        x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
        x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
    }
}

static inline __attribute__((always_inline)) void crt_local_stage_dit_radix8_31(__local GF31* x,
                                          __global const GF31* tw,
                                          const uint chunk,
                                          const uint len,
                                          const uint lid,
                                          const uint wg) {
    const uint q = len >> 1;
    const uint total = chunk >> 3;
    const uint o1 = q - 1u;
    const uint o2 = len - 1u;
    const uint o3 = (len << 1) - 1u;
    for (uint t = lid; t < total; t += wg) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * (len << 2);
        const uint i0 = base + j;
        const uint i1 = i0 + q;
        const uint i2 = i1 + q;
        const uint i3 = i2 + q;
        const uint i4 = i3 + q;
        const uint i5 = i4 + q;
        const uint i6 = i5 + q;
        const uint i7 = i6 + q;

        GF31 x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
        GF31 x4 = x[i4], x5 = x[i5], x6 = x[i6], x7 = x[i7];

        if (len == 2u) {
            
            GF31 y0 = x1;
            GF31 y1 = x3;
            GF31 y2 = x5;
            GF31 y3 = x7;
            x1 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
            x3 = f31_sub(x2, y1); x2 = f31_add(x2, y1);
            x5 = f31_sub(x4, y2); x4 = f31_add(x4, y2);
            x7 = f31_sub(x6, y3); x6 = f31_add(x6, y3);

            y0 = x2;
            y1 = f31_mul_i_fast(x3);
            GF31 y4 = x6;
            GF31 y5 = f31_mul_i_fast(x7);
            x2 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
            x3 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
            x6 = f31_sub(x4, y4); x4 = f31_add(x4, y4);
            x7 = f31_sub(x5, y5); x5 = f31_add(x5, y5);

            y0 = x4;
            y1 = f31_mul_w8_inv_fast(x5);
            y2 = f31_mul_i_fast(x6);
            y3 = f31_mul_w8_inv3_fast(x7);
            x4 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
            x5 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
            x6 = f31_sub(x2, y2); x2 = f31_add(x2, y2);
            x7 = f31_sub(x3, y3); x3 = f31_add(x3, y3);

            x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
            x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
            continue;
        }

        GF31 y0 = f31_mul(x1, tw[o1 + j]);
        GF31 y1 = f31_mul(x3, tw[o1 + j]);
        GF31 y2 = f31_mul(x5, tw[o1 + j]);
        GF31 y3 = f31_mul(x7, tw[o1 + j]);
        x1 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
        x3 = f31_sub(x2, y1); x2 = f31_add(x2, y1);
        x5 = f31_sub(x4, y2); x4 = f31_add(x4, y2);
        x7 = f31_sub(x6, y3); x6 = f31_add(x6, y3);

        y0 = f31_mul(x2, tw[o2 + j]);
        y1 = f31_mul(x3, tw[o2 + j + q]);
        GF31 y4 = f31_mul(x6, tw[o2 + j]);
        GF31 y5 = f31_mul(x7, tw[o2 + j + q]);
        x2 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
        x3 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
        x6 = f31_sub(x4, y4); x4 = f31_add(x4, y4);
        x7 = f31_sub(x5, y5); x5 = f31_add(x5, y5);

        y0 = f31_mul(x4, tw[o3 + j]);
        y1 = f31_mul(x5, tw[o3 + j + q]);
        y2 = f31_mul(x6, tw[o3 + j + (q << 1)]);
        y3 = f31_mul(x7, tw[o3 + j + (q * 3u)]);
        x4 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
        x5 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
        x6 = f31_sub(x2, y2); x2 = f31_add(x2, y2);
        x7 = f31_sub(x3, y3); x3 = f31_add(x3, y3);

        x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
        x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_forward_bridge_512_to_256(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n)
{
    __local GF l61[512];
    __local GF31 l31[512];
    uint lid = get_local_id(0);
    uint block = get_group_id(0);
    uint base = block * 512u;
    for (uint t = lid; t < 512u; t += 128u) { l61[t] = a61[base + t]; l31[t] = a31[base + t]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint t = lid; t < 256u; t += 128u) {
        GF A = l61[t]; GF B = l61[t + 256u];
        l61[t] = gf_add(A, B);
        l61[t + 256u] = gf_mul(gf_sub(A, B), tw61[255u + t]);
        GF31 C = l31[t]; GF31 D = l31[t + 256u];
        l31[t] = f31_add(C, D);
        l31[t + 256u] = f31_mul(f31_sub(C, D), tw31[255u + t]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint t = lid; t < 512u; t += 128u) { a61[base + t] = l61[t]; a31[base + t] = l31[t]; }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_center_fused_256(
    __global GF* a61, __global const GF* twf61, __global const GF* twi61,
    __global GF31* a31, __global const GF31* twf31, __global const GF31* twi31,
    uint n)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint block = get_group_id(0);
    const uint base = block * 256u;

    __local GF l61[256];
    __local GF31 l31[256];

    for (uint t = lid; t < 256u; t += 64u) {
        l61[t] = a61[base + t];
        l31[t] = a31[base + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dif_radix4_pow2(l61, twf61, 256u, 256u, lid, 64u);
    crt_local_stage_dif_radix4_31(l31, twf31, 256u, 256u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(l61, twf61, 256u, 64u,  lid, 64u);
    crt_local_stage_dif_radix4_31(l31, twf31, 256u, 64u,  lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(l61, twf61, 256u, 16u,  lid, 64u);
    crt_local_stage_dif_radix4_31(l31, twf31, 256u, 16u,  lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dif_radix4_pow2(l61, twf61, 256u, 4u,   lid, 64u);
    crt_local_stage_dif_radix4_31(l31, twf31, 256u, 4u,   lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 256u; t += 64u) {
        l61[t] = gf_sqr(l61[t]);
        l31[t] = f31_sqr(l31[t]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dit_radix4_pow2(l61, twi61, 256u, 2u,   lid, 64u);
    crt_local_stage_dit_radix4_31(l31, twi31, 256u, 2u,   lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(l61, twi61, 256u, 8u,   lid, 64u);
    crt_local_stage_dit_radix4_31(l31, twi31, 256u, 8u,   lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(l61, twi61, 256u, 32u,  lid, 64u);
    crt_local_stage_dit_radix4_31(l31, twi31, 256u, 32u,  lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_stage_dit_radix4_pow2(l61, twi61, 256u, 128u, lid, 64u);
    crt_local_stage_dit_radix4_31(l31, twi31, 256u, 128u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 256u; t += 64u) {
        a61[base + t] = l61[t];
        a31[base + t] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_center_fused_256_dualwave(
    __global GF* a61, __global const GF* twf61, __global const GF* twi61,
    __global GF31* a31, __global const GF31* twf31, __global const GF31* twi31,
    uint n)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    const uint block = get_group_id(0);
    const uint base = block * 256u;

    __local GF l61[256];
    __local GF31 l31[256];

    if (is61) {
        for (uint t = lane; t < 256u; t += 64u) l61[t] = a61[base + t];
    } else {
        for (uint t = lane; t < 256u; t += 64u) l31[t] = a31[base + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is61) local_stage_dif_radix4_pow2(l61, twf61, 256u, 256u, lane, 64u);
    else      crt_local_stage_dif_radix4_31(l31, twf31, 256u, 256u, lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is61) local_stage_dif_radix4_pow2(l61, twf61, 256u, 64u,  lane, 64u);
    else      crt_local_stage_dif_radix4_31(l31, twf31, 256u, 64u,  lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is61) local_stage_dif_radix4_pow2(l61, twf61, 256u, 16u,  lane, 64u);
    else      crt_local_stage_dif_radix4_31(l31, twf31, 256u, 16u,  lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is61) local_stage_dif_radix4_pow2(l61, twf61, 256u, 4u,   lane, 64u);
    else      crt_local_stage_dif_radix4_31(l31, twf31, 256u, 4u,   lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is61) {
        for (uint t = lane; t < 256u; t += 64u) l61[t] = gf_sqr(l61[t]);
    } else {
        for (uint t = lane; t < 256u; t += 64u) l31[t] = f31_sqr(l31[t]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is61) local_stage_dit_radix4_pow2(l61, twi61, 256u, 2u,   lane, 64u);
    else      crt_local_stage_dit_radix4_31(l31, twi31, 256u, 2u,   lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is61) local_stage_dit_radix4_pow2(l61, twi61, 256u, 8u,   lane, 64u);
    else      crt_local_stage_dit_radix4_31(l31, twi31, 256u, 8u,   lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is61) local_stage_dit_radix4_pow2(l61, twi61, 256u, 32u,  lane, 64u);
    else      crt_local_stage_dit_radix4_31(l31, twi31, 256u, 32u,  lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is61) local_stage_dit_radix4_pow2(l61, twi61, 256u, 128u, lane, 64u);
    else      crt_local_stage_dit_radix4_31(l31, twi31, 256u, 128u, lane, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is61) {
        for (uint t = lane; t < 256u; t += 64u) a61[base + t] = l61[t];
    } else {
        for (uint t = lane; t < 256u; t += 64u) a31[base + t] = l31[t];
    }
}


#define REG_DIF8_61(X0,X1,X2,X3,X4,X5,X6,X7,TW,LEN,J) do { \
    const uint _q = ((LEN) >> 3); \
    const uint _o1 = ((LEN) >> 1) - 1u; \
    const uint _o2 = ((LEN) >> 2) - 1u; \
    const uint _o3 = _q - 1u; \
    GF _t0 = gf_add((X0), (X4)); GF _t4 = gf_mul(gf_sub((X0), (X4)), (TW)[_o1 + (J)]); \
    GF _t1 = gf_add((X1), (X5)); GF _t5 = gf_mul(gf_sub((X1), (X5)), (TW)[_o1 + (J) + _q]); \
    GF _t2 = gf_add((X2), (X6)); GF _t6 = gf_mul(gf_sub((X2), (X6)), (TW)[_o1 + (J) + (_q << 1)]); \
    GF _t3 = gf_add((X3), (X7)); GF _t7 = gf_mul(gf_sub((X3), (X7)), (TW)[_o1 + (J) + (_q * 3u)]); \
    (X0) = gf_add(_t0, _t2); (X2) = gf_mul(gf_sub(_t0, _t2), (TW)[_o2 + (J)]); \
    (X1) = gf_add(_t1, _t3); (X3) = gf_mul(gf_sub(_t1, _t3), (TW)[_o2 + (J) + _q]); \
    (X4) = gf_add(_t4, _t6); (X6) = gf_mul(gf_sub(_t4, _t6), (TW)[_o2 + (J)]); \
    (X5) = gf_add(_t5, _t7); (X7) = gf_mul(gf_sub(_t5, _t7), (TW)[_o2 + (J) + _q]); \
    _t0 = gf_add((X0), (X1)); _t1 = gf_mul(gf_sub((X0), (X1)), (TW)[_o3 + (J)]); \
    _t2 = gf_add((X2), (X3)); _t3 = gf_mul(gf_sub((X2), (X3)), (TW)[_o3 + (J)]); \
    _t4 = gf_add((X4), (X5)); _t5 = gf_mul(gf_sub((X4), (X5)), (TW)[_o3 + (J)]); \
    _t6 = gf_add((X6), (X7)); _t7 = gf_mul(gf_sub((X6), (X7)), (TW)[_o3 + (J)]); \
    (X0)=_t0; (X1)=_t1; (X2)=_t2; (X3)=_t3; (X4)=_t4; (X5)=_t5; (X6)=_t6; (X7)=_t7; \
} while (0)

#define REG_DIT8_61(X0,X1,X2,X3,X4,X5,X6,X7,TW,LEN,J) do { \
    const uint _q = ((LEN) >> 1); \
    const uint _o1 = _q - 1u; \
    const uint _o2 = (LEN) - 1u; \
    const uint _o3 = ((LEN) << 1) - 1u; \
    GF _y0 = gf_mul((X1), (TW)[_o1 + (J)]); \
    GF _y1 = gf_mul((X3), (TW)[_o1 + (J)]); \
    GF _y2 = gf_mul((X5), (TW)[_o1 + (J)]); \
    GF _y3 = gf_mul((X7), (TW)[_o1 + (J)]); \
    (X1) = gf_sub((X0), _y0); (X0) = gf_add((X0), _y0); \
    (X3) = gf_sub((X2), _y1); (X2) = gf_add((X2), _y1); \
    (X5) = gf_sub((X4), _y2); (X4) = gf_add((X4), _y2); \
    (X7) = gf_sub((X6), _y3); (X6) = gf_add((X6), _y3); \
    _y0 = gf_mul((X2), (TW)[_o2 + (J)]); \
    _y1 = gf_mul((X3), (TW)[_o2 + (J) + _q]); \
    GF _y4 = gf_mul((X6), (TW)[_o2 + (J)]); \
    GF _y5 = gf_mul((X7), (TW)[_o2 + (J) + _q]); \
    (X2) = gf_sub((X0), _y0); (X0) = gf_add((X0), _y0); \
    (X3) = gf_sub((X1), _y1); (X1) = gf_add((X1), _y1); \
    (X6) = gf_sub((X4), _y4); (X4) = gf_add((X4), _y4); \
    (X7) = gf_sub((X5), _y5); (X5) = gf_add((X5), _y5); \
    _y0 = gf_mul((X4), (TW)[_o3 + (J)]); \
    _y1 = gf_mul((X5), (TW)[_o3 + (J) + _q]); \
    _y2 = gf_mul((X6), (TW)[_o3 + (J) + (_q << 1)]); \
    _y3 = gf_mul((X7), (TW)[_o3 + (J) + (_q * 3u)]); \
    (X4) = gf_sub((X0), _y0); (X0) = gf_add((X0), _y0); \
    (X5) = gf_sub((X1), _y1); (X1) = gf_add((X1), _y1); \
    (X6) = gf_sub((X2), _y2); (X2) = gf_add((X2), _y2); \
    (X7) = gf_sub((X3), _y3); (X3) = gf_add((X3), _y3); \
} while (0)

#define REG_DIF8_31(X0,X1,X2,X3,X4,X5,X6,X7,TW,LEN,J) do { \
    const uint _q = ((LEN) >> 3); \
    const uint _o1 = ((LEN) >> 1) - 1u; \
    const uint _o2 = ((LEN) >> 2) - 1u; \
    const uint _o3 = _q - 1u; \
    GF31 _t0 = f31_add((X0), (X4)); GF31 _t4 = f31_mul(f31_sub((X0), (X4)), (TW)[_o1 + (J)]); \
    GF31 _t1 = f31_add((X1), (X5)); GF31 _t5 = f31_mul(f31_sub((X1), (X5)), (TW)[_o1 + (J) + _q]); \
    GF31 _t2 = f31_add((X2), (X6)); GF31 _t6 = f31_mul(f31_sub((X2), (X6)), (TW)[_o1 + (J) + (_q << 1)]); \
    GF31 _t3 = f31_add((X3), (X7)); GF31 _t7 = f31_mul(f31_sub((X3), (X7)), (TW)[_o1 + (J) + (_q * 3u)]); \
    (X0) = f31_add(_t0, _t2); (X2) = f31_mul(f31_sub(_t0, _t2), (TW)[_o2 + (J)]); \
    (X1) = f31_add(_t1, _t3); (X3) = f31_mul(f31_sub(_t1, _t3), (TW)[_o2 + (J) + _q]); \
    (X4) = f31_add(_t4, _t6); (X6) = f31_mul(f31_sub(_t4, _t6), (TW)[_o2 + (J)]); \
    (X5) = f31_add(_t5, _t7); (X7) = f31_mul(f31_sub(_t5, _t7), (TW)[_o2 + (J) + _q]); \
    _t0 = f31_add((X0), (X1)); _t1 = f31_mul(f31_sub((X0), (X1)), (TW)[_o3 + (J)]); \
    _t2 = f31_add((X2), (X3)); _t3 = f31_mul(f31_sub((X2), (X3)), (TW)[_o3 + (J)]); \
    _t4 = f31_add((X4), (X5)); _t5 = f31_mul(f31_sub((X4), (X5)), (TW)[_o3 + (J)]); \
    _t6 = f31_add((X6), (X7)); _t7 = f31_mul(f31_sub((X6), (X7)), (TW)[_o3 + (J)]); \
    (X0)=_t0; (X1)=_t1; (X2)=_t2; (X3)=_t3; (X4)=_t4; (X5)=_t5; (X6)=_t6; (X7)=_t7; \
} while (0)

#define REG_DIT8_31(X0,X1,X2,X3,X4,X5,X6,X7,TW,LEN,J) do { \
    const uint _q = ((LEN) >> 1); \
    const uint _o1 = _q - 1u; \
    const uint _o2 = (LEN) - 1u; \
    const uint _o3 = ((LEN) << 1) - 1u; \
    GF31 _y0 = f31_mul((X1), (TW)[_o1 + (J)]); \
    GF31 _y1 = f31_mul((X3), (TW)[_o1 + (J)]); \
    GF31 _y2 = f31_mul((X5), (TW)[_o1 + (J)]); \
    GF31 _y3 = f31_mul((X7), (TW)[_o1 + (J)]); \
    (X1) = f31_sub((X0), _y0); (X0) = f31_add((X0), _y0); \
    (X3) = f31_sub((X2), _y1); (X2) = f31_add((X2), _y1); \
    (X5) = f31_sub((X4), _y2); (X4) = f31_add((X4), _y2); \
    (X7) = f31_sub((X6), _y3); (X6) = f31_add((X6), _y3); \
    _y0 = f31_mul((X2), (TW)[_o2 + (J)]); \
    _y1 = f31_mul((X3), (TW)[_o2 + (J) + _q]); \
    GF31 _y4 = f31_mul((X6), (TW)[_o2 + (J)]); \
    GF31 _y5 = f31_mul((X7), (TW)[_o2 + (J) + _q]); \
    (X2) = f31_sub((X0), _y0); (X0) = f31_add((X0), _y0); \
    (X3) = f31_sub((X1), _y1); (X1) = f31_add((X1), _y1); \
    (X6) = f31_sub((X4), _y4); (X4) = f31_add((X4), _y4); \
    (X7) = f31_sub((X5), _y5); (X5) = f31_add((X5), _y5); \
    _y0 = f31_mul((X4), (TW)[_o3 + (J)]); \
    _y1 = f31_mul((X5), (TW)[_o3 + (J) + _q]); \
    _y2 = f31_mul((X6), (TW)[_o3 + (J) + (_q << 1)]); \
    _y3 = f31_mul((X7), (TW)[_o3 + (J) + (_q * 3u)]); \
    (X4) = f31_sub((X0), _y0); (X0) = f31_add((X0), _y0); \
    (X5) = f31_sub((X1), _y1); (X1) = f31_add((X1), _y1); \
    (X6) = f31_sub((X2), _y2); (X2) = f31_add((X2), _y2); \
    (X7) = f31_sub((X3), _y3); (X3) = f31_add((X3), _y3); \
} while (0)

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_center_fused_512_reglds(
    __global GF* a61, __global const GF* twf61, __global const GF* twi61,
    __global GF31* a31, __global const GF31* twf31, __global const GF31* twi31,
    uint n)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    const uint block = get_group_id(0);
    const uint base = block * 512u;

    __local GF l61[512];
    __local GF31 l31[512];

    if (is61) {
        GF x0 = a61[base + lane +   0u];
        GF x1 = a61[base + lane +  64u];
        GF x2 = a61[base + lane + 128u];
        GF x3 = a61[base + lane + 192u];
        GF x4 = a61[base + lane + 256u];
        GF x5 = a61[base + lane + 320u];
        GF x6 = a61[base + lane + 384u];
        GF x7 = a61[base + lane + 448u];

        REG_DIF8_61(x0,x1,x2,x3,x4,x5,x6,x7,twf61,512u,lane);
        l61[lane +   0u] = x0; l61[lane +  64u] = x1; l61[lane + 128u] = x2; l61[lane + 192u] = x3;
        l61[lane + 256u] = x4; l61[lane + 320u] = x5; l61[lane + 384u] = x6; l61[lane + 448u] = x7;
    } else {
        GF31 x0 = a31[base + lane +   0u];
        GF31 x1 = a31[base + lane +  64u];
        GF31 x2 = a31[base + lane + 128u];
        GF31 x3 = a31[base + lane + 192u];
        GF31 x4 = a31[base + lane + 256u];
        GF31 x5 = a31[base + lane + 320u];
        GF31 x6 = a31[base + lane + 384u];
        GF31 x7 = a31[base + lane + 448u];

        REG_DIF8_31(x0,x1,x2,x3,x4,x5,x6,x7,twf31,512u,lane);
        l31[lane +   0u] = x0; l31[lane +  64u] = x1; l31[lane + 128u] = x2; l31[lane + 192u] = x3;
        l31[lane + 256u] = x4; l31[lane + 320u] = x5; l31[lane + 384u] = x6; l31[lane + 448u] = x7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint j2 = lane & 7u;
    const uint b2 = (lane >> 3) * 64u + j2;
    if (is61) {
        GF x0 = l61[b2 +  0u]; GF x1 = l61[b2 +  8u]; GF x2 = l61[b2 + 16u]; GF x3 = l61[b2 + 24u];
        GF x4 = l61[b2 + 32u]; GF x5 = l61[b2 + 40u]; GF x6 = l61[b2 + 48u]; GF x7 = l61[b2 + 56u];
        REG_DIF8_61(x0,x1,x2,x3,x4,x5,x6,x7,twf61,64u,j2);
        l61[b2 +  0u] = x0; l61[b2 +  8u] = x1; l61[b2 + 16u] = x2; l61[b2 + 24u] = x3;
        l61[b2 + 32u] = x4; l61[b2 + 40u] = x5; l61[b2 + 48u] = x6; l61[b2 + 56u] = x7;
    } else {
        GF31 x0 = l31[b2 +  0u]; GF31 x1 = l31[b2 +  8u]; GF31 x2 = l31[b2 + 16u]; GF31 x3 = l31[b2 + 24u];
        GF31 x4 = l31[b2 + 32u]; GF31 x5 = l31[b2 + 40u]; GF31 x6 = l31[b2 + 48u]; GF31 x7 = l31[b2 + 56u];
        REG_DIF8_31(x0,x1,x2,x3,x4,x5,x6,x7,twf31,64u,j2);
        l31[b2 +  0u] = x0; l31[b2 +  8u] = x1; l31[b2 + 16u] = x2; l31[b2 + 24u] = x3;
        l31[b2 + 32u] = x4; l31[b2 + 40u] = x5; l31[b2 + 48u] = x6; l31[b2 + 56u] = x7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint b3 = lane << 3;
    if (is61) {
        GF x0 = l61[b3 + 0u]; GF x1 = l61[b3 + 1u]; GF x2 = l61[b3 + 2u]; GF x3 = l61[b3 + 3u];
        GF x4 = l61[b3 + 4u]; GF x5 = l61[b3 + 5u]; GF x6 = l61[b3 + 6u]; GF x7 = l61[b3 + 7u];
        REG_DIF8_61(x0,x1,x2,x3,x4,x5,x6,x7,twf61,8u,0u);
        x0 = gf_sqr(x0); x1 = gf_sqr(x1); x2 = gf_sqr(x2); x3 = gf_sqr(x3);
        x4 = gf_sqr(x4); x5 = gf_sqr(x5); x6 = gf_sqr(x6); x7 = gf_sqr(x7);
        REG_DIT8_61(x0,x1,x2,x3,x4,x5,x6,x7,twi61,2u,0u);
        l61[b3 + 0u] = x0; l61[b3 + 1u] = x1; l61[b3 + 2u] = x2; l61[b3 + 3u] = x3;
        l61[b3 + 4u] = x4; l61[b3 + 5u] = x5; l61[b3 + 6u] = x6; l61[b3 + 7u] = x7;
    } else {
        GF31 x0 = l31[b3 + 0u]; GF31 x1 = l31[b3 + 1u]; GF31 x2 = l31[b3 + 2u]; GF31 x3 = l31[b3 + 3u];
        GF31 x4 = l31[b3 + 4u]; GF31 x5 = l31[b3 + 5u]; GF31 x6 = l31[b3 + 6u]; GF31 x7 = l31[b3 + 7u];
        REG_DIF8_31(x0,x1,x2,x3,x4,x5,x6,x7,twf31,8u,0u);
        x0 = f31_sqr(x0); x1 = f31_sqr(x1); x2 = f31_sqr(x2); x3 = f31_sqr(x3);
        x4 = f31_sqr(x4); x5 = f31_sqr(x5); x6 = f31_sqr(x6); x7 = f31_sqr(x7);
        REG_DIT8_31(x0,x1,x2,x3,x4,x5,x6,x7,twi31,2u,0u);
        l31[b3 + 0u] = x0; l31[b3 + 1u] = x1; l31[b3 + 2u] = x2; l31[b3 + 3u] = x3;
        l31[b3 + 4u] = x4; l31[b3 + 5u] = x5; l31[b3 + 6u] = x6; l31[b3 + 7u] = x7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is61) {
        GF x0 = l61[b2 +  0u]; GF x1 = l61[b2 +  8u]; GF x2 = l61[b2 + 16u]; GF x3 = l61[b2 + 24u];
        GF x4 = l61[b2 + 32u]; GF x5 = l61[b2 + 40u]; GF x6 = l61[b2 + 48u]; GF x7 = l61[b2 + 56u];
        REG_DIT8_61(x0,x1,x2,x3,x4,x5,x6,x7,twi61,16u,j2);
        l61[b2 +  0u] = x0; l61[b2 +  8u] = x1; l61[b2 + 16u] = x2; l61[b2 + 24u] = x3;
        l61[b2 + 32u] = x4; l61[b2 + 40u] = x5; l61[b2 + 48u] = x6; l61[b2 + 56u] = x7;
    } else {
        GF31 x0 = l31[b2 +  0u]; GF31 x1 = l31[b2 +  8u]; GF31 x2 = l31[b2 + 16u]; GF31 x3 = l31[b2 + 24u];
        GF31 x4 = l31[b2 + 32u]; GF31 x5 = l31[b2 + 40u]; GF31 x6 = l31[b2 + 48u]; GF31 x7 = l31[b2 + 56u];
        REG_DIT8_31(x0,x1,x2,x3,x4,x5,x6,x7,twi31,16u,j2);
        l31[b2 +  0u] = x0; l31[b2 +  8u] = x1; l31[b2 + 16u] = x2; l31[b2 + 24u] = x3;
        l31[b2 + 32u] = x4; l31[b2 + 40u] = x5; l31[b2 + 48u] = x6; l31[b2 + 56u] = x7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is61) {
        GF x0 = l61[lane +   0u]; GF x1 = l61[lane +  64u]; GF x2 = l61[lane + 128u]; GF x3 = l61[lane + 192u];
        GF x4 = l61[lane + 256u]; GF x5 = l61[lane + 320u]; GF x6 = l61[lane + 384u]; GF x7 = l61[lane + 448u];
        REG_DIT8_61(x0,x1,x2,x3,x4,x5,x6,x7,twi61,128u,lane);
        a61[base + lane +   0u] = x0; a61[base + lane +  64u] = x1; a61[base + lane + 128u] = x2; a61[base + lane + 192u] = x3;
        a61[base + lane + 256u] = x4; a61[base + lane + 320u] = x5; a61[base + lane + 384u] = x6; a61[base + lane + 448u] = x7;
    } else {
        GF31 x0 = l31[lane +   0u]; GF31 x1 = l31[lane +  64u]; GF31 x2 = l31[lane + 128u]; GF31 x3 = l31[lane + 192u];
        GF31 x4 = l31[lane + 256u]; GF31 x5 = l31[lane + 320u]; GF31 x6 = l31[lane + 384u]; GF31 x7 = l31[lane + 448u];
        REG_DIT8_31(x0,x1,x2,x3,x4,x5,x6,x7,twi31,128u,lane);
        a31[base + lane +   0u] = x0; a31[base + lane +  64u] = x1; a31[base + lane + 128u] = x2; a31[base + lane + 192u] = x3;
        a31[base + lane + 256u] = x4; a31[base + lane + 320u] = x5; a31[base + lane + 384u] = x6; a31[base + lane + 448u] = x7;
    }
}


#define DECL_CRT_CENTER_FUSED_DUALWAVE(NAME, CHUNK) \
__kernel __attribute__((reqd_work_group_size(128,1,1))) \
void NAME( \
    __global GF* a61, __global const GF* twf61, __global const GF* twi61, \
    __global GF31* a31, __global const GF31* twf31, __global const GF31* twi31, \
    uint n) \
{ \
    (void)n; \
    const uint lid = get_local_id(0); \
    const uint lane = lid & 63u; \
    const uint is61 = (lid < 64u); \
    const uint block = get_group_id(0); \
    const uint base = block * (CHUNK); \
    __local GF l61[(CHUNK)]; \
    __local GF31 l31[(CHUNK)]; \
    if (is61) { \
        for (uint t = lane; t < (CHUNK); t += 64u) l61[t] = a61[base + t]; \
    } else { \
        for (uint t = lane; t < (CHUNK); t += 64u) l31[t] = a31[base + t]; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint f_len = (CHUNK); \
    for (; f_len >= 8u; f_len >>= 3) { \
        if (is61) local_stage_dif_radix8_pow2(l61, twf61, (CHUNK), f_len, lane, 64u); \
        else      crt_local_stage_dif_radix8_31(l31, twf31, (CHUNK), f_len, lane, 64u); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (f_len == 8u) { f_len = 1u; break; } \
    } \
    if (f_len == 4u) { \
        if (is61) local_stage_dif_radix4_pow2(l61, twf61, (CHUNK), 4u, lane, 64u); \
        else      crt_local_stage_dif_radix4_31(l31, twf31, (CHUNK), 4u, lane, 64u); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } else if (f_len == 2u) { \
        if (is61) local_stage_dif_pow2(l61, twf61, (CHUNK), 2u, lane, 64u); \
        else      crt_local_stage_dif_pow2_31(l31, twf31, (CHUNK), 2u, lane, 64u); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (is61) { \
        for (uint t = lane; t < (CHUNK); t += 64u) l61[t] = gf_sqr(l61[t]); \
    } else { \
        for (uint t = lane; t < (CHUNK); t += 64u) l31[t] = f31_sqr(l31[t]); \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint i_len = 2u; \
    for (; (i_len << 2) <= (CHUNK); i_len <<= 3) { \
        if (is61) local_stage_dit_radix8_pow2(l61, twi61, (CHUNK), i_len, lane, 64u); \
        else      crt_local_stage_dit_radix8_31(l31, twi31, (CHUNK), i_len, lane, 64u); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if ((i_len << 1) <= (CHUNK)) { \
        if (is61) local_stage_dit_radix4_pow2(l61, twi61, (CHUNK), i_len, lane, 64u); \
        else      crt_local_stage_dit_radix4_31(l31, twi31, (CHUNK), i_len, lane, 64u); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        i_len <<= 2; \
    } \
    if (i_len <= (CHUNK)) { \
        if (is61) local_stage_dit_pow2(l61, twi61, (CHUNK), i_len, lane, 64u); \
        else      crt_local_stage_dit_pow2_31(l31, twi31, (CHUNK), i_len, lane, 64u); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (is61) { \
        for (uint t = lane; t < (CHUNK); t += 64u) a61[base + t] = l61[t]; \
    } else { \
        for (uint t = lane; t < (CHUNK); t += 64u) a31[base + t] = l31[t]; \
    } \
}

DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_8_dualwave, 8u)
DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_16_dualwave, 16u)
DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_32_dualwave, 32u)
DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_64_dualwave, 64u)
DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_128_dualwave, 128u)
DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_512_dualwave, 512u)
DECL_CRT_CENTER_FUSED_DUALWAVE(gf61_crt_center_fused_1024_dualwave, 1024u)


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_first_stage_dif_radix4_wg64_61(
    __global const ulong* digits,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n, uint len)
{
    uint gid = get_global_id(0);
    uint q = len >> 2;
    uint h = len >> 1;
    uint total = n >> 2;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + h;
    uint i3 = i2 + q;
    uint r61_0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    uint r61_1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    uint r61_2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    uint r61_3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));
    GF A = GF61_WEIGHT_LOAD(digits, i0, r61_0, lr2_61);
    GF B = GF61_WEIGHT_LOAD(digits, i2, r61_2, lr2_61);
    GF C = gf_add(A, B);
    GF D = gf_mul(gf_sub(A, B), tw61[((len >> 1) - 1u) + j]);
    A = GF61_WEIGHT_LOAD(digits, i1, r61_1, lr2_61);
    B = GF61_WEIGHT_LOAD(digits, i3, r61_3, lr2_61);
    GF E = gf_add(A, B);
    GF F = gf_mul(gf_sub(A, B), tw61[((len >> 1) - 1u) + j + q]);
    GF W = tw61[((h >> 1) - 1u) + j];
    a61[i0] = gf_add(C, E);
    a61[i1] = gf_mul(gf_sub(C, E), W);
    a61[i2] = gf_add(D, F);
    a61[i3] = gf_mul(gf_sub(D, F), W);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_first_stage_dif_radix4_wg64_31(
    __global const ulong* digits,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n, uint len)
{
    uint gid = get_global_id(0);
    uint q = len >> 2;
    uint h = len >> 1;
    uint total = n >> 2;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + h;
    uint i3 = i2 + q;
    GF31 A31 = f31_weight_digit(digits[i0], i0, p, n, lr2_31);
    GF31 B31 = f31_weight_digit(digits[i2], i2, p, n, lr2_31);
    GF31 C31 = f31_add(A31, B31);
    GF31 D31 = f31_mul(f31_sub(A31, B31), tw31[((len >> 1) - 1u) + j]);
    A31 = f31_weight_digit(digits[i1], i1, p, n, lr2_31);
    B31 = f31_weight_digit(digits[i3], i3, p, n, lr2_31);
    GF31 E31 = f31_add(A31, B31);
    GF31 F31 = f31_mul(f31_sub(A31, B31), tw31[((len >> 1) - 1u) + j + q]);
    GF31 W31 = tw31[((h >> 1) - 1u) + j];
    a31[i0] = f31_add(C31, E31);
    a31[i1] = f31_mul(f31_sub(C31, E31), W31);
    a31[i2] = f31_add(D31, F31);
    a31[i3] = f31_mul(f31_sub(D31, F31), W31);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix8_61(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    crt_radix8_dif_61(a61, tw61, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_ntt_stage_dif_radix8_61_wg128(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    crt_radix8_dif_61(a61, tw61, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix8_61(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    crt_radix8_dit_61(a61, tw61, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix4_61(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    crt_radix4_dif_61(a61, tw61, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix4_61(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    crt_radix4_dit_61(a61, tw61, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix2_61(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    uint gid = get_global_id(0);
    uint h = len >> 1;
    uint total = n >> 1;
    if (gid >= total) return;
    uint j = gid & (h - 1u);
    uint base = (gid - j) << 1;
    uint i0 = base + j;
    uint i1 = i0 + h;
    GF u = a61[i0];
    GF v = a61[i1];
    a61[i0] = gf_add(u, v);
    a61[i1] = gf_mul(gf_sub(u, v), tw61[(h - 1u) + j]);
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix2_61(__global GF* a61, __global const GF* tw61, uint n, uint len) {
    uint gid = get_global_id(0);
    uint h = len >> 1;
    uint total = n >> 1;
    if (gid >= total) return;
    uint j = gid & (h - 1u);
    uint base = (gid - j) << 1;
    uint i0 = base + j;
    uint i1 = i0 + h;
    GF u = a61[i0];
    GF v = gf_mul(a61[i1], tw61[(h - 1u) + j]);
    a61[i0] = gf_add(u, v);
    a61[i1] = gf_sub(u, v);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix8_31(__global GF31* a31, __global const GF31* tw31, uint n, uint len) {
    crt_radix8_dif_31(a31, tw31, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix8_31(__global GF31* a31, __global const GF31* tw31, uint n, uint len) {
    crt_radix8_dit_31(a31, tw31, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix4_31(__global GF31* a31, __global const GF31* tw31, uint n, uint len) {
    crt_radix4_dif_31(a31, tw31, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix4_31(__global GF31* a31, __global const GF31* tw31, uint n, uint len) {
    crt_radix4_dit_31(a31, tw31, n, len, get_global_id(0));
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dif_radix2_31(__global GF31* a31, __global const GF31* tw31, uint n, uint len) {
    uint gid = get_global_id(0);
    uint h = len >> 1;
    uint total = n >> 1;
    if (gid >= total) return;
    uint j = gid & (h - 1u);
    uint base = (gid - j) << 1;
    uint i0 = base + j;
    uint i1 = i0 + h;
    GF31 u = a31[i0];
    GF31 v = a31[i1];
    a31[i0] = f31_add(u, v);
    a31[i1] = f31_mul(f31_sub(u, v), tw31[(h - 1u) + j]);
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_ntt_stage_dit_radix2_31(__global GF31* a31, __global const GF31* tw31, uint n, uint len) {
    uint gid = get_global_id(0);
    uint h = len >> 1;
    uint total = n >> 1;
    if (gid >= total) return;
    uint j = gid & (h - 1u);
    uint base = (gid - j) << 1;
    uint i0 = base + j;
    uint i1 = i0 + h;
    GF31 u = a31[i0];
    GF31 v = f31_mul(a31[i1], tw31[(h - 1u) + j]);
    a31[i0] = f31_add(u, v);
    a31[i1] = f31_sub(u, v);
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_center_fused_512_61(__global GF* a61, __global const GF* twf61, __global const GF* twi61, uint n)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint block = get_group_id(0);
    const uint base = block * 512u;
    __local GF l61[512];
    for (uint t = lid; t < 512u; t += 64u) l61[t] = a61[base + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint f_len = 512u; f_len >= 8u; f_len >>= 3) {
        local_stage_dif_radix8_pow2(l61, twf61, 512u, f_len, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (f_len == 8u) break;
    }
    for (uint t = lid; t < 512u; t += 64u) l61[t] = gf_sqr(l61[t]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i_len = 2u; (i_len << 2) <= 512u; i_len <<= 3) {
        local_stage_dit_radix8_pow2(l61, twi61, 512u, i_len, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[base + t] = l61[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_center_fused_512_31(__global GF31* a31, __global const GF31* twf31, __global const GF31* twi31, uint n)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint block = get_group_id(0);
    const uint base = block * 512u;
    __local GF31 l31[512];
    for (uint t = lid; t < 512u; t += 64u) l31[t] = a31[base + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint f_len = 512u; f_len >= 8u; f_len >>= 3) {
        crt_local_stage_dif_radix8_31(l31, twf31, 512u, f_len, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (f_len == 8u) break;
    }
    for (uint t = lid; t < 512u; t += 64u) l31[t] = f31_sqr(l31[t]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i_len = 2u; (i_len << 2) <= 512u; i_len <<= 3) {
        crt_local_stage_dit_radix8_31(l31, twi31, 512u, i_len, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[base + t] = l31[t];
}


static inline __attribute__((always_inline))
void local_stage_dif_radix8_len512_gload_storeT65_512_wg64(__global const GF* restrict a,
                                                           __local GF* t,
                                                           __global const GF* restrict tw,
                                                           const u32 gbase,
                                                           const u32 lid)
{
    const u32 b = lid >> 3;
    const u32 c = lid & 7u;
    const u32 j = lid;

    GF x0 = a[gbase + j +   0u];
    GF x1 = a[gbase + j +  64u];
    GF x2 = a[gbase + j + 128u];
    GF x3 = a[gbase + j + 192u];
    GF x4 = a[gbase + j + 256u];
    GF x5 = a[gbase + j + 320u];
    GF x6 = a[gbase + j + 384u];
    GF x7 = a[gbase + j + 448u];

    GF t0 = gf_add(x0, x4);
    GF t4 = gf_mul(gf_sub(x0, x4), tw[255u + j]);

    GF t1 = gf_add(x1, x5);
    GF t5 = gf_mul(gf_sub(x1, x5), tw[255u + j + 64u]);

    GF t2 = gf_add(x2, x6);
    GF t6 = gf_mul(gf_sub(x2, x6), tw[255u + j + 128u]);

    GF t3 = gf_add(x3, x7);
    GF t7 = gf_mul(gf_sub(x3, x7), tw[255u + j + 192u]);

    x0 = gf_add(t0, t2);
    x2 = gf_mul(gf_sub(t0, t2), tw[127u + j]);

    x1 = gf_add(t1, t3);
    x3 = gf_mul(gf_sub(t1, t3), tw[127u + j + 64u]);

    x4 = gf_add(t4, t6);
    x6 = gf_mul(gf_sub(t4, t6), tw[127u + j]);

    x5 = gf_add(t5, t7);
    x7 = gf_mul(gf_sub(t5, t7), tw[127u + j + 64u]);

    t0 = gf_add(x0, x1);
    t1 = gf_mul(gf_sub(x0, x1), tw[63u + j]);

    t2 = gf_add(x2, x3);
    t3 = gf_mul(gf_sub(x2, x3), tw[63u + j]);

    t4 = gf_add(x4, x5);
    t5 = gf_mul(gf_sub(x4, x5), tw[63u + j]);

    t6 = gf_add(x6, x7);
    t7 = gf_mul(gf_sub(x6, x7), tw[63u + j]);

    const u32 dst = b * 65u + c;

    t[dst + 0u * 8u] = t0;
    t[dst + 1u * 8u] = t1;
    t[dst + 2u * 8u] = t2;
    t[dst + 3u * 8u] = t3;
    t[dst + 4u * 8u] = t4;
    t[dst + 5u * 8u] = t5;
    t[dst + 6u * 8u] = t6;
    t[dst + 7u * 8u] = t7;
}

static inline __attribute__((always_inline))
void local_stage_dif_radix8_len64_loadT65_storeN_512_wg64(__local GF* t,
                                                          __local GF* n,
                                                          __global const GF* restrict tw,
                                                          const u32 lid)
{
    const u32 a = lid >> 3;
    const u32 c = lid & 7u;
    const u32 j = c;

    GF x0 = t[0u * 65u + a * 8u + c];
    GF x1 = t[1u * 65u + a * 8u + c];
    GF x2 = t[2u * 65u + a * 8u + c];
    GF x3 = t[3u * 65u + a * 8u + c];
    GF x4 = t[4u * 65u + a * 8u + c];
    GF x5 = t[5u * 65u + a * 8u + c];
    GF x6 = t[6u * 65u + a * 8u + c];
    GF x7 = t[7u * 65u + a * 8u + c];

    GF t0 = gf_add(x0, x4);
    GF t4 = gf_mul(gf_sub(x0, x4), tw[31u + j]);

    GF t1 = gf_add(x1, x5);
    GF t5 = gf_mul(gf_sub(x1, x5), tw[31u + j + 8u]);

    GF t2 = gf_add(x2, x6);
    GF t6 = gf_mul(gf_sub(x2, x6), tw[31u + j + 16u]);

    GF t3 = gf_add(x3, x7);
    GF t7 = gf_mul(gf_sub(x3, x7), tw[31u + j + 24u]);

    x0 = gf_add(t0, t2);
    x2 = gf_mul(gf_sub(t0, t2), tw[15u + j]);

    x1 = gf_add(t1, t3);
    x3 = gf_mul(gf_sub(t1, t3), tw[15u + j + 8u]);

    x4 = gf_add(t4, t6);
    x6 = gf_mul(gf_sub(t4, t6), tw[15u + j]);

    x5 = gf_add(t5, t7);
    x7 = gf_mul(gf_sub(t5, t7), tw[15u + j + 8u]);

    t0 = gf_add(x0, x1);
    t1 = gf_mul(gf_sub(x0, x1), tw[7u + j]);

    t2 = gf_add(x2, x3);
    t3 = gf_mul(gf_sub(x2, x3), tw[7u + j]);

    t4 = gf_add(x4, x5);
    t5 = gf_mul(gf_sub(x4, x5), tw[7u + j]);

    t6 = gf_add(x6, x7);
    t7 = gf_mul(gf_sub(x6, x7), tw[7u + j]);

    const u32 dst = a * 64u + c;

    n[dst + 0u * 8u] = t0;
    n[dst + 1u * 8u] = t1;
    n[dst + 2u * 8u] = t2;
    n[dst + 3u * 8u] = t3;
    n[dst + 4u * 8u] = t4;
    n[dst + 5u * 8u] = t5;
    n[dst + 6u * 8u] = t6;
    n[dst + 7u * 8u] = t7;
}

static inline __attribute__((always_inline))
void local_stage_dif8_sqr_dit8_loadN_storeT65_512_wg64(__local GF* n,
                                                       __local GF* t,
                                                       const u32 lid)
{
    const u32 aidx = lid >> 3;
    const u32 bidx = lid & 7u;
    const u32 base = lid << 3;

    GF x0 = n[base + 0u];
    GF x1 = n[base + 1u];
    GF x2 = n[base + 2u];
    GF x3 = n[base + 3u];
    GF x4 = n[base + 4u];
    GF x5 = n[base + 5u];
    GF x6 = n[base + 6u];
    GF x7 = n[base + 7u];

    GF a, b;

    a = gf_add(x0, x4); b = gf_sub(x0, x4); x0 = a; x4 = b;
    a = gf_add(x1, x5); b = gf_sub(x1, x5); x1 = a; x5 = gf_mul_w8_fast(b);
    a = gf_add(x2, x6); b = gf_sub(x2, x6); x2 = a; x6 = gf_mul_i_fast(b);
    a = gf_add(x3, x7); b = gf_sub(x3, x7); x3 = a; x7 = gf_mul_w8_3_fast(b);

    a = gf_add(x0, x2); b = gf_sub(x0, x2); x0 = a; x2 = b;
    a = gf_add(x1, x3); b = gf_sub(x1, x3); x1 = a; x3 = gf_mul_i_fast(b);
    a = gf_add(x4, x6); b = gf_sub(x4, x6); x4 = a; x6 = b;
    a = gf_add(x5, x7); b = gf_sub(x5, x7); x5 = a; x7 = gf_mul_i_fast(b);

    a = gf_add(x0, x1); b = gf_sub(x0, x1); x0 = a; x1 = b;
    a = gf_add(x2, x3); b = gf_sub(x2, x3); x2 = a; x3 = b;
    a = gf_add(x4, x5); b = gf_sub(x4, x5); x4 = a; x5 = b;
    a = gf_add(x6, x7); b = gf_sub(x6, x7); x6 = a; x7 = b;

    x0 = gf_sqr(x0);
    x1 = gf_sqr(x1);
    x2 = gf_sqr(x2);
    x3 = gf_sqr(x3);
    x4 = gf_sqr(x4);
    x5 = gf_sqr(x5);
    x6 = gf_sqr(x6);
    x7 = gf_sqr(x7);

    GF y0 = x1;
    GF y1 = x3;
    GF y2 = x5;
    GF y3 = x7;

    x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
    x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
    x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);

    y0 = x2;
    y1 = gf_mul_minus_i_fast(x3);
    GF y4 = x6;
    GF y5 = gf_mul_minus_i_fast(x7);

    x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
    x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);

    y0 = x4;
    y1 = gf_mul_w8_inv_fast(x5);
    y2 = gf_mul_minus_i_fast(x6);
    y3 = gf_mul_w8_inv3_fast(x7);

    x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
    x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);

    const u32 dst = bidx * 65u + aidx * 8u;

    t[dst + 0u] = x0;
    t[dst + 1u] = x1;
    t[dst + 2u] = x2;
    t[dst + 3u] = x3;
    t[dst + 4u] = x4;
    t[dst + 5u] = x5;
    t[dst + 6u] = x6;
    t[dst + 7u] = x7;
}

static inline __attribute__((always_inline))
void local_stage_dit_radix8_len16_loadT65_storeN_512_wg64(__local GF* t,
                                                          __local GF* n,
                                                          __global const GF* restrict tw,
                                                          const u32 lid)
{
    const u32 a = lid >> 3;
    const u32 c = lid & 7u;
    const u32 j = c;

    GF x0 = t[0u * 65u + a * 8u + c];
    GF x1 = t[1u * 65u + a * 8u + c];
    GF x2 = t[2u * 65u + a * 8u + c];
    GF x3 = t[3u * 65u + a * 8u + c];
    GF x4 = t[4u * 65u + a * 8u + c];
    GF x5 = t[5u * 65u + a * 8u + c];
    GF x6 = t[6u * 65u + a * 8u + c];
    GF x7 = t[7u * 65u + a * 8u + c];

    GF y0 = gf_mul(x1, tw[7u + j]);
    GF y1 = gf_mul(x3, tw[7u + j]);
    GF y2 = gf_mul(x5, tw[7u + j]);
    GF y3 = gf_mul(x7, tw[7u + j]);

    x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
    x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
    x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);

    y0 = gf_mul(x2, tw[15u + j]);
    y1 = gf_mul(x3, tw[15u + j + 8u]);

    GF y4 = gf_mul(x6, tw[15u + j]);
    GF y5 = gf_mul(x7, tw[15u + j + 8u]);

    x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
    x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);

    y0 = gf_mul(x4, tw[31u + j]);
    y1 = gf_mul(x5, tw[31u + j + 8u]);
    y2 = gf_mul(x6, tw[31u + j + 16u]);
    y3 = gf_mul(x7, tw[31u + j + 24u]);

    x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
    x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);

    const u32 dst = a * 64u + c;

    n[dst + 0u * 8u] = x0;
    n[dst + 1u * 8u] = x1;
    n[dst + 2u * 8u] = x2;
    n[dst + 3u * 8u] = x3;
    n[dst + 4u * 8u] = x4;
    n[dst + 5u * 8u] = x5;
    n[dst + 6u * 8u] = x6;
    n[dst + 7u * 8u] = x7;
}

static inline __attribute__((always_inline))
void local_stage_dit_radix8_len128_loadN_gstore_512_wg64(__local GF* n,
                                                         __global GF* restrict a,
                                                         __global const GF* restrict tw,
                                                         const u32 gbase,
                                                         const u32 lid)
{
    const u32 j = lid;

    GF x0 = n[j +   0u];
    GF x1 = n[j +  64u];
    GF x2 = n[j + 128u];
    GF x3 = n[j + 192u];
    GF x4 = n[j + 256u];
    GF x5 = n[j + 320u];
    GF x6 = n[j + 384u];
    GF x7 = n[j + 448u];

    GF y0 = gf_mul(x1, tw[63u + j]);
    GF y1 = gf_mul(x3, tw[63u + j]);
    GF y2 = gf_mul(x5, tw[63u + j]);
    GF y3 = gf_mul(x7, tw[63u + j]);

    x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
    x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
    x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);

    y0 = gf_mul(x2, tw[127u + j]);
    y1 = gf_mul(x3, tw[127u + j + 64u]);

    GF y4 = gf_mul(x6, tw[127u + j]);
    GF y5 = gf_mul(x7, tw[127u + j + 64u]);

    x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
    x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);

    y0 = gf_mul(x4, tw[255u + j]);
    y1 = gf_mul(x5, tw[255u + j + 64u]);
    y2 = gf_mul(x6, tw[255u + j + 128u]);
    y3 = gf_mul(x7, tw[255u + j + 192u]);

    x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
    x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);

    a[gbase + j +   0u] = x0;
    a[gbase + j +  64u] = x1;
    a[gbase + j + 128u] = x2;
    a[gbase + j + 192u] = x3;
    a[gbase + j + 256u] = x4;
    a[gbase + j + 320u] = x5;
    a[gbase + j + 384u] = x6;
    a[gbase + j + 448u] = x7;
}

static inline __attribute__((always_inline))
void local_stage_dif_radix8_len64_512_wg64(__local GF* restrict x,
                                           __global const GF* restrict tw,
                                           const u32 lid)
{
    const u32 a = lid >> 3;
    const u32 j = lid & 7u;
    const u32 base = a << 6;

    const u32 i0 = base + j;
    const u32 i1 = i0 +  8u;
    const u32 i2 = i0 + 16u;
    const u32 i3 = i0 + 24u;
    const u32 i4 = i0 + 32u;
    const u32 i5 = i0 + 40u;
    const u32 i6 = i0 + 48u;
    const u32 i7 = i0 + 56u;

    GF x0 = x[i0]; GF x1 = x[i1]; GF x2 = x[i2]; GF x3 = x[i3];
    GF x4 = x[i4]; GF x5 = x[i5]; GF x6 = x[i6]; GF x7 = x[i7];
    GF t0, t1, t2, t3, t4, t5, t6, t7;

    const GF w31_0 = tw[31u + j];
    const GF w31_1 = tw[31u + j +  8u];
    const GF w31_2 = tw[31u + j + 16u];
    const GF w31_3 = tw[31u + j + 24u];
    const GF w15_0 = tw[15u + j];
    const GF w15_1 = tw[15u + j +  8u];
    const GF w7    = tw[ 7u + j];

    GF_DIF_MUL(x0, x4, w31_0, t0, t4);
    GF_DIF_MUL(x1, x5, w31_1, t1, t5);
    GF_DIF_MUL(x2, x6, w31_2, t2, t6);
    GF_DIF_MUL(x3, x7, w31_3, t3, t7);

    GF_DIF_MUL(t0, t2, w15_0, x0, x2);
    GF_DIF_MUL(t1, t3, w15_1, x1, x3);
    GF_DIF_MUL(t4, t6, w15_0, x4, x6);
    GF_DIF_MUL(t5, t7, w15_1, x5, x7);

    GF_DIF_MUL(x0, x1, w7, t0, t1);
    GF_DIF_MUL(x2, x3, w7, t2, t3);
    GF_DIF_MUL(x4, x5, w7, t4, t5);
    GF_DIF_MUL(x6, x7, w7, t6, t7);

    x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
    x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
}


static inline __attribute__((always_inline))
void local_stage_dit_radix8_len16_512_wg64(__local GF* restrict x,
                                           __global const GF* restrict tw,
                                           const u32 lid)
{
    const u32 a = lid >> 3;
    const u32 j = lid & 7u;
    const u32 base = a << 6;

    const u32 i0 = base + j;
    const u32 i1 = i0 +  8u;
    const u32 i2 = i0 + 16u;
    const u32 i3 = i0 + 24u;
    const u32 i4 = i0 + 32u;
    const u32 i5 = i0 + 40u;
    const u32 i6 = i0 + 48u;
    const u32 i7 = i0 + 56u;

    GF x0 = x[i0]; GF x1 = x[i1]; GF x2 = x[i2]; GF x3 = x[i3];
    GF x4 = x[i4]; GF x5 = x[i5]; GF x6 = x[i6]; GF x7 = x[i7];

    const GF w7    = tw[ 7u + j];
    const GF w15_0 = tw[15u + j];
    const GF w15_1 = tw[15u + j +  8u];
    const GF w31_0 = tw[31u + j];
    const GF w31_1 = tw[31u + j +  8u];
    const GF w31_2 = tw[31u + j + 16u];
    const GF w31_3 = tw[31u + j + 24u];

    GF_DIT_MUL(x0, x1, w7, x0, x1);
    GF_DIT_MUL(x2, x3, w7, x2, x3);
    GF_DIT_MUL(x4, x5, w7, x4, x5);
    GF_DIT_MUL(x6, x7, w7, x6, x7);

    GF_DIT_MUL(x0, x2, w15_0, x0, x2);
    GF_DIT_MUL(x1, x3, w15_1, x1, x3);
    GF_DIT_MUL(x4, x6, w15_0, x4, x6);
    GF_DIT_MUL(x5, x7, w15_1, x5, x7);

    GF_DIT_MUL(x0, x4, w31_0, x0, x4);
    GF_DIT_MUL(x1, x5, w31_1, x1, x5);
    GF_DIT_MUL(x2, x6, w31_2, x2, x6);
    GF_DIT_MUL(x3, x7, w31_3, x3, x7);

    x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
    x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
}


static inline __attribute__((always_inline))
void local_stage_tw56_load_512_wg64(__local GF* restrict twl,
                                    __global const GF* restrict tw,
                                    const u32 lid)
{
    if (lid < 56u) twl[lid] = tw[7u + lid];
}

static inline __attribute__((always_inline))
void local_stage_dif_radix8_len64_512_wg64_twl(__local GF* restrict x,
                                               __local GF* restrict twl,
                                               const u32 lid)
{
    const u32 a = lid >> 3;
    const u32 j = lid & 7u;
    const u32 base = a << 6;

    const u32 i0 = base + j;
    const u32 i1 = i0 +  8u;
    const u32 i2 = i0 + 16u;
    const u32 i3 = i0 + 24u;
    const u32 i4 = i0 + 32u;
    const u32 i5 = i0 + 40u;
    const u32 i6 = i0 + 48u;
    const u32 i7 = i0 + 56u;

    GF x0 = x[i0]; GF x1 = x[i1]; GF x2 = x[i2]; GF x3 = x[i3];
    GF x4 = x[i4]; GF x5 = x[i5]; GF x6 = x[i6]; GF x7 = x[i7];
    GF t0, t1, t2, t3, t4, t5, t6, t7;

    const GF w7    = twl[ 0u + j];
    const GF w15_0 = twl[ 8u + j];
    const GF w15_1 = twl[16u + j];
    const GF w31_0 = twl[24u + j];
    const GF w31_1 = twl[32u + j];
    const GF w31_2 = twl[40u + j];
    const GF w31_3 = twl[48u + j];

    GF_DIF_MUL(x0, x4, w31_0, t0, t4);
    GF_DIF_MUL(x1, x5, w31_1, t1, t5);
    GF_DIF_MUL(x2, x6, w31_2, t2, t6);
    GF_DIF_MUL(x3, x7, w31_3, t3, t7);

    GF_DIF_MUL(t0, t2, w15_0, x0, x2);
    GF_DIF_MUL(t1, t3, w15_1, x1, x3);
    GF_DIF_MUL(t4, t6, w15_0, x4, x6);
    GF_DIF_MUL(t5, t7, w15_1, x5, x7);

    GF_DIF_MUL(x0, x1, w7, t0, t1);
    GF_DIF_MUL(x2, x3, w7, t2, t3);
    GF_DIF_MUL(x4, x5, w7, t4, t5);
    GF_DIF_MUL(x6, x7, w7, t6, t7);

    x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
    x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
}

static inline __attribute__((always_inline))
void local_stage_dit_radix8_len16_512_wg64_twl(__local GF* restrict x,
                                               __local GF* restrict twl,
                                               const u32 lid)
{
    const u32 a = lid >> 3;
    const u32 j = lid & 7u;
    const u32 base = a << 6;

    const u32 i0 = base + j;
    const u32 i1 = i0 +  8u;
    const u32 i2 = i0 + 16u;
    const u32 i3 = i0 + 24u;
    const u32 i4 = i0 + 32u;
    const u32 i5 = i0 + 40u;
    const u32 i6 = i0 + 48u;
    const u32 i7 = i0 + 56u;

    GF x0 = x[i0]; GF x1 = x[i1]; GF x2 = x[i2]; GF x3 = x[i3];
    GF x4 = x[i4]; GF x5 = x[i5]; GF x6 = x[i6]; GF x7 = x[i7];

    const GF w7    = twl[ 0u + j];
    const GF w15_0 = twl[ 8u + j];
    const GF w15_1 = twl[16u + j];
    const GF w31_0 = twl[24u + j];
    const GF w31_1 = twl[32u + j];
    const GF w31_2 = twl[40u + j];
    const GF w31_3 = twl[48u + j];

    GF_DIT_MUL(x0, x1, w7, x0, x1);
    GF_DIT_MUL(x2, x3, w7, x2, x3);
    GF_DIT_MUL(x4, x5, w7, x4, x5);
    GF_DIT_MUL(x6, x7, w7, x6, x7);

    GF_DIT_MUL(x0, x2, w15_0, x0, x2);
    GF_DIT_MUL(x1, x3, w15_1, x1, x3);
    GF_DIT_MUL(x4, x6, w15_0, x4, x6);
    GF_DIT_MUL(x5, x7, w15_1, x5, x7);

    GF_DIT_MUL(x0, x4, w31_0, x0, x4);
    GF_DIT_MUL(x1, x5, w31_1, x1, x5);
    GF_DIT_MUL(x2, x6, w31_2, x2, x6);
    GF_DIT_MUL(x3, x7, w31_3, x3, x7);

    x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
    x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_center_fused_512_61_opt(__global GF* restrict a61,
                                      __global const GF* restrict twf61,
                                      __global const GF* restrict twi61,
                                      uint n)
{
    (void)n;

    const u32 lid  = (u32)get_local_id(0);
    const u32 base = (u32)get_group_id(0) * 512u;

    __local GF l61[512];
    __local GF tw56[56];

    local_stage_dif_radix8_len512_gload_512_wg64(a61, l61, tw56, twf61, base, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dif_radix8_len64_512_wg64_twl(l61, tw56, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dif8_sqr_dit8_512_wg64(l61, tw56, twi61, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dit_radix8_len16_512_wg64_twl(l61, tw56, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    local_stage_dit_radix8_len128_gstore_512_wg64(l61, a61, twi61, base, lid);
}

static inline __attribute__((always_inline))
void crt_local_stage_dif_radix8_len512_gload_31_512_wg64(__global const GF31* restrict a,
                                                         __local GF31* restrict x,
                                                         __local GF31* restrict twl,
                                                         __global const GF31* restrict tw,
                                                         const uint gbase,
                                                         const uint lid)
{
    if (lid < 56u) twl[lid] = tw[7u + lid];
    const uint j = lid;

    GF31 x0 = a[gbase + j +   0u];
    GF31 x1 = a[gbase + j +  64u];
    GF31 x2 = a[gbase + j + 128u];
    GF31 x3 = a[gbase + j + 192u];
    GF31 x4 = a[gbase + j + 256u];
    GF31 x5 = a[gbase + j + 320u];
    GF31 x6 = a[gbase + j + 384u];
    GF31 x7 = a[gbase + j + 448u];

    GF31 t0, t1, t2, t3, t4, t5, t6, t7;
    const GF31 w255_0 = tw[255u + j];
    const GF31 w255_1 = tw[255u + j +  64u];
    const GF31 w255_2 = tw[255u + j + 128u];
    const GF31 w255_3 = tw[255u + j + 192u];
    const GF31 w127_0 = tw[127u + j];
    const GF31 w127_1 = tw[127u + j +  64u];
    const GF31 w63    = tw[ 63u + j];

    F31_DIF_MUL(x0, x4, w255_0, t0, t4);
    F31_DIF_MUL(x1, x5, w255_1, t1, t5);
    F31_DIF_MUL(x2, x6, w255_2, t2, t6);
    F31_DIF_MUL(x3, x7, w255_3, t3, t7);

    F31_DIF_MUL(t0, t2, w127_0, x0, x2);
    F31_DIF_MUL(t1, t3, w127_1, x1, x3);
    F31_DIF_MUL(t4, t6, w127_0, x4, x6);
    F31_DIF_MUL(t5, t7, w127_1, x5, x7);

    F31_DIF_MUL(x0, x1, w63, t0, t1);
    F31_DIF_MUL(x2, x3, w63, t2, t3);
    F31_DIF_MUL(x4, x5, w63, t4, t5);
    F31_DIF_MUL(x6, x7, w63, t6, t7);

    x[j +   0u] = t0; x[j +  64u] = t1; x[j + 128u] = t2; x[j + 192u] = t3;
    x[j + 256u] = t4; x[j + 320u] = t5; x[j + 384u] = t6; x[j + 448u] = t7;
}


static inline __attribute__((always_inline))
void crt_local_stage_dif_radix8_len64_31_512_wg64(__local GF31* restrict x,
                                                  __global const GF31* restrict tw,
                                                  const uint lid)
{
    const uint a = lid >> 3;
    const uint j = lid & 7u;
    const uint base = a << 6;

    const uint i0 = base + j;
    const uint i1 = i0 +  8u;
    const uint i2 = i0 + 16u;
    const uint i3 = i0 + 24u;
    const uint i4 = i0 + 32u;
    const uint i5 = i0 + 40u;
    const uint i6 = i0 + 48u;
    const uint i7 = i0 + 56u;

    GF31 x0 = x[i0]; GF31 x1 = x[i1]; GF31 x2 = x[i2]; GF31 x3 = x[i3];
    GF31 x4 = x[i4]; GF31 x5 = x[i5]; GF31 x6 = x[i6]; GF31 x7 = x[i7];
    GF31 t0, t1, t2, t3, t4, t5, t6, t7;

    const GF31 w31_0 = tw[31u + j];
    const GF31 w31_1 = tw[31u + j +  8u];
    const GF31 w31_2 = tw[31u + j + 16u];
    const GF31 w31_3 = tw[31u + j + 24u];
    const GF31 w15_0 = tw[15u + j];
    const GF31 w15_1 = tw[15u + j +  8u];
    const GF31 w7    = tw[ 7u + j];

    F31_DIF_MUL(x0, x4, w31_0, t0, t4);
    F31_DIF_MUL(x1, x5, w31_1, t1, t5);
    F31_DIF_MUL(x2, x6, w31_2, t2, t6);
    F31_DIF_MUL(x3, x7, w31_3, t3, t7);

    F31_DIF_MUL(t0, t2, w15_0, x0, x2);
    F31_DIF_MUL(t1, t3, w15_1, x1, x3);
    F31_DIF_MUL(t4, t6, w15_0, x4, x6);
    F31_DIF_MUL(t5, t7, w15_1, x5, x7);

    F31_DIF_MUL(x0, x1, w7, t0, t1);
    F31_DIF_MUL(x2, x3, w7, t2, t3);
    F31_DIF_MUL(x4, x5, w7, t4, t5);
    F31_DIF_MUL(x6, x7, w7, t6, t7);

    x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
    x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
}


static inline __attribute__((always_inline))
void crt_local_stage_dif8_sqr_dit8_31_512_wg64(__local GF31* restrict x,
                                               __local GF31* restrict twl,
                                               __global const GF31* restrict twi,
                                               const uint lid)
{
    if (lid < 56u) twl[lid] = twi[7u + lid];
    const uint base = lid << 3;

    GF31 x0 = x[base + 0u]; GF31 x1 = x[base + 1u]; GF31 x2 = x[base + 2u]; GF31 x3 = x[base + 3u];
    GF31 x4 = x[base + 4u]; GF31 x5 = x[base + 5u]; GF31 x6 = x[base + 6u]; GF31 x7 = x[base + 7u];
    GF31 b;

    F31_ADD_SUB(x0, x4, x0, x4);
    F31_ADD_SUB(x1, x5, x1, b); x5 = f31_mul_w8_fast(b);
    F31_ADD_SUB(x2, x6, x2, b); x6 = f31_mul_minus_i_fast(b);
    F31_ADD_SUB(x3, x7, x3, b); x7 = f31_mul_w8_3_fast(b);

    F31_ADD_SUB(x0, x2, x0, x2);
    F31_ADD_SUB(x1, x3, x1, b); x3 = f31_mul_minus_i_fast(b);
    F31_ADD_SUB(x4, x6, x4, x6);
    F31_ADD_SUB(x5, x7, x5, b); x7 = f31_mul_minus_i_fast(b);

    F31_ADD_SUB(x0, x1, x0, x1);
    F31_ADD_SUB(x2, x3, x2, x3);
    F31_ADD_SUB(x4, x5, x4, x5);
    F31_ADD_SUB(x6, x7, x6, x7);

    x0 = f31_sqr(x0); x1 = f31_sqr(x1); x2 = f31_sqr(x2); x3 = f31_sqr(x3);
    x4 = f31_sqr(x4); x5 = f31_sqr(x5); x6 = f31_sqr(x6); x7 = f31_sqr(x7);

    GF31 y0 = x1; GF31 y1 = x3; GF31 y2 = x5; GF31 y3 = x7;
    F31_ADD_SUB(x0, y0, x0, x1);
    F31_ADD_SUB(x2, y1, x2, x3);
    F31_ADD_SUB(x4, y2, x4, x5);
    F31_ADD_SUB(x6, y3, x6, x7);

    y0 = x2;
    y1 = f31_mul_i_fast(x3);
    GF31 y4 = x6;
    GF31 y5 = f31_mul_i_fast(x7);
    F31_ADD_SUB(x0, y0, x0, x2);
    F31_ADD_SUB(x1, y1, x1, x3);
    F31_ADD_SUB(x4, y4, x4, x6);
    F31_ADD_SUB(x5, y5, x5, x7);

    y0 = x4;
    y1 = f31_mul_w8_inv_fast(x5);
    y2 = f31_mul_i_fast(x6);
    y3 = f31_mul_w8_inv3_fast(x7);
    F31_ADD_SUB(x0, y0, x0, x4);
    F31_ADD_SUB(x1, y1, x1, x5);
    F31_ADD_SUB(x2, y2, x2, x6);
    F31_ADD_SUB(x3, y3, x3, x7);

    x[base + 0u] = x0; x[base + 1u] = x1; x[base + 2u] = x2; x[base + 3u] = x3;
    x[base + 4u] = x4; x[base + 5u] = x5; x[base + 6u] = x6; x[base + 7u] = x7;
}


static inline __attribute__((always_inline))
void crt_local_stage_dit_radix8_len16_31_512_wg64(__local GF31* restrict x,
                                                  __global const GF31* restrict tw,
                                                  const uint lid)
{
    const uint a = lid >> 3;
    const uint j = lid & 7u;
    const uint base = a << 6;

    const uint i0 = base + j;
    const uint i1 = i0 +  8u;
    const uint i2 = i0 + 16u;
    const uint i3 = i0 + 24u;
    const uint i4 = i0 + 32u;
    const uint i5 = i0 + 40u;
    const uint i6 = i0 + 48u;
    const uint i7 = i0 + 56u;

    GF31 x0 = x[i0]; GF31 x1 = x[i1]; GF31 x2 = x[i2]; GF31 x3 = x[i3];
    GF31 x4 = x[i4]; GF31 x5 = x[i5]; GF31 x6 = x[i6]; GF31 x7 = x[i7];

    const GF31 w7    = tw[ 7u + j];
    const GF31 w15_0 = tw[15u + j];
    const GF31 w15_1 = tw[15u + j +  8u];
    const GF31 w31_0 = tw[31u + j];
    const GF31 w31_1 = tw[31u + j +  8u];
    const GF31 w31_2 = tw[31u + j + 16u];
    const GF31 w31_3 = tw[31u + j + 24u];

    F31_DIT_MUL(x0, x1, w7, x0, x1);
    F31_DIT_MUL(x2, x3, w7, x2, x3);
    F31_DIT_MUL(x4, x5, w7, x4, x5);
    F31_DIT_MUL(x6, x7, w7, x6, x7);

    F31_DIT_MUL(x0, x2, w15_0, x0, x2);
    F31_DIT_MUL(x1, x3, w15_1, x1, x3);
    F31_DIT_MUL(x4, x6, w15_0, x4, x6);
    F31_DIT_MUL(x5, x7, w15_1, x5, x7);

    F31_DIT_MUL(x0, x4, w31_0, x0, x4);
    F31_DIT_MUL(x1, x5, w31_1, x1, x5);
    F31_DIT_MUL(x2, x6, w31_2, x2, x6);
    F31_DIT_MUL(x3, x7, w31_3, x3, x7);

    x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
    x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
}


static inline __attribute__((always_inline))
void crt_local_stage_dit_radix8_len128_gstore_31_512_wg64(__local GF31* restrict x,
                                                          __global GF31* restrict a,
                                                          __global const GF31* restrict tw,
                                                          const uint gbase,
                                                          const uint lid)
{
    const uint j = lid;

    GF31 x0 = x[j +   0u]; GF31 x1 = x[j +  64u]; GF31 x2 = x[j + 128u]; GF31 x3 = x[j + 192u];
    GF31 x4 = x[j + 256u]; GF31 x5 = x[j + 320u]; GF31 x6 = x[j + 384u]; GF31 x7 = x[j + 448u];

    const GF31 w63    = tw[ 63u + j];
    const GF31 w127_0 = tw[127u + j];
    const GF31 w127_1 = tw[127u + j +  64u];
    const GF31 w255_0 = tw[255u + j];
    const GF31 w255_1 = tw[255u + j +  64u];
    const GF31 w255_2 = tw[255u + j + 128u];
    const GF31 w255_3 = tw[255u + j + 192u];

    F31_DIT_MUL(x0, x1, w63, x0, x1);
    F31_DIT_MUL(x2, x3, w63, x2, x3);
    F31_DIT_MUL(x4, x5, w63, x4, x5);
    F31_DIT_MUL(x6, x7, w63, x6, x7);

    F31_DIT_MUL(x0, x2, w127_0, x0, x2);
    F31_DIT_MUL(x1, x3, w127_1, x1, x3);
    F31_DIT_MUL(x4, x6, w127_0, x4, x6);
    F31_DIT_MUL(x5, x7, w127_1, x5, x7);

    F31_DIT_MUL(x0, x4, w255_0, x0, x4);
    F31_DIT_MUL(x1, x5, w255_1, x1, x5);
    F31_DIT_MUL(x2, x6, w255_2, x2, x6);
    F31_DIT_MUL(x3, x7, w255_3, x3, x7);

    a[gbase + j +   0u] = x0; a[gbase + j +  64u] = x1; a[gbase + j + 128u] = x2; a[gbase + j + 192u] = x3;
    a[gbase + j + 256u] = x4; a[gbase + j + 320u] = x5; a[gbase + j + 384u] = x6; a[gbase + j + 448u] = x7;
}


static inline __attribute__((always_inline))
void crt_local_stage_tw56_load_31_512_wg64(__local GF31* restrict twl,
                                           __global const GF31* restrict tw,
                                           const uint lid)
{
    if (lid < 56u) twl[lid] = tw[7u + lid];
}

static inline __attribute__((always_inline))
void crt_local_stage_dif_radix8_len64_31_512_wg64_twl(__local GF31* restrict x,
                                                      __local GF31* restrict twl,
                                                      const uint lid)
{
    const uint a = lid >> 3;
    const uint j = lid & 7u;
    const uint base = a << 6;

    const uint i0 = base + j;
    const uint i1 = i0 +  8u;
    const uint i2 = i0 + 16u;
    const uint i3 = i0 + 24u;
    const uint i4 = i0 + 32u;
    const uint i5 = i0 + 40u;
    const uint i6 = i0 + 48u;
    const uint i7 = i0 + 56u;

    GF31 x0 = x[i0]; GF31 x1 = x[i1]; GF31 x2 = x[i2]; GF31 x3 = x[i3];
    GF31 x4 = x[i4]; GF31 x5 = x[i5]; GF31 x6 = x[i6]; GF31 x7 = x[i7];
    GF31 t0, t1, t2, t3, t4, t5, t6, t7;

    const GF31 w7    = twl[ 0u + j];
    const GF31 w15_0 = twl[ 8u + j];
    const GF31 w15_1 = twl[16u + j];
    const GF31 w31_0 = twl[24u + j];
    const GF31 w31_1 = twl[32u + j];
    const GF31 w31_2 = twl[40u + j];
    const GF31 w31_3 = twl[48u + j];

    F31_DIF_MUL(x0, x4, w31_0, t0, t4);
    F31_DIF_MUL(x1, x5, w31_1, t1, t5);
    F31_DIF_MUL(x2, x6, w31_2, t2, t6);
    F31_DIF_MUL(x3, x7, w31_3, t3, t7);

    F31_DIF_MUL(t0, t2, w15_0, x0, x2);
    F31_DIF_MUL(t1, t3, w15_1, x1, x3);
    F31_DIF_MUL(t4, t6, w15_0, x4, x6);
    F31_DIF_MUL(t5, t7, w15_1, x5, x7);

    F31_DIF_MUL(x0, x1, w7, t0, t1);
    F31_DIF_MUL(x2, x3, w7, t2, t3);
    F31_DIF_MUL(x4, x5, w7, t4, t5);
    F31_DIF_MUL(x6, x7, w7, t6, t7);

    x[i0] = t0; x[i1] = t1; x[i2] = t2; x[i3] = t3;
    x[i4] = t4; x[i5] = t5; x[i6] = t6; x[i7] = t7;
}

static inline __attribute__((always_inline))
void crt_local_stage_dit_radix8_len16_31_512_wg64_twl(__local GF31* restrict x,
                                                      __local GF31* restrict twl,
                                                      const uint lid)
{
    const uint a = lid >> 3;
    const uint j = lid & 7u;
    const uint base = a << 6;

    const uint i0 = base + j;
    const uint i1 = i0 +  8u;
    const uint i2 = i0 + 16u;
    const uint i3 = i0 + 24u;
    const uint i4 = i0 + 32u;
    const uint i5 = i0 + 40u;
    const uint i6 = i0 + 48u;
    const uint i7 = i0 + 56u;

    GF31 x0 = x[i0]; GF31 x1 = x[i1]; GF31 x2 = x[i2]; GF31 x3 = x[i3];
    GF31 x4 = x[i4]; GF31 x5 = x[i5]; GF31 x6 = x[i6]; GF31 x7 = x[i7];

    const GF31 w7    = twl[ 0u + j];
    const GF31 w15_0 = twl[ 8u + j];
    const GF31 w15_1 = twl[16u + j];
    const GF31 w31_0 = twl[24u + j];
    const GF31 w31_1 = twl[32u + j];
    const GF31 w31_2 = twl[40u + j];
    const GF31 w31_3 = twl[48u + j];

    F31_DIT_MUL(x0, x1, w7, x0, x1);
    F31_DIT_MUL(x2, x3, w7, x2, x3);
    F31_DIT_MUL(x4, x5, w7, x4, x5);
    F31_DIT_MUL(x6, x7, w7, x6, x7);

    F31_DIT_MUL(x0, x2, w15_0, x0, x2);
    F31_DIT_MUL(x1, x3, w15_1, x1, x3);
    F31_DIT_MUL(x4, x6, w15_0, x4, x6);
    F31_DIT_MUL(x5, x7, w15_1, x5, x7);

    F31_DIT_MUL(x0, x4, w31_0, x0, x4);
    F31_DIT_MUL(x1, x5, w31_1, x1, x5);
    F31_DIT_MUL(x2, x6, w31_2, x2, x6);
    F31_DIT_MUL(x3, x7, w31_3, x3, x7);

    x[i0] = x0; x[i1] = x1; x[i2] = x2; x[i3] = x3;
    x[i4] = x4; x[i5] = x5; x[i6] = x6; x[i7] = x7;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_center_fused_512_31_opt(__global GF31* restrict a31,
                                      __global const GF31* restrict twf31,
                                      __global const GF31* restrict twi31,
                                      uint n)
{
    (void)n;

    const uint lid  = get_local_id(0);
    const uint base = get_group_id(0) * 512u;

    __local GF31 l31[512];
    __local GF31 tw56[56];

    crt_local_stage_dif_radix8_len512_gload_31_512_wg64(a31, l31, tw56, twf31, base, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_local_stage_dif_radix8_len64_31_512_wg64_twl(l31, tw56, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_local_stage_dif8_sqr_dit8_31_512_wg64(l31, tw56, twi31, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_local_stage_dit_radix8_len16_31_512_wg64_twl(l31, tw56, lid);
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_local_stage_dit_radix8_len128_gstore_31_512_wg64(l31, a31, twi31, base, lid);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_last_stage_dit_radix4_unweight_wg64_61(
    __global ulong* digits61,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n, uint len, uint log_n)
{
    uint gid = get_global_id(0);
    uint q = len >> 1;
    uint groups = n / (len << 1);
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + len;
    uint i3 = i2 + q;
    GF A = a61[i0]; GF B = gf_mul(a61[i1], tw61[((len >> 1) - 1u) + j]);
    GF C = gf_add(A, B); GF D = gf_sub(A, B);
    GF E = a61[i2]; GF F = gf_mul(a61[i3], tw61[((len >> 1) - 1u) + j]);
    GF G = gf_add(E, F); GF H = gf_sub(E, F);
    GF TG = gf_mul(G, tw61[(len - 1u) + j]);
    GF TH = gf_mul(H, tw61[(len - 1u) + j + q]);
    GF y0 = gf_add(C, TG);
    GF y2 = gf_sub(C, TG);
    GF y1 = gf_add(D, TH);
    GF y3 = gf_sub(D, TH);
    uint ur0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    uint ur1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    uint ur2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    uint ur3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));
    GF61_STORE_UNWEIGHT(digits61, i0, y0, ur0, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i1, y1, ur1, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i2, y2, ur2, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i3, y3, ur3, lr2_61, log_n);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_last_stage_dit_radix4_unweight_wg64_31(
    __global uint* digits31,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n, uint len, uint log_n)
{
    uint gid = get_global_id(0);
    uint q = len >> 1;
    uint groups = n / (len << 1);
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + len;
    uint i3 = i2 + q;
    GF31 A = a31[i0]; GF31 B = f31_mul(a31[i1], tw31[((len >> 1) - 1u) + j]);
    GF31 C = f31_add(A, B); GF31 D = f31_sub(A, B);
    GF31 E = a31[i2]; GF31 F = f31_mul(a31[i3], tw31[((len >> 1) - 1u) + j]);
    GF31 G = f31_add(E, F); GF31 H = f31_sub(E, F);
    GF31 TG = f31_mul(G, tw31[(len - 1u) + j]);
    GF31 TH = f31_mul(H, tw31[(len - 1u) + j + q]);
    GF31 v0 = f31_add(C, TG);
    GF31 v2 = f31_sub(C, TG);
    GF31 v1 = f31_add(D, TH);
    GF31 v3 = f31_sub(D, TH);
    uint ur0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    uint ur1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    uint ur2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    uint ur3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));
    digits31[i0] = f31_unweight_digit_logn(v0, ur0, lr2_31, log_n);
    digits31[i1] = f31_unweight_digit_logn(v1, ur1, lr2_31, log_n);
    digits31[i2] = f31_unweight_digit_logn(v2, ur2, lr2_31, log_n);
    digits31[i3] = f31_unweight_digit_logn(v3, ur3, lr2_31, log_n);
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_first_stage_dif_edge_61(
    __global const ulong* digits,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n, uint log_radix)
{
    const uint R = 1u << log_radix;
    const uint q = n >> log_radix;
    const uint j = get_global_id(0);
    if (j >= q) return;
    GF x[16];
    for (uint k = 0u; k < R; ++k) {
        const uint idx = j + k * q;
        const uint r = (uint)(((ulong)idx * (ulong)p) & (ulong)(n - 1u));
        x[k] = GF61_WEIGHT_LOAD(digits, idx, r, lr2_61);
    }
    for (uint s = 0u; s < log_radix; ++s) {
        const uint span = R >> s;
        const uint h2 = span >> 1;
        const uint len = n >> s;
        const uint twbase = (len >> 1) - 1u;
        for (uint b = 0u; b < R; b += span) {
            for (uint t = 0u; t < h2; ++t) {
                const uint k0 = b + t;
                const uint k1 = k0 + h2;
                GF u = x[k0];
                GF v = x[k1];
                x[k0] = gf_add(u, v);
                x[k1] = gf_mul(gf_sub(u, v), tw61[twbase + j + t * q]);
            }
        }
    }
    for (uint k = 0u; k < R; ++k) a61[j + k * q] = x[k];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_first_stage_dif_edge_31(
    __global const ulong* digits,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n, uint log_radix)
{
    const uint R = 1u << log_radix;
    const uint q = n >> log_radix;
    const uint j = get_global_id(0);
    if (j >= q) return;
    GF31 x[16];
    for (uint k = 0u; k < R; ++k) {
        const uint idx = j + k * q;
        const uint r = (uint)(((ulong)idx * (ulong)p) & (ulong)(n - 1u));
        x[k] = f31_weight_digit(digits[idx], idx, p, n, lr2_31);
    }
    for (uint s = 0u; s < log_radix; ++s) {
        const uint span = R >> s;
        const uint h2 = span >> 1;
        const uint len = n >> s;
        const uint twbase = (len >> 1) - 1u;
        for (uint b = 0u; b < R; b += span) {
            for (uint t = 0u; t < h2; ++t) {
                const uint k0 = b + t;
                const uint k1 = k0 + h2;
                GF31 u = x[k0];
                GF31 v = x[k1];
                x[k0] = f31_add(u, v);
                x[k1] = f31_mul(f31_sub(u, v), tw31[twbase + j + t * q]);
            }
        }
    }
    for (uint k = 0u; k < R; ++k) a31[j + k * q] = x[k];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_last_stage_dit_unweight_edge_61(
    __global ulong* digits61,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n, uint log_n, uint log_radix)
{
    const uint R = 1u << log_radix;
    const uint q = n >> log_radix;
    const uint j = get_global_id(0);
    if (j >= q) return;
    GF x[16];
    for (uint k = 0u; k < R; ++k) x[k] = a61[j + k * q];
    for (uint s = 0u; s < log_radix; ++s) {
        const uint stride = 1u << s;
        const uint span = stride << 1;
        const uint len = q * span;
        const uint twbase = (len >> 1) - 1u;
        for (uint b = 0u; b < R; b += span) {
            for (uint t = 0u; t < stride; ++t) {
                const uint k0 = b + t;
                const uint k1 = k0 + stride;
                GF u = x[k0];
                GF v = gf_mul(x[k1], tw61[twbase + j + t * q]);
                x[k0] = gf_add(u, v);
                x[k1] = gf_sub(u, v);
            }
        }
    }
    for (uint k = 0u; k < R; ++k) {
        const uint idx = j + k * q;
        const uint ur = (uint)(((ulong)idx * (ulong)p) & (ulong)(n - 1u));
        GF61_STORE_UNWEIGHT(digits61, idx, x[k], ur, lr2_61, log_n);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_last_stage_dit_unweight_edge_31(
    __global uint* digits31,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n, uint log_n, uint log_radix)
{
    const uint R = 1u << log_radix;
    const uint q = n >> log_radix;
    const uint j = get_global_id(0);
    if (j >= q) return;
    GF31 x[16];
    for (uint k = 0u; k < R; ++k) x[k] = a31[j + k * q];
    for (uint s = 0u; s < log_radix; ++s) {
        const uint stride = 1u << s;
        const uint span = stride << 1;
        const uint len = q * span;
        const uint twbase = (len >> 1) - 1u;
        for (uint b = 0u; b < R; b += span) {
            for (uint t = 0u; t < stride; ++t) {
                const uint k0 = b + t;
                const uint k1 = k0 + stride;
                GF31 u = x[k0];
                GF31 v = f31_mul(x[k1], tw31[twbase + j + t * q]);
                x[k0] = f31_add(u, v);
                x[k1] = f31_sub(u, v);
            }
        }
    }
    for (uint k = 0u; k < R; ++k) {
        const uint idx = j + k * q;
        const uint ur = (uint)(((ulong)idx * (ulong)p) & (ulong)(n - 1u));
        digits31[idx] = f31_unweight_digit_logn(x[k], ur, lr2_31, log_n);
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_radix4_fwd_radix8_61(
    __global const ulong* digits,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n)
{
    const uint lid = get_local_id(0);
    const uint tile = lid >> 3;
    const uint lane = lid & 7u;
    const uint len1 = n >> 2;
    const uint q1 = len1 >> 3;
    const uint j = get_group_id(0) * 8u + tile;
    if (j >= q1) return;

    const uint jw = j + lane * q1;
    const uint q0 = n >> 2;
    const uint h0 = n >> 1;
    const uint i0 = jw;
    const uint i1 = jw + q0;
    const uint i2 = jw + h0;
    const uint i3 = i2 + q0;

    const uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    const uint r1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    const uint r2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    const uint r3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));

    GF A = GF61_WEIGHT_LOAD(digits, i0, r0, lr2_61);
    GF B = GF61_WEIGHT_LOAD(digits, i2, r2, lr2_61);
    GF C = gf_add(A, B);
    GF D = gf_mul(gf_sub(A, B), tw61[((n >> 1) - 1u) + jw]);
    A = GF61_WEIGHT_LOAD(digits, i1, r1, lr2_61);
    B = GF61_WEIGHT_LOAD(digits, i3, r3, lr2_61);
    GF E = gf_add(A, B);
    GF F = gf_mul(gf_sub(A, B), tw61[((n >> 1) - 1u) + jw + q0]);
    GF W = tw61[((h0 >> 1) - 1u) + jw];

    __local GF l61[256];
    l61[0u   + tile * 8u + lane] = gf_add(C, E);
    l61[64u  + tile * 8u + lane] = gf_mul(gf_sub(C, E), W);
    l61[128u + tile * 8u + lane] = gf_add(D, F);
    l61[192u + tile * 8u + lane] = gf_mul(gf_sub(D, F), W);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane < 4u) {
        const uint quarter = lane;
        const uint off = quarter * 64u + tile * 8u;
        GF x0 = l61[off + 0u], x1 = l61[off + 1u], x2 = l61[off + 2u], x3 = l61[off + 3u];
        GF x4 = l61[off + 4u], x5 = l61[off + 5u], x6 = l61[off + 6u], x7 = l61[off + 7u];
        const uint o1 = (len1 >> 1) - 1u;
        GF t0 = gf_add(x0, x4); GF t4 = gf_mul(gf_sub(x0, x4), tw61[o1 + j]);
        GF t1 = gf_add(x1, x5); GF t5 = gf_mul(gf_sub(x1, x5), tw61[o1 + j + q1]);
        GF t2 = gf_add(x2, x6); GF t6 = gf_mul(gf_sub(x2, x6), tw61[o1 + j + (q1 << 1)]);
        GF t3 = gf_add(x3, x7); GF t7 = gf_mul(gf_sub(x3, x7), tw61[o1 + j + (q1 * 3u)]);
        const uint o2 = (len1 >> 2) - 1u;
        x0 = gf_add(t0, t2); x2 = gf_mul(gf_sub(t0, t2), tw61[o2 + j]);
        x1 = gf_add(t1, t3); x3 = gf_mul(gf_sub(t1, t3), tw61[o2 + j + q1]);
        x4 = gf_add(t4, t6); x6 = gf_mul(gf_sub(t4, t6), tw61[o2 + j]);
        x5 = gf_add(t5, t7); x7 = gf_mul(gf_sub(t5, t7), tw61[o2 + j + q1]);
        const uint o3 = q1 - 1u;
        t0 = gf_add(x0, x1); t1 = gf_mul(gf_sub(x0, x1), tw61[o3 + j]);
        t2 = gf_add(x2, x3); t3 = gf_mul(gf_sub(x2, x3), tw61[o3 + j]);
        t4 = gf_add(x4, x5); t5 = gf_mul(gf_sub(x4, x5), tw61[o3 + j]);
        t6 = gf_add(x6, x7); t7 = gf_mul(gf_sub(x6, x7), tw61[o3 + j]);
        const uint base = quarter * len1 + j;
        a61[base + 0u * q1] = t0; a61[base + 1u * q1] = t1;
        a61[base + 2u * q1] = t2; a61[base + 3u * q1] = t3;
        a61[base + 4u * q1] = t4; a61[base + 5u * q1] = t5;
        a61[base + 6u * q1] = t6; a61[base + 7u * q1] = t7;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_weight_radix4_fwd_radix8_31(
    __global const ulong* digits,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n)
{
    const uint lid = get_local_id(0);
    const uint tile = lid >> 3;
    const uint lane = lid & 7u;
    const uint len1 = n >> 2;
    const uint q1 = len1 >> 3;
    const uint j = get_group_id(0) * 8u + tile;
    if (j >= q1) return;

    const uint jw = j + lane * q1;
    const uint q0 = n >> 2;
    const uint h0 = n >> 1;
    const uint i0 = jw;
    const uint i1 = jw + q0;
    const uint i2 = jw + h0;
    const uint i3 = i2 + q0;

    GF31 A = f31_weight_digit(digits[i0], i0, p, n, lr2_31);
    GF31 B = f31_weight_digit(digits[i2], i2, p, n, lr2_31);
    GF31 C = f31_add(A, B);
    GF31 D = f31_mul(f31_sub(A, B), tw31[((n >> 1) - 1u) + jw]);
    A = f31_weight_digit(digits[i1], i1, p, n, lr2_31);
    B = f31_weight_digit(digits[i3], i3, p, n, lr2_31);
    GF31 E = f31_add(A, B);
    GF31 F = f31_mul(f31_sub(A, B), tw31[((n >> 1) - 1u) + jw + q0]);
    GF31 W = tw31[((h0 >> 1) - 1u) + jw];

    __local GF31 l31[256];
    l31[0u   + tile * 8u + lane] = f31_add(C, E);
    l31[64u  + tile * 8u + lane] = f31_mul(f31_sub(C, E), W);
    l31[128u + tile * 8u + lane] = f31_add(D, F);
    l31[192u + tile * 8u + lane] = f31_mul(f31_sub(D, F), W);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane < 4u) {
        const uint quarter = lane;
        const uint off = quarter * 64u + tile * 8u;
        GF31 x0 = l31[off + 0u], x1 = l31[off + 1u], x2 = l31[off + 2u], x3 = l31[off + 3u];
        GF31 x4 = l31[off + 4u], x5 = l31[off + 5u], x6 = l31[off + 6u], x7 = l31[off + 7u];
        const uint o1 = (len1 >> 1) - 1u;
        GF31 t0 = f31_add(x0, x4); GF31 t4 = f31_mul(f31_sub(x0, x4), tw31[o1 + j]);
        GF31 t1 = f31_add(x1, x5); GF31 t5 = f31_mul(f31_sub(x1, x5), tw31[o1 + j + q1]);
        GF31 t2 = f31_add(x2, x6); GF31 t6 = f31_mul(f31_sub(x2, x6), tw31[o1 + j + (q1 << 1)]);
        GF31 t3 = f31_add(x3, x7); GF31 t7 = f31_mul(f31_sub(x3, x7), tw31[o1 + j + (q1 * 3u)]);
        const uint o2 = (len1 >> 2) - 1u;
        x0 = f31_add(t0, t2); x2 = f31_mul(f31_sub(t0, t2), tw31[o2 + j]);
        x1 = f31_add(t1, t3); x3 = f31_mul(f31_sub(t1, t3), tw31[o2 + j + q1]);
        x4 = f31_add(t4, t6); x6 = f31_mul(f31_sub(t4, t6), tw31[o2 + j]);
        x5 = f31_add(t5, t7); x7 = f31_mul(f31_sub(t5, t7), tw31[o2 + j + q1]);
        const uint o3 = q1 - 1u;
        t0 = f31_add(x0, x1); t1 = f31_mul(f31_sub(x0, x1), tw31[o3 + j]);
        t2 = f31_add(x2, x3); t3 = f31_mul(f31_sub(x2, x3), tw31[o3 + j]);
        t4 = f31_add(x4, x5); t5 = f31_mul(f31_sub(x4, x5), tw31[o3 + j]);
        t6 = f31_add(x6, x7); t7 = f31_mul(f31_sub(x6, x7), tw31[o3 + j]);
        const uint base = quarter * len1 + j;
        a31[base + 0u * q1] = t0; a31[base + 1u * q1] = t1;
        a31[base + 2u * q1] = t2; a31[base + 3u * q1] = t3;
        a31[base + 4u * q1] = t4; a31[base + 5u * q1] = t5;
        a31[base + 6u * q1] = t6; a31[base + 7u * q1] = t7;
    }
}


static inline __attribute__((always_inline)) GF crt_inv4_output_61(__global GF* a61, __global const GF* tw61, uint n, uint quarter, uint off)
{
    const uint len = n >> 3;
    const uint q = len >> 1;
    const uint inner = off & (q - 1u);
    const uint pos = ((off & q) ? 1u : 0u) | ((off & (q << 1)) ? 2u : 0u);
    const uint baseq = quarter * (n >> 2);
    const uint i0 = baseq + inner;
    const uint i1 = i0 + q;
    const uint i2 = i0 + len;
    const uint i3 = i2 + q;
    const uint tw0 = ((len >> 1) - 1u) + inner;

    GF A = a61[i0];
    GF B = gf_mul(a61[i1], tw61[tw0]);
    GF E = a61[i2];
    GF F = gf_mul(a61[i3], tw61[tw0]);

    if ((pos & 1u) == 0u) {
        GF C = gf_add(A, B);
        GF G = gf_add(E, F);
        GF TG = gf_mul(G, tw61[(len - 1u) + inner]);
        if (pos == 0u) return gf_add(C, TG);
        return gf_sub(C, TG);
    } else {
        GF D = gf_sub(A, B);
        GF H = gf_sub(E, F);
        GF TH = gf_mul(H, tw61[(len - 1u) + inner + q]);
        if (pos == 1u) return gf_add(D, TH);
        return gf_sub(D, TH);
    }
}

static inline __attribute__((always_inline)) GF31 crt_inv4_output_31(__global GF31* a31, __global const GF31* tw31, uint n, uint quarter, uint off)
{
    const uint len = n >> 3;
    const uint q = len >> 1;
    const uint inner = off & (q - 1u);
    const uint pos = ((off & q) ? 1u : 0u) | ((off & (q << 1)) ? 2u : 0u);
    const uint baseq = quarter * (n >> 2);
    const uint i0 = baseq + inner;
    const uint i1 = i0 + q;
    const uint i2 = i0 + len;
    const uint i3 = i2 + q;
    const uint tw0 = ((len >> 1) - 1u) + inner;

    GF31 A = a31[i0];
    GF31 B = f31_mul(a31[i1], tw31[tw0]);
    GF31 E = a31[i2];
    GF31 F = f31_mul(a31[i3], tw31[tw0]);

    if ((pos & 1u) == 0u) {
        GF31 C = f31_add(A, B);
        GF31 G = f31_add(E, F);
        GF31 TG = f31_mul(G, tw31[(len - 1u) + inner]);
        if (pos == 0u) return f31_add(C, TG);
        return f31_sub(C, TG);
    } else {
        GF31 D = f31_sub(A, B);
        GF31 H = f31_sub(E, F);
        GF31 TH = f31_mul(H, tw31[(len - 1u) + inner + q]);
        if (pos == 1u) return f31_add(D, TH);
        return f31_sub(D, TH);
    }
}


static inline __attribute__((always_inline)) void crt_inv4_all_outputs_61(
    __global GF* a61, __global const GF* tw61, uint n, uint quarter, uint inner, __private GF* out)
{
    const uint len = n >> 3;
    const uint q = len >> 1;
    const uint baseq = quarter * (n >> 2);
    const uint i0 = baseq + inner;
    const uint i1 = i0 + q;
    const uint i2 = i0 + len;
    const uint i3 = i2 + q;
    const uint tw0 = ((len >> 1) - 1u) + inner;
    GF A = a61[i0];
    GF B = gf_mul(a61[i1], tw61[tw0]);
    GF C = gf_add(A, B);
    GF D = gf_sub(A, B);
    GF E = a61[i2];
    GF F = gf_mul(a61[i3], tw61[tw0]);
    GF G = gf_add(E, F);
    GF H = gf_sub(E, F);
    GF TG = gf_mul(G, tw61[(len - 1u) + inner]);
    GF TH = gf_mul(H, tw61[(len - 1u) + inner + q]);
    out[0] = gf_add(C, TG);
    out[1] = gf_add(D, TH);
    out[2] = gf_sub(C, TG);
    out[3] = gf_sub(D, TH);
}

static inline __attribute__((always_inline)) void crt_inv4_all_outputs_31(
    __global GF31* a31, __global const GF31* tw31, uint n, uint quarter, uint inner, __private GF31* out)
{
    const uint len = n >> 3;
    const uint q = len >> 1;
    const uint baseq = quarter * (n >> 2);
    const uint i0 = baseq + inner;
    const uint i1 = i0 + q;
    const uint i2 = i0 + len;
    const uint i3 = i2 + q;
    const uint tw0 = ((len >> 1) - 1u) + inner;
    GF31 A = a31[i0];
    GF31 B = f31_mul(a31[i1], tw31[tw0]);
    GF31 C = f31_add(A, B);
    GF31 D = f31_sub(A, B);
    GF31 E = a31[i2];
    GF31 F = f31_mul(a31[i3], tw31[tw0]);
    GF31 G = f31_add(E, F);
    GF31 H = f31_sub(E, F);
    GF31 TG = f31_mul(G, tw31[(len - 1u) + inner]);
    GF31 TH = f31_mul(H, tw31[(len - 1u) + inner + q]);
    out[0] = f31_add(C, TG);
    out[1] = f31_add(D, TH);
    out[2] = f31_sub(C, TG);
    out[3] = f31_sub(D, TH);
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_inv_radix4_last_unweight_block16_61(
    __global ulong* digits61,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n, uint log_n)
{
    const uint inner = get_global_id(0);
    const uint q16 = n >> 4;
    if (inner >= q16) return;
    const uint q4 = n >> 2;
    const uint mask = n - 1u;

    GF o0[4], o1[4], o2[4], o3[4];
    crt_inv4_all_outputs_61(a61, tw61, n, 0u, inner, o0);
    crt_inv4_all_outputs_61(a61, tw61, n, 1u, inner, o1);
    crt_inv4_all_outputs_61(a61, tw61, n, 2u, inner, o2);
    crt_inv4_all_outputs_61(a61, tw61, n, 3u, inner, o3);

    #pragma unroll
    for (uint pos = 0u; pos < 4u; ++pos) {
        const uint j = inner + pos * q16;
        const GF B = gf_mul(o1[pos], tw61[((n >> 2) - 1u) + j]);
        const GF C = gf_add(o0[pos], B);
        const GF D = gf_sub(o0[pos], B);
        const GF F = gf_mul(o3[pos], tw61[((n >> 2) - 1u) + j]);
        const GF G = gf_add(o2[pos], F);
        const GF H = gf_sub(o2[pos], F);
        const GF TG = gf_mul(G, tw61[((n >> 1) - 1u) + j]);
        const GF TH = gf_mul(H, tw61[((n >> 1) - 1u) + j + q4]);
        const GF y0 = gf_add(C, TG);
        const GF y2 = gf_sub(C, TG);
        const GF y1 = gf_add(D, TH);
        const GF y3 = gf_sub(D, TH);
        const uint i0 = j;
        const uint i1 = j + q4;
        const uint i2 = j + (q4 << 1);
        const uint i3 = i2 + q4;
        const uint ur0 = (uint)(((ulong)j * (ulong)p) & (ulong)mask);
        const uint ur_step = (q4 * (p & 3u)) & mask;
        const uint ur1 = (ur0 + ur_step) & mask;
        const uint ur2 = (ur1 + ur_step) & mask;
        const uint ur3 = (ur2 + ur_step) & mask;
        GF61_STORE_UNWEIGHT(digits61, i0, y0, ur0, lr2_61, log_n);
        GF61_STORE_UNWEIGHT(digits61, i1, y1, ur1, lr2_61, log_n);
        GF61_STORE_UNWEIGHT(digits61, i2, y2, ur2, lr2_61, log_n);
        GF61_STORE_UNWEIGHT(digits61, i3, y3, ur3, lr2_61, log_n);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_inv_radix4_last_unweight_block16_31(
    __global uint* digits31,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n, uint log_n)
{
    const uint inner = get_global_id(0);
    const uint q16 = n >> 4;
    if (inner >= q16) return;
    const uint q4 = n >> 2;
    const uint mask = n - 1u;

    GF31 o0[4], o1[4], o2[4], o3[4];
    crt_inv4_all_outputs_31(a31, tw31, n, 0u, inner, o0);
    crt_inv4_all_outputs_31(a31, tw31, n, 1u, inner, o1);
    crt_inv4_all_outputs_31(a31, tw31, n, 2u, inner, o2);
    crt_inv4_all_outputs_31(a31, tw31, n, 3u, inner, o3);

    #pragma unroll
    for (uint pos = 0u; pos < 4u; ++pos) {
        const uint j = inner + pos * q16;
        const GF31 B = f31_mul(o1[pos], tw31[((n >> 2) - 1u) + j]);
        const GF31 C = f31_add(o0[pos], B);
        const GF31 D = f31_sub(o0[pos], B);
        const GF31 F = f31_mul(o3[pos], tw31[((n >> 2) - 1u) + j]);
        const GF31 G = f31_add(o2[pos], F);
        const GF31 H = f31_sub(o2[pos], F);
        const GF31 TG = f31_mul(G, tw31[((n >> 1) - 1u) + j]);
        const GF31 TH = f31_mul(H, tw31[((n >> 1) - 1u) + j + q4]);
        const GF31 v0 = f31_add(C, TG);
        const GF31 v2 = f31_sub(C, TG);
        const GF31 v1 = f31_add(D, TH);
        const GF31 v3 = f31_sub(D, TH);
        const uint i0 = j;
        const uint i1 = j + q4;
        const uint i2 = j + (q4 << 1);
        const uint i3 = i2 + q4;
        const uint ur0 = (uint)(((ulong)j * (ulong)p) & (ulong)mask);
        const uint ur_step = (q4 * (p & 3u)) & mask;
        const uint ur1 = (ur0 + ur_step) & mask;
        const uint ur2 = (ur1 + ur_step) & mask;
        const uint ur3 = (ur2 + ur_step) & mask;
        digits31[i0] = f31_unweight_digit_logn(v0, ur0, lr2_31, log_n);
        digits31[i1] = f31_unweight_digit_logn(v1, ur1, lr2_31, log_n);
        digits31[i2] = f31_unweight_digit_logn(v2, ur2, lr2_31, log_n);
        digits31[i3] = f31_unweight_digit_logn(v3, ur3, lr2_31, log_n);
    }
}


static inline __attribute__((always_inline)) void crt_store_coeff_from_tail_residues(
    __global u64* restrict coeff_lo,
    __global u32* restrict coeff_hi,
    const u32 i,
    const u64 d61,
    const u32 d31)
{
    u64 lo, hi;
    crt_coeff_from_residues(d61, (u64)d31, &lo, &hi);
    coeff_lo[i] = lo;
    coeff_hi[i] = (u32)hi;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_inv_radix4_last_unweight_block16_crtcoeff(
    __global u64* restrict coeff_lo,
    __global u32* restrict coeff_hi,
    __global GF* restrict a61, __global const GF* restrict tw61,
    __global GF31* restrict a31, __global const GF31* restrict tw31,
    uint p, uint lr2_61, uint lr2_31, uint n, uint log_n)
{
    const uint inner = get_global_id(0);
    const uint q16 = n >> 4;
    if (inner >= q16) return;
    const uint q4 = n >> 2;
    const uint mask = n - 1u;

    GF o61_0[4], o61_1[4], o61_2[4], o61_3[4];
    GF31 o31_0[4], o31_1[4], o31_2[4], o31_3[4];
    crt_inv4_all_outputs_61(a61, tw61, n, 0u, inner, o61_0);
    crt_inv4_all_outputs_61(a61, tw61, n, 1u, inner, o61_1);
    crt_inv4_all_outputs_61(a61, tw61, n, 2u, inner, o61_2);
    crt_inv4_all_outputs_61(a61, tw61, n, 3u, inner, o61_3);
    crt_inv4_all_outputs_31(a31, tw31, n, 0u, inner, o31_0);
    crt_inv4_all_outputs_31(a31, tw31, n, 1u, inner, o31_1);
    crt_inv4_all_outputs_31(a31, tw31, n, 2u, inner, o31_2);
    crt_inv4_all_outputs_31(a31, tw31, n, 3u, inner, o31_3);

    #pragma unroll
    for (uint pos = 0u; pos < 4u; ++pos) {
        const uint j = inner + pos * q16;
        const uint i0 = j;
        const uint i1 = j + q4;
        const uint i2 = j + (q4 << 1);
        const uint i3 = i2 + q4;
        const uint ur0 = (uint)(((ulong)j * (ulong)p) & (ulong)mask);
        const uint ur_step = (q4 * (p & 3u)) & mask;
        const uint ur1 = (ur0 + ur_step) & mask;
        const uint ur2 = (ur1 + ur_step) & mask;
        const uint ur3 = (ur2 + ur_step) & mask;

        const GF B61 = gf_mul(o61_1[pos], tw61[((n >> 2) - 1u) + j]);
        const GF C61 = gf_add(o61_0[pos], B61);
        const GF D61 = gf_sub(o61_0[pos], B61);
        const GF F61 = gf_mul(o61_3[pos], tw61[((n >> 2) - 1u) + j]);
        const GF G61 = gf_add(o61_2[pos], F61);
        const GF H61 = gf_sub(o61_2[pos], F61);
        const GF TG61 = gf_mul(G61, tw61[((n >> 1) - 1u) + j]);
        const GF TH61 = gf_mul(H61, tw61[((n >> 1) - 1u) + j + q4]);
        const GF y61_0 = gf_add(C61, TG61);
        const GF y61_2 = gf_sub(C61, TG61);
        const GF y61_1 = gf_add(D61, TH61);
        const GF y61_3 = gf_sub(D61, TH61);

        const GF31 B31 = f31_mul(o31_1[pos], tw31[((n >> 2) - 1u) + j]);
        const GF31 C31 = f31_add(o31_0[pos], B31);
        const GF31 D31 = f31_sub(o31_0[pos], B31);
        const GF31 F31 = f31_mul(o31_3[pos], tw31[((n >> 2) - 1u) + j]);
        const GF31 G31 = f31_add(o31_2[pos], F31);
        const GF31 H31 = f31_sub(o31_2[pos], F31);
        const GF31 TG31 = f31_mul(G31, tw31[((n >> 1) - 1u) + j]);
        const GF31 TH31 = f31_mul(H31, tw31[((n >> 1) - 1u) + j + q4]);
        const GF31 y31_0 = f31_add(C31, TG31);
        const GF31 y31_2 = f31_sub(C31, TG31);
        const GF31 y31_1 = f31_add(D31, TH31);
        const GF31 y31_3 = f31_sub(D31, TH31);

        crt_store_coeff_from_tail_residues(coeff_lo, coeff_hi, i0,
            gf61_unweight_digit_from_r(y61_0, ur0, lr2_61, log_n),
            f31_unweight_digit_logn(y31_0, ur0, lr2_31, log_n));
        crt_store_coeff_from_tail_residues(coeff_lo, coeff_hi, i1,
            gf61_unweight_digit_from_r(y61_1, ur1, lr2_61, log_n),
            f31_unweight_digit_logn(y31_1, ur1, lr2_31, log_n));
        crt_store_coeff_from_tail_residues(coeff_lo, coeff_hi, i2,
            gf61_unweight_digit_from_r(y61_2, ur2, lr2_61, log_n),
            f31_unweight_digit_logn(y31_2, ur2, lr2_31, log_n));
        crt_store_coeff_from_tail_residues(coeff_lo, coeff_hi, i3,
            gf61_unweight_digit_from_r(y61_3, ur3, lr2_61, log_n),
            f31_unweight_digit_logn(y31_3, ur3, lr2_31, log_n));
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_inv_radix4_last_unweight_61(
    __global ulong* digits61,
    __global GF* a61, __global const GF* tw61,
    uint p, uint lr2_61, uint n, uint log_n)
{
    const uint j = get_global_id(0);
    const uint q = n >> 2;
    if (j >= q) return;
    const uint i0 = j;
    const uint i1 = j + q;
    const uint i2 = j + (q << 1);
    const uint i3 = i2 + q;
    GF A = crt_inv4_output_61(a61, tw61, n, 0u, j);
    GF B = gf_mul(crt_inv4_output_61(a61, tw61, n, 1u, j), tw61[((n >> 2) - 1u) + j]);
    GF C = gf_add(A, B);
    GF D = gf_sub(A, B);
    GF E = crt_inv4_output_61(a61, tw61, n, 2u, j);
    GF F = gf_mul(crt_inv4_output_61(a61, tw61, n, 3u, j), tw61[((n >> 2) - 1u) + j]);
    GF G = gf_add(E, F);
    GF H = gf_sub(E, F);
    GF TG = gf_mul(G, tw61[((n >> 1) - 1u) + j]);
    GF TH = gf_mul(H, tw61[((n >> 1) - 1u) + j + q]);
    GF y0 = gf_add(C, TG);
    GF y2 = gf_sub(C, TG);
    GF y1 = gf_add(D, TH);
    GF y3 = gf_sub(D, TH);
    const uint mask = n - 1u;
    const uint ur0 = (uint)(((ulong)j * (ulong)p) & (ulong)mask);
    const uint ur_step = (q * (p & 3u)) & mask;
    const uint ur1 = (ur0 + ur_step) & mask;
    const uint ur2 = (ur1 + ur_step) & mask;
    const uint ur3 = (ur2 + ur_step) & mask;
    GF61_STORE_UNWEIGHT(digits61, i0, y0, ur0, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i1, y1, ur1, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i2, y2, ur2, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i3, y3, ur3, lr2_61, log_n);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_inv_radix4_last_unweight_31(
    __global uint* digits31,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_31, uint n, uint log_n)
{
    const uint j = get_global_id(0);
    const uint q = n >> 2;
    if (j >= q) return;
    const uint i0 = j;
    const uint i1 = j + q;
    const uint i2 = j + (q << 1);
    const uint i3 = i2 + q;
    GF31 A = crt_inv4_output_31(a31, tw31, n, 0u, j);
    GF31 B = f31_mul(crt_inv4_output_31(a31, tw31, n, 1u, j), tw31[((n >> 2) - 1u) + j]);
    GF31 C = f31_add(A, B);
    GF31 D = f31_sub(A, B);
    GF31 E = crt_inv4_output_31(a31, tw31, n, 2u, j);
    GF31 F = f31_mul(crt_inv4_output_31(a31, tw31, n, 3u, j), tw31[((n >> 2) - 1u) + j]);
    GF31 G = f31_add(E, F);
    GF31 H = f31_sub(E, F);
    GF31 TG = f31_mul(G, tw31[((n >> 1) - 1u) + j]);
    GF31 TH = f31_mul(H, tw31[((n >> 1) - 1u) + j + q]);
    GF31 v0 = f31_add(C, TG);
    GF31 v2 = f31_sub(C, TG);
    GF31 v1 = f31_add(D, TH);
    GF31 v3 = f31_sub(D, TH);
    const uint mask = n - 1u;
    const uint ur0 = (uint)(((ulong)j * (ulong)p) & (ulong)mask);
    const uint ur_step = (q * (p & 3u)) & mask;
    const uint ur1 = (ur0 + ur_step) & mask;
    const uint ur2 = (ur1 + ur_step) & mask;
    const uint ur3 = (ur2 + ur_step) & mask;
    digits31[i0] = f31_unweight_digit_logn(v0, ur0, lr2_31, log_n);
    digits31[i1] = f31_unweight_digit_logn(v1, ur1, lr2_31, log_n);
    digits31[i2] = f31_unweight_digit_logn(v2, ur2, lr2_31, log_n);
    digits31[i3] = f31_unweight_digit_logn(v3, ur3, lr2_31, log_n);
}


static inline __attribute__((always_inline)) uint gf61_crt_valid_stage_radix(uint radix) {
    return (radix == 1024u || radix == 512u || radix == 256u || radix == 128u ||
            radix == 64u || radix == 32u || radix == 16u || radix == 8u || radix == 4u || radix == 2u);
}


static inline __attribute__((always_inline)) void crt_lds_dif2_61(__local GF* x, __global const GF* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint h = L >> 1;
    const uint total = radix >> 1;
    const uint tw0 = h * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint group = t / h;
        const uint j = t - group * h;
        const uint i0 = group * L + j;
        const uint i1 = i0 + h;
        const GF a = x[i0];
        const GF b = x[i1];
        x[i0] = gf_add(a, b);
        x[i1] = gf_mul(gf_sub(a, b), tw[tw0 + j * stride]);
    }
}
static inline __attribute__((always_inline)) void crt_lds_dif2_31(__local GF31* x, __global const GF31* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint h = L >> 1;
    const uint total = radix >> 1;
    const uint tw0 = h * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint group = t / h;
        const uint j = t - group * h;
        const uint i0 = group * L + j;
        const uint i1 = i0 + h;
        const GF31 a = x[i0];
        const GF31 b = x[i1];
        x[i0] = f31_add(a, b);
        x[i1] = f31_mul(f31_sub(a, b), tw[tw0 + j * stride]);
    }
}
static inline __attribute__((always_inline)) void crt_lds_dit2_61(__local GF* x, __global const GF* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint h = L >> 1;
    const uint total = radix >> 1;
    const uint tw0 = h * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint group = t / h;
        const uint j = t - group * h;
        const uint i0 = group * L + j;
        const uint i1 = i0 + h;
        const GF a = x[i0];
        const GF b = gf_mul(x[i1], tw[tw0 + j * stride]);
        x[i0] = gf_add(a, b);
        x[i1] = gf_sub(a, b);
    }
}
static inline __attribute__((always_inline)) void crt_lds_dit2_31(__local GF31* x, __global const GF31* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint h = L >> 1;
    const uint total = radix >> 1;
    const uint tw0 = h * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint group = t / h;
        const uint j = t - group * h;
        const uint i0 = group * L + j;
        const uint i1 = i0 + h;
        const GF31 a = x[i0];
        const GF31 b = f31_mul(x[i1], tw[tw0 + j * stride]);
        x[i0] = f31_add(a, b);
        x[i1] = f31_sub(a, b);
    }
}

static inline __attribute__((always_inline)) void crt_lds_dif4_61(__local GF* x, __global const GF* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 2;
    const uint total = radix >> 2;
    const uint o1 = (L >> 1) * stride - 1u + j0;
    const uint o2 = q * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * L;
        const uint i0 = base + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
        const GF a0 = x[i0], a1 = x[i1], a2 = x[i2], a3 = x[i3];
        const GF b0 = gf_add(a0, a2);
        const GF b2 = gf_mul(gf_sub(a0, a2), tw[o1 + j * stride]);
        const GF b1 = gf_add(a1, a3);
        const GF b3 = gf_mul(gf_sub(a1, a3), tw[o1 + (j + q) * stride]);
        x[i0] = gf_add(b0, b1);
        x[i1] = gf_mul(gf_sub(b0, b1), tw[o2 + j * stride]);
        x[i2] = gf_add(b2, b3);
        x[i3] = gf_mul(gf_sub(b2, b3), tw[o2 + j * stride]);
    }
}
static inline __attribute__((always_inline)) void crt_lds_dif4_31(__local GF31* x, __global const GF31* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 2;
    const uint total = radix >> 2;
    const uint o1 = (L >> 1) * stride - 1u + j0;
    const uint o2 = q * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * L;
        const uint i0 = base + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
        const GF31 a0 = x[i0], a1 = x[i1], a2 = x[i2], a3 = x[i3];
        const GF31 b0 = f31_add(a0, a2);
        const GF31 b2 = f31_mul(f31_sub(a0, a2), tw[o1 + j * stride]);
        const GF31 b1 = f31_add(a1, a3);
        const GF31 b3 = f31_mul(f31_sub(a1, a3), tw[o1 + (j + q) * stride]);
        x[i0] = f31_add(b0, b1);
        x[i1] = f31_mul(f31_sub(b0, b1), tw[o2 + j * stride]);
        x[i2] = f31_add(b2, b3);
        x[i3] = f31_mul(f31_sub(b2, b3), tw[o2 + j * stride]);
    }
}
static inline __attribute__((always_inline)) void crt_lds_dit4_61(__local GF* x, __global const GF* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 1;
    const uint total = radix >> 2;
    const uint o1 = q * stride - 1u + j0;
    const uint o2 = L * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * (L << 1);
        const uint i0 = base + j, i1 = i0 + q, i2 = i0 + L, i3 = i2 + q;
        const GF x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
        const GF y0 = gf_mul(x1, tw[o1 + j * stride]);
        const GF y1 = gf_mul(x3, tw[o1 + j * stride]);
        const GF a0 = gf_add(x0, y0), a1 = gf_sub(x0, y0);
        const GF a2 = gf_add(x2, y1), a3 = gf_sub(x2, y1);
        const GF z0 = gf_mul(a2, tw[o2 + j * stride]);
        const GF z1 = gf_mul(a3, tw[o2 + (j + q) * stride]);
        x[i0] = gf_add(a0, z0); x[i2] = gf_sub(a0, z0);
        x[i1] = gf_add(a1, z1); x[i3] = gf_sub(a1, z1);
    }
}
static inline __attribute__((always_inline)) void crt_lds_dit4_31(__local GF31* x, __global const GF31* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 1;
    const uint total = radix >> 2;
    const uint o1 = q * stride - 1u + j0;
    const uint o2 = L * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * (L << 1);
        const uint i0 = base + j, i1 = i0 + q, i2 = i0 + L, i3 = i2 + q;
        const GF31 x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
        const GF31 y0 = f31_mul(x1, tw[o1 + j * stride]);
        const GF31 y1 = f31_mul(x3, tw[o1 + j * stride]);
        const GF31 a0 = f31_add(x0, y0), a1 = f31_sub(x0, y0);
        const GF31 a2 = f31_add(x2, y1), a3 = f31_sub(x2, y1);
        const GF31 z0 = f31_mul(a2, tw[o2 + j * stride]);
        const GF31 z1 = f31_mul(a3, tw[o2 + (j + q) * stride]);
        x[i0] = f31_add(a0, z0); x[i2] = f31_sub(a0, z0);
        x[i1] = f31_add(a1, z1); x[i3] = f31_sub(a1, z1);
    }
}

static inline __attribute__((always_inline)) void crt_lds_dif8_61(__local GF* x, __global const GF* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 3;
    const uint total = radix >> 3;
    const uint o1 = (L >> 1) * stride - 1u + j0;
    const uint o2 = (L >> 2) * stride - 1u + j0;
    const uint o3 = q * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * L;
        const uint i0 = base + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
        const uint i4 = i3 + q, i5 = i4 + q, i6 = i5 + q, i7 = i6 + q;
        GF x0=x[i0], x1=x[i1], x2=x[i2], x3=x[i3], x4=x[i4], x5=x[i5], x6=x[i6], x7=x[i7];
        GF t0 = gf_add(x0,x4); GF t4 = gf_mul(gf_sub(x0,x4), tw[o1 + j * stride]);
        GF t1 = gf_add(x1,x5); GF t5 = gf_mul(gf_sub(x1,x5), tw[o1 + (j + q) * stride]);
        GF t2 = gf_add(x2,x6); GF t6 = gf_mul(gf_sub(x2,x6), tw[o1 + (j + (q << 1)) * stride]);
        GF t3 = gf_add(x3,x7); GF t7 = gf_mul(gf_sub(x3,x7), tw[o1 + (j + q * 3u) * stride]);
        x0 = gf_add(t0,t2); x2 = gf_mul(gf_sub(t0,t2), tw[o2 + j * stride]);
        x1 = gf_add(t1,t3); x3 = gf_mul(gf_sub(t1,t3), tw[o2 + (j + q) * stride]);
        x4 = gf_add(t4,t6); x6 = gf_mul(gf_sub(t4,t6), tw[o2 + j * stride]);
        x5 = gf_add(t5,t7); x7 = gf_mul(gf_sub(t5,t7), tw[o2 + (j + q) * stride]);
        t0 = gf_add(x0,x1); t1 = gf_mul(gf_sub(x0,x1), tw[o3 + j * stride]);
        t2 = gf_add(x2,x3); t3 = gf_mul(gf_sub(x2,x3), tw[o3 + j * stride]);
        t4 = gf_add(x4,x5); t5 = gf_mul(gf_sub(x4,x5), tw[o3 + j * stride]);
        t6 = gf_add(x6,x7); t7 = gf_mul(gf_sub(x6,x7), tw[o3 + j * stride]);
        x[i0]=t0; x[i1]=t1; x[i2]=t2; x[i3]=t3; x[i4]=t4; x[i5]=t5; x[i6]=t6; x[i7]=t7;
    }
}
static inline __attribute__((always_inline)) void crt_lds_dif8_31(__local GF31* x, __global const GF31* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 3;
    const uint total = radix >> 3;
    const uint o1 = (L >> 1) * stride - 1u + j0;
    const uint o2 = (L >> 2) * stride - 1u + j0;
    const uint o3 = q * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * L;
        const uint i0 = base + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
        const uint i4 = i3 + q, i5 = i4 + q, i6 = i5 + q, i7 = i6 + q;
        GF31 x0=x[i0], x1=x[i1], x2=x[i2], x3=x[i3], x4=x[i4], x5=x[i5], x6=x[i6], x7=x[i7];
        GF31 t0 = f31_add(x0,x4); GF31 t4 = f31_mul(f31_sub(x0,x4), tw[o1 + j * stride]);
        GF31 t1 = f31_add(x1,x5); GF31 t5 = f31_mul(f31_sub(x1,x5), tw[o1 + (j + q) * stride]);
        GF31 t2 = f31_add(x2,x6); GF31 t6 = f31_mul(f31_sub(x2,x6), tw[o1 + (j + (q << 1)) * stride]);
        GF31 t3 = f31_add(x3,x7); GF31 t7 = f31_mul(f31_sub(x3,x7), tw[o1 + (j + q * 3u) * stride]);
        x0 = f31_add(t0,t2); x2 = f31_mul(f31_sub(t0,t2), tw[o2 + j * stride]);
        x1 = f31_add(t1,t3); x3 = f31_mul(f31_sub(t1,t3), tw[o2 + (j + q) * stride]);
        x4 = f31_add(t4,t6); x6 = f31_mul(f31_sub(t4,t6), tw[o2 + j * stride]);
        x5 = f31_add(t5,t7); x7 = f31_mul(f31_sub(t5,t7), tw[o2 + (j + q) * stride]);
        t0 = f31_add(x0,x1); t1 = f31_mul(f31_sub(x0,x1), tw[o3 + j * stride]);
        t2 = f31_add(x2,x3); t3 = f31_mul(f31_sub(x2,x3), tw[o3 + j * stride]);
        t4 = f31_add(x4,x5); t5 = f31_mul(f31_sub(x4,x5), tw[o3 + j * stride]);
        t6 = f31_add(x6,x7); t7 = f31_mul(f31_sub(x6,x7), tw[o3 + j * stride]);
        x[i0]=t0; x[i1]=t1; x[i2]=t2; x[i3]=t3; x[i4]=t4; x[i5]=t5; x[i6]=t6; x[i7]=t7;
    }
}
static inline __attribute__((always_inline)) void crt_lds_dit8_61(__local GF* x, __global const GF* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 1;
    const uint total = radix >> 3;
    const uint o1 = q * stride - 1u + j0;
    const uint o2 = L * stride - 1u + j0;
    const uint o3 = (L << 1) * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * (L << 2);
        const uint i0 = base + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
        const uint i4 = i3 + q, i5 = i4 + q, i6 = i5 + q, i7 = i6 + q;
        GF x0=x[i0], x1=x[i1], x2=x[i2], x3=x[i3], x4=x[i4], x5=x[i5], x6=x[i6], x7=x[i7];
        GF y0=gf_mul(x1, tw[o1 + j * stride]); GF y1=gf_mul(x3, tw[o1 + j * stride]);
        GF y2=gf_mul(x5, tw[o1 + j * stride]); GF y3=gf_mul(x7, tw[o1 + j * stride]);
        x1=gf_sub(x0,y0); x0=gf_add(x0,y0); x3=gf_sub(x2,y1); x2=gf_add(x2,y1);
        x5=gf_sub(x4,y2); x4=gf_add(x4,y2); x7=gf_sub(x6,y3); x6=gf_add(x6,y3);
        y0=gf_mul(x2, tw[o2 + j * stride]); y1=gf_mul(x3, tw[o2 + (j + q) * stride]);
        GF y4=gf_mul(x6, tw[o2 + j * stride]); GF y5=gf_mul(x7, tw[o2 + (j + q) * stride]);
        x2=gf_sub(x0,y0); x0=gf_add(x0,y0); x3=gf_sub(x1,y1); x1=gf_add(x1,y1);
        x6=gf_sub(x4,y4); x4=gf_add(x4,y4); x7=gf_sub(x5,y5); x5=gf_add(x5,y5);
        y0=gf_mul(x4, tw[o3 + j * stride]); y1=gf_mul(x5, tw[o3 + (j + q) * stride]);
        y2=gf_mul(x6, tw[o3 + (j + (q << 1)) * stride]); y3=gf_mul(x7, tw[o3 + (j + q * 3u) * stride]);
        x4=gf_sub(x0,y0); x0=gf_add(x0,y0); x5=gf_sub(x1,y1); x1=gf_add(x1,y1);
        x6=gf_sub(x2,y2); x2=gf_add(x2,y2); x7=gf_sub(x3,y3); x3=gf_add(x3,y3);
        x[i0]=x0; x[i1]=x1; x[i2]=x2; x[i3]=x3; x[i4]=x4; x[i5]=x5; x[i6]=x6; x[i7]=x7;
    }
}
static inline __attribute__((always_inline)) void crt_lds_dit8_31(__local GF31* x, __global const GF31* tw, uint radix, uint L, uint stride, uint j0, uint lane) {
    const uint q = L >> 1;
    const uint total = radix >> 3;
    const uint o1 = q * stride - 1u + j0;
    const uint o2 = L * stride - 1u + j0;
    const uint o3 = (L << 1) * stride - 1u + j0;
    for (uint t = lane; t < total; t += 64u) {
        const uint block = t / q;
        const uint j = t - block * q;
        const uint base = block * (L << 2);
        const uint i0 = base + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
        const uint i4 = i3 + q, i5 = i4 + q, i6 = i5 + q, i7 = i6 + q;
        GF31 x0=x[i0], x1=x[i1], x2=x[i2], x3=x[i3], x4=x[i4], x5=x[i5], x6=x[i6], x7=x[i7];
        GF31 y0=f31_mul(x1, tw[o1 + j * stride]); GF31 y1=f31_mul(x3, tw[o1 + j * stride]);
        GF31 y2=f31_mul(x5, tw[o1 + j * stride]); GF31 y3=f31_mul(x7, tw[o1 + j * stride]);
        x1=f31_sub(x0,y0); x0=f31_add(x0,y0); x3=f31_sub(x2,y1); x2=f31_add(x2,y1);
        x5=f31_sub(x4,y2); x4=f31_add(x4,y2); x7=f31_sub(x6,y3); x6=f31_add(x6,y3);
        y0=f31_mul(x2, tw[o2 + j * stride]); y1=f31_mul(x3, tw[o2 + (j + q) * stride]);
        GF31 y4=f31_mul(x6, tw[o2 + j * stride]); GF31 y5=f31_mul(x7, tw[o2 + (j + q) * stride]);
        x2=f31_sub(x0,y0); x0=f31_add(x0,y0); x3=f31_sub(x1,y1); x1=f31_add(x1,y1);
        x6=f31_sub(x4,y4); x4=f31_add(x4,y4); x7=f31_sub(x5,y5); x5=f31_add(x5,y5);
        y0=f31_mul(x4, tw[o3 + j * stride]); y1=f31_mul(x5, tw[o3 + (j + q) * stride]);
        y2=f31_mul(x6, tw[o3 + (j + (q << 1)) * stride]); y3=f31_mul(x7, tw[o3 + (j + q * 3u) * stride]);
        x4=f31_sub(x0,y0); x0=f31_add(x0,y0); x5=f31_sub(x1,y1); x1=f31_add(x1,y1);
        x6=f31_sub(x2,y2); x2=f31_add(x2,y2); x7=f31_sub(x3,y3); x3=f31_add(x3,y3);
        x[i0]=x0; x[i1]=x1; x[i2]=x2; x[i3]=x3; x[i4]=x4; x[i5]=x5; x[i6]=x6; x[i7]=x7;
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_lds_stage_dif_pow2(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u) return;

    const uint stride = len / radix;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[1024];
    __local GF31 l31[1024];

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) l61[t] = a61[base + t * stride];
    } else {
        for (uint t = lane; t < radix; t += 64u) l31[t] = a31[base + t * stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        if (is61) crt_lds_dif8_61(l61, tw61, radix, L, stride, j0, lane);
        else      crt_lds_dif8_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L >>= 3;
    }
    if (L == 4u) {
        if (is61) crt_lds_dif4_61(l61, tw61, radix, 4u, stride, j0, lane);
        else      crt_lds_dif4_31(l31, tw31, radix, 4u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        if (is61) crt_lds_dif2_61(l61, tw61, radix, 2u, stride, j0, lane);
        else      crt_lds_dif2_31(l31, tw31, radix, 2u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) a61[base + t * stride] = l61[t];
    } else {
        for (uint t = lane; t < radix; t += 64u) a31[base + t * stride] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_lds_stage_dit_pow2(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;

    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[1024];
    __local GF31 l31[1024];

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) l61[t] = a61[base + t * stride];
    } else {
        for (uint t = lane; t < radix; t += 64u) l31[t] = a31[base + t * stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        if (is61) crt_lds_dit2_61(l61, tw61, radix, L, stride, j0, lane);
        else      crt_lds_dit2_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        if (is61) crt_lds_dit4_61(l61, tw61, radix, L, stride, j0, lane);
        else      crt_lds_dit4_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        if (is61) crt_lds_dit8_61(l61, tw61, radix, L, stride, j0, lane);
        else      crt_lds_dit8_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) a61[base + t * stride] = l61[t];
    } else {
        for (uint t = lane; t < radix; t += 64u) a31[base + t * stride] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_lds_stage_dif_pow2_tile2(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u || radix > 512u) return;

    const uint stride = len / radix;
    if (stride == 0u) return;
    const uint pair_cols = (stride + 1u) >> 1u;
    const uint gid = get_group_id(0);
    const uint block = gid / pair_cols;
    const uint pair = gid - block * pair_cols;
    const uint jbase = pair << 1;
    const uint cols = ((jbase + 1u) < stride) ? 2u : 1u;
    const uint base0 = block * len + jbase;

    __local GF l61[1024];
    __local GF31 l31[1024];

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) {
            l61[t] = a61[base0 + t * stride];
            if (cols > 1u) l61[512u + t] = a61[base0 + 1u + t * stride];
        }
    } else {
        for (uint t = lane; t < radix; t += 64u) {
            l31[t] = a31[base0 + t * stride];
            if (cols > 1u) l31[512u + t] = a31[base0 + 1u + t * stride];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        if (is61) {
            crt_lds_dif8_61(l61, tw61, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dif8_61(l61 + 512u, tw61, radix, L, stride, jbase + 1u, lane);
        } else {
            crt_lds_dif8_31(l31, tw31, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dif8_31(l31 + 512u, tw31, radix, L, stride, jbase + 1u, lane);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        L >>= 3;
    }
    if (L == 4u) {
        if (is61) {
            crt_lds_dif4_61(l61, tw61, radix, 4u, stride, jbase, lane);
            if (cols > 1u) crt_lds_dif4_61(l61 + 512u, tw61, radix, 4u, stride, jbase + 1u, lane);
        } else {
            crt_lds_dif4_31(l31, tw31, radix, 4u, stride, jbase, lane);
            if (cols > 1u) crt_lds_dif4_31(l31 + 512u, tw31, radix, 4u, stride, jbase + 1u, lane);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        if (is61) {
            crt_lds_dif2_61(l61, tw61, radix, 2u, stride, jbase, lane);
            if (cols > 1u) crt_lds_dif2_61(l61 + 512u, tw61, radix, 2u, stride, jbase + 1u, lane);
        } else {
            crt_lds_dif2_31(l31, tw31, radix, 2u, stride, jbase, lane);
            if (cols > 1u) crt_lds_dif2_31(l31 + 512u, tw31, radix, 2u, stride, jbase + 1u, lane);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) {
            a61[base0 + t * stride] = l61[t];
            if (cols > 1u) a61[base0 + 1u + t * stride] = l61[512u + t];
        }
    } else {
        for (uint t = lane; t < radix; t += 64u) {
            a31[base0 + t * stride] = l31[t];
            if (cols > 1u) a31[base0 + 1u + t * stride] = l31[512u + t];
        }
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_lds_stage_dit_pow2_tile2(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u || radix > 512u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;
    const uint pair_cols = (stride + 1u) >> 1u;
    const uint gid = get_group_id(0);
    const uint block = gid / pair_cols;
    const uint pair = gid - block * pair_cols;
    const uint jbase = pair << 1;
    const uint cols = ((jbase + 1u) < stride) ? 2u : 1u;
    const uint base0 = block * len + jbase;

    __local GF l61[1024];
    __local GF31 l31[1024];

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) {
            l61[t] = a61[base0 + t * stride];
            if (cols > 1u) l61[512u + t] = a61[base0 + 1u + t * stride];
        }
    } else {
        for (uint t = lane; t < radix; t += 64u) {
            l31[t] = a31[base0 + t * stride];
            if (cols > 1u) l31[512u + t] = a31[base0 + 1u + t * stride];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        if (is61) {
            crt_lds_dit2_61(l61, tw61, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dit2_61(l61 + 512u, tw61, radix, L, stride, jbase + 1u, lane);
        } else {
            crt_lds_dit2_31(l31, tw31, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dit2_31(l31 + 512u, tw31, radix, L, stride, jbase + 1u, lane);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        if (is61) {
            crt_lds_dit4_61(l61, tw61, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dit4_61(l61 + 512u, tw61, radix, L, stride, jbase + 1u, lane);
        } else {
            crt_lds_dit4_31(l31, tw31, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dit4_31(l31 + 512u, tw31, radix, L, stride, jbase + 1u, lane);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        if (is61) {
            crt_lds_dit8_61(l61, tw61, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dit8_61(l61 + 512u, tw61, radix, L, stride, jbase + 1u, lane);
        } else {
            crt_lds_dit8_31(l31, tw31, radix, L, stride, jbase, lane);
            if (cols > 1u) crt_lds_dit8_31(l31 + 512u, tw31, radix, L, stride, jbase + 1u, lane);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    if (is61) {
        for (uint t = lane; t < radix; t += 64u) {
            a61[base0 + t * stride] = l61[t];
            if (cols > 1u) a61[base0 + 1u + t * stride] = l61[512u + t];
        }
    } else {
        for (uint t = lane; t < radix; t += 64u) {
            a31[base0 + t * stride] = l31[t];
            if (cols > 1u) a31[base0 + 1u + t * stride] = l31[512u + t];
        }
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_61(
    __global GF* a61, __global const GF* tw61,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u || radix > 1024u) return;

    const uint stride = len / radix;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[1024];
    for (uint t = lane; t < radix; t += 64u) l61[t] = a61[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        crt_lds_dif8_61(l61, tw61, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L >>= 3;
    }
    if (L == 4u) {
        crt_lds_dif4_61(l61, tw61, radix, 4u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        crt_lds_dif2_61(l61, tw61, radix, 2u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lane; t < radix; t += 64u) a61[base + t * stride] = l61[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_61(
    __global GF* a61, __global const GF* tw61,
    uint n, uint base_len, uint radix)
{
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u || radix > 1024u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;

    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[1024];
    for (uint t = lane; t < radix; t += 64u) l61[t] = a61[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        crt_lds_dit2_61(l61, tw61, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        crt_lds_dit4_61(l61, tw61, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        crt_lds_dit8_61(l61, tw61, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    for (uint t = lane; t < radix; t += 64u) a61[base + t * stride] = l61[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_31(
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u || radix > 1024u) return;

    const uint stride = len / radix;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF31 l31[1024];
    for (uint t = lane; t < radix; t += 64u) l31[t] = a31[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        crt_lds_dif8_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L >>= 3;
    }
    if (L == 4u) {
        crt_lds_dif4_31(l31, tw31, radix, 4u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        crt_lds_dif2_31(l31, tw31, radix, 2u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lane; t < radix; t += 64u) a31[base + t * stride] = l31[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_31(
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u || radix > 1024u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;

    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF31 l31[1024];
    for (uint t = lane; t < radix; t += 64u) l31[t] = a31[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        crt_lds_dit2_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        crt_lds_dit4_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        crt_lds_dit8_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    for (uint t = lane; t < radix; t += 64u) a31[base + t * stride] = l31[t];
}


static inline __attribute__((always_inline)) void crt_lds_dif8_61_leaf512(__local GF* l, uint lane) {
    const uint i0 = lane * 8u;
    const uint i1 = i0 + 1u, i2 = i0 + 2u, i3 = i0 + 3u;
    const uint i4 = i0 + 4u, i5 = i0 + 5u, i6 = i0 + 6u, i7 = i0 + 7u;
    GF x0 = l[i0], x1 = l[i1], x2 = l[i2], x3 = l[i3];
    GF x4 = l[i4], x5 = l[i5], x6 = l[i6], x7 = l[i7];
    GF t0 = gf_add(x0, x4); GF t4 = gf_sub(x0, x4);
    GF t1 = gf_add(x1, x5); GF t5 = gf_mul_w8_fast(gf_sub(x1, x5));
    GF t2 = gf_add(x2, x6); GF t6 = gf_mul_minus_i_fast(gf_sub(x2, x6));
    GF t3 = gf_add(x3, x7); GF t7 = gf_mul_w8_3_fast(gf_sub(x3, x7));
    x0 = gf_add(t0, t2); x2 = gf_sub(t0, t2);
    x1 = gf_add(t1, t3); x3 = gf_mul_minus_i_fast(gf_sub(t1, t3));
    x4 = gf_add(t4, t6); x6 = gf_sub(t4, t6);
    x5 = gf_add(t5, t7); x7 = gf_mul_minus_i_fast(gf_sub(t5, t7));
    t0 = gf_add(x0, x1); t1 = gf_sub(x0, x1);
    t2 = gf_add(x2, x3); t3 = gf_sub(x2, x3);
    t4 = gf_add(x4, x5); t5 = gf_sub(x4, x5);
    t6 = gf_add(x6, x7); t7 = gf_sub(x6, x7);
    l[i0] = t0; l[i1] = t1; l[i2] = t2; l[i3] = t3;
    l[i4] = t4; l[i5] = t5; l[i6] = t6; l[i7] = t7;
}

static inline __attribute__((always_inline)) void crt_lds_dit8_61_leaf512(__local GF* l, uint lane) {
    const uint i0 = lane * 8u;
    const uint i1 = i0 + 1u, i2 = i0 + 2u, i3 = i0 + 3u;
    const uint i4 = i0 + 4u, i5 = i0 + 5u, i6 = i0 + 6u, i7 = i0 + 7u;
    GF x0 = l[i0], x1 = l[i1], x2 = l[i2], x3 = l[i3];
    GF x4 = l[i4], x5 = l[i5], x6 = l[i6], x7 = l[i7];
    GF y0 = x1;
    GF y1 = x3;
    GF y2 = x5;
    GF y3 = x7;
    x1 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x2, y1); x2 = gf_add(x2, y1);
    x5 = gf_sub(x4, y2); x4 = gf_add(x4, y2);
    x7 = gf_sub(x6, y3); x6 = gf_add(x6, y3);
    y0 = x2;
    y1 = gf_mul_minus_i_fast(x3);
    GF y4 = x6;
    GF y5 = gf_mul_minus_i_fast(x7);
    x2 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x3 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x4, y4); x4 = gf_add(x4, y4);
    x7 = gf_sub(x5, y5); x5 = gf_add(x5, y5);
    y0 = x4;
    y1 = gf_mul_w8_inv_fast(x5);
    y2 = gf_mul_minus_i_fast(x6);
    y3 = gf_mul_w8_inv3_fast(x7);
    x4 = gf_sub(x0, y0); x0 = gf_add(x0, y0);
    x5 = gf_sub(x1, y1); x1 = gf_add(x1, y1);
    x6 = gf_sub(x2, y2); x2 = gf_add(x2, y2);
    x7 = gf_sub(x3, y3); x3 = gf_add(x3, y3);
    l[i0] = x0; l[i1] = x1; l[i2] = x2; l[i3] = x3;
    l[i4] = x4; l[i5] = x5; l[i6] = x6; l[i7] = x7;
}

static inline __attribute__((always_inline)) void crt_lds_dif8_31_leaf512(__local GF31* l, uint lane) {
    const uint i0 = lane * 8u;
    const uint i1 = i0 + 1u, i2 = i0 + 2u, i3 = i0 + 3u;
    const uint i4 = i0 + 4u, i5 = i0 + 5u, i6 = i0 + 6u, i7 = i0 + 7u;
    GF31 x0 = l[i0], x1 = l[i1], x2 = l[i2], x3 = l[i3];
    GF31 x4 = l[i4], x5 = l[i5], x6 = l[i6], x7 = l[i7];
    GF31 t0 = f31_add(x0, x4); GF31 t4 = f31_sub(x0, x4);
    GF31 t1 = f31_add(x1, x5); GF31 t5 = f31_mul_w8_fast(f31_sub(x1, x5));
    GF31 t2 = f31_add(x2, x6); GF31 t6 = f31_mul_minus_i_fast(f31_sub(x2, x6));
    GF31 t3 = f31_add(x3, x7); GF31 t7 = f31_mul_w8_3_fast(f31_sub(x3, x7));
    x0 = f31_add(t0, t2); x2 = f31_sub(t0, t2);
    x1 = f31_add(t1, t3); x3 = f31_mul_minus_i_fast(f31_sub(t1, t3));
    x4 = f31_add(t4, t6); x6 = f31_sub(t4, t6);
    x5 = f31_add(t5, t7); x7 = f31_mul_minus_i_fast(f31_sub(t5, t7));
    t0 = f31_add(x0, x1); t1 = f31_sub(x0, x1);
    t2 = f31_add(x2, x3); t3 = f31_sub(x2, x3);
    t4 = f31_add(x4, x5); t5 = f31_sub(x4, x5);
    t6 = f31_add(x6, x7); t7 = f31_sub(x6, x7);
    l[i0] = t0; l[i1] = t1; l[i2] = t2; l[i3] = t3;
    l[i4] = t4; l[i5] = t5; l[i6] = t6; l[i7] = t7;
}

static inline __attribute__((always_inline)) void crt_lds_dit8_31_leaf512(__local GF31* l, uint lane) {
    const uint i0 = lane * 8u;
    const uint i1 = i0 + 1u, i2 = i0 + 2u, i3 = i0 + 3u;
    const uint i4 = i0 + 4u, i5 = i0 + 5u, i6 = i0 + 6u, i7 = i0 + 7u;
    GF31 x0 = l[i0], x1 = l[i1], x2 = l[i2], x3 = l[i3];
    GF31 x4 = l[i4], x5 = l[i5], x6 = l[i6], x7 = l[i7];
    GF31 y0 = x1;
    GF31 y1 = x3;
    GF31 y2 = x5;
    GF31 y3 = x7;
    x1 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
    x3 = f31_sub(x2, y1); x2 = f31_add(x2, y1);
    x5 = f31_sub(x4, y2); x4 = f31_add(x4, y2);
    x7 = f31_sub(x6, y3); x6 = f31_add(x6, y3);
    y0 = x2;
    y1 = f31_mul_i_fast(x3);
    GF31 y4 = x6;
    GF31 y5 = f31_mul_i_fast(x7);
    x2 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
    x3 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
    x6 = f31_sub(x4, y4); x4 = f31_add(x4, y4);
    x7 = f31_sub(x5, y5); x5 = f31_add(x5, y5);
    y0 = x4;
    y1 = f31_mul_w8_inv_fast(x5);
    y2 = f31_mul_i_fast(x6);
    y3 = f31_mul_w8_inv3_fast(x7);
    x4 = f31_sub(x0, y0); x0 = f31_add(x0, y0);
    x5 = f31_sub(x1, y1); x1 = f31_add(x1, y1);
    x6 = f31_sub(x2, y2); x2 = f31_add(x2, y2);
    x7 = f31_sub(x3, y3); x3 = f31_add(x3, y3);
    l[i0] = x0; l[i1] = x1; l[i2] = x2; l[i3] = x3;
    l[i4] = x4; l[i5] = x5; l[i6] = x6; l[i7] = x7;
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_61_512opt(
    __global GF* a61, __global const GF* tw61,
    uint n, uint len, uint radix)
{
    (void)n;
    (void)radix;
    const uint lane = get_local_id(0);
    const uint stride = len >> 9;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[512];
    for (uint t = lane; t < 512u; t += 64u) l61[t] = a61[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dif8_61(l61, tw61, 512u, 512u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61(l61, tw61, 512u,  64u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61_leaf512(l61, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) a61[base + t * stride] = l61[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_61_512opt(
    __global GF* a61, __global const GF* tw61,
    uint n, uint base_len, uint radix)
{
    (void)radix;
    const uint lane = get_local_id(0);
    (void)n;
    const uint stride = base_len;
    const uint len = base_len << 9;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[512];
    for (uint t = lane; t < 512u; t += 64u) l61[t] = a61[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dit8_61_leaf512(l61, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u,  16u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u, 128u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) a61[base + t * stride] = l61[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_31_512opt(
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    (void)radix;
    const uint lane = get_local_id(0);
    const uint stride = len >> 9;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF31 l31[512];
    for (uint t = lane; t < 512u; t += 64u) l31[t] = a31[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dif8_31(l31, tw31, 512u, 512u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_31(l31, tw31, 512u,  64u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_31_leaf512(l31, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) a31[base + t * stride] = l31[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_31_512opt(
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    (void)radix;
    const uint lane = get_local_id(0);
    (void)n;
    const uint stride = base_len;
    const uint len = base_len << 9;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF31 l31[512];
    for (uint t = lane; t < 512u; t += 64u) l31[t] = a31[base + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dit8_31_leaf512(l31, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_31(l31, tw31, 512u,  16u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_31(l31, tw31, 512u, 128u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) a31[base + t * stride] = l31[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_61x31_512opt(
    __global GF* a61, __global GF31* a31,
    __global const GF* tw61, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    (void)radix;
    const uint lane = get_local_id(0);
    const uint stride = len >> 9;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[512];
    __local GF31 l31[512];
    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = base + t * stride;
        l61[t] = a61[idx];
        l31[t] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dif8_61(l61, tw61, 512u, 512u, stride, j0, lane);
    crt_lds_dif8_31(l31, tw31, 512u, 512u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61(l61, tw61, 512u, 64u, stride, j0, lane);
    crt_lds_dif8_31(l31, tw31, 512u, 64u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61_leaf512(l61, lane);
    crt_lds_dif8_31_leaf512(l31, lane);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = base + t * stride;
        a61[idx] = l61[t];
        a31[idx] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_61x31_512opt(
    __global GF* a61, __global GF31* a31,
    __global const GF* tw61, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    (void)n;
    (void)radix;
    const uint lane = get_local_id(0);
    const uint stride = base_len;
    const uint len = base_len << 9;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[512];
    __local GF31 l31[512];
    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = base + t * stride;
        l61[t] = a61[idx];
        l31[t] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dit8_61_leaf512(l61, lane);
    crt_lds_dit8_31_leaf512(l31, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u, 16u, stride, j0, lane);
    crt_lds_dit8_31(l31, tw31, 512u, 16u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u, 128u, stride, j0, lane);
    crt_lds_dit8_31(l31, tw31, 512u, 128u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = base + t * stride;
        a61[idx] = l61[t];
        a31[idx] = l31[t];
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_61x31_any(
    __global GF* a61, __global GF31* a31,
    __global const GF* tw61, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u || radix > 1024u) return;

    const uint stride = len / radix;
    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[1024];
    __local GF31 l31[1024];
    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = base + t * stride;
        l61[t] = a61[idx];
        l31[t] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        crt_lds_dif8_61(l61, tw61, radix, L, stride, j0, lane);
        crt_lds_dif8_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) { L = 1u; break; }
        L >>= 3;
    }
    if (L == 4u) {
        crt_lds_dif4_61(l61, tw61, radix, 4u, stride, j0, lane);
        crt_lds_dif4_31(l31, tw31, radix, 4u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        crt_lds_dif2_61(l61, tw61, radix, 2u, stride, j0, lane);
        crt_lds_dif2_31(l31, tw31, radix, 2u, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = base + t * stride;
        a61[idx] = l61[t];
        a31[idx] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_61x31_any(
    __global GF* a61, __global GF31* a31,
    __global const GF* tw61, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u || radix > 1024u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;

    const uint gid = get_group_id(0);
    const uint block = gid / stride;
    const uint j0 = gid - block * stride;
    const uint base = block * len + j0;

    __local GF l61[1024];
    __local GF31 l31[1024];
    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = base + t * stride;
        l61[t] = a61[idx];
        l31[t] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        crt_lds_dit2_61(l61, tw61, radix, L, stride, j0, lane);
        crt_lds_dit2_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        crt_lds_dit4_61(l61, tw61, radix, L, stride, j0, lane);
        crt_lds_dit4_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        crt_lds_dit8_61(l61, tw61, radix, L, stride, j0, lane);
        crt_lds_dit8_31(l31, tw31, radix, L, stride, j0, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = base + t * stride;
        a61[idx] = l61[t];
        a31[idx] = l31[t];
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_61_tile4(
    __global GF* a61, __global const GF* tw61,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u || radix > 512u) return;

    const uint stride = len / radix;
    if (stride == 0u) return;
    const uint tile_cols = (stride + 3u) >> 2u;
    const uint gid = get_group_id(0);
    const uint block = gid / tile_cols;
    const uint tile = gid - block * tile_cols;
    const uint jbase = tile << 2;
    const uint remain = stride - jbase;
    const uint cols = remain >= 4u ? 4u : remain;
    const uint base0 = block * len + jbase;

    __local GF l61[2048];
    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        l61[t] = a61[base0 + off];
        if (cols > 1u) l61[ 512u + t] = a61[base0 + 1u + off];
        if (cols > 2u) l61[1024u + t] = a61[base0 + 2u + off];
        if (cols > 3u) l61[1536u + t] = a61[base0 + 3u + off];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        crt_lds_dif8_61(l61, tw61, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dif8_61(l61 +  512u, tw61, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dif8_61(l61 + 1024u, tw61, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dif8_61(l61 + 1536u, tw61, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L >>= 3;
    }
    if (L == 4u) {
        crt_lds_dif4_61(l61, tw61, radix, 4u, stride, jbase, lane);
        if (cols > 1u) crt_lds_dif4_61(l61 +  512u, tw61, radix, 4u, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dif4_61(l61 + 1024u, tw61, radix, 4u, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dif4_61(l61 + 1536u, tw61, radix, 4u, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        crt_lds_dif2_61(l61, tw61, radix, 2u, stride, jbase, lane);
        if (cols > 1u) crt_lds_dif2_61(l61 +  512u, tw61, radix, 2u, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dif2_61(l61 + 1024u, tw61, radix, 2u, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dif2_61(l61 + 1536u, tw61, radix, 2u, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        a61[base0 + off] = l61[t];
        if (cols > 1u) a61[base0 + 1u + off] = l61[ 512u + t];
        if (cols > 2u) a61[base0 + 2u + off] = l61[1024u + t];
        if (cols > 3u) a61[base0 + 3u + off] = l61[1536u + t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_61_tile4(
    __global GF* a61, __global const GF* tw61,
    uint n, uint base_len, uint radix)
{
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u || radix > 512u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;
    const uint tile_cols = (stride + 3u) >> 2u;
    const uint gid = get_group_id(0);
    const uint block = gid / tile_cols;
    const uint tile = gid - block * tile_cols;
    const uint jbase = tile << 2;
    const uint remain = stride - jbase;
    const uint cols = remain >= 4u ? 4u : remain;
    const uint base0 = block * len + jbase;

    __local GF l61[2048];
    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        l61[t] = a61[base0 + off];
        if (cols > 1u) l61[ 512u + t] = a61[base0 + 1u + off];
        if (cols > 2u) l61[1024u + t] = a61[base0 + 2u + off];
        if (cols > 3u) l61[1536u + t] = a61[base0 + 3u + off];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        crt_lds_dit2_61(l61, tw61, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dit2_61(l61 +  512u, tw61, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dit2_61(l61 + 1024u, tw61, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dit2_61(l61 + 1536u, tw61, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        crt_lds_dit4_61(l61, tw61, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dit4_61(l61 +  512u, tw61, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dit4_61(l61 + 1024u, tw61, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dit4_61(l61 + 1536u, tw61, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        crt_lds_dit8_61(l61, tw61, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dit8_61(l61 +  512u, tw61, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dit8_61(l61 + 1024u, tw61, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dit8_61(l61 + 1536u, tw61, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        a61[base0 + off] = l61[t];
        if (cols > 1u) a61[base0 + 1u + off] = l61[ 512u + t];
        if (cols > 2u) a61[base0 + 2u + off] = l61[1024u + t];
        if (cols > 3u) a61[base0 + 3u + off] = l61[1536u + t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dif_pow2_31_tile4(
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint len, uint radix)
{
    (void)n;
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || len < radix || (len & (radix - 1u)) != 0u || radix > 512u) return;

    const uint stride = len / radix;
    if (stride == 0u) return;
    const uint tile_cols = (stride + 3u) >> 2u;
    const uint gid = get_group_id(0);
    const uint block = gid / tile_cols;
    const uint tile = gid - block * tile_cols;
    const uint jbase = tile << 2;
    const uint remain = stride - jbase;
    const uint cols = remain >= 4u ? 4u : remain;
    const uint base0 = block * len + jbase;

    __local GF31 l31[2048];
    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        l31[t] = a31[base0 + off];
        if (cols > 1u) l31[ 512u + t] = a31[base0 + 1u + off];
        if (cols > 2u) l31[1024u + t] = a31[base0 + 2u + off];
        if (cols > 3u) l31[1536u + t] = a31[base0 + 3u + off];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) {
        crt_lds_dif8_31(l31, tw31, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dif8_31(l31 +  512u, tw31, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dif8_31(l31 + 1024u, tw31, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dif8_31(l31 + 1536u, tw31, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L >>= 3;
    }
    if (L == 4u) {
        crt_lds_dif4_31(l31, tw31, radix, 4u, stride, jbase, lane);
        if (cols > 1u) crt_lds_dif4_31(l31 +  512u, tw31, radix, 4u, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dif4_31(l31 + 1024u, tw31, radix, 4u, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dif4_31(l31 + 1536u, tw31, radix, 4u, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        crt_lds_dif2_31(l31, tw31, radix, 2u, stride, jbase, lane);
        if (cols > 1u) crt_lds_dif2_31(l31 +  512u, tw31, radix, 2u, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dif2_31(l31 + 1024u, tw31, radix, 2u, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dif2_31(l31 + 1536u, tw31, radix, 2u, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        a31[base0 + off] = l31[t];
        if (cols > 1u) a31[base0 + 1u + off] = l31[ 512u + t];
        if (cols > 2u) a31[base0 + 2u + off] = l31[1024u + t];
        if (cols > 3u) a31[base0 + 3u + off] = l31[1536u + t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_lds_stage_dit_pow2_31_tile4(
    __global GF31* a31, __global const GF31* tw31,
    uint n, uint base_len, uint radix)
{
    const uint lane = get_local_id(0);
    if (!gf61_crt_valid_stage_radix(radix) || base_len == 0u || radix > 512u) return;

    const uint stride = base_len;
    const uint len = base_len * radix;
    if (len > n) return;
    const uint tile_cols = (stride + 3u) >> 2u;
    const uint gid = get_group_id(0);
    const uint block = gid / tile_cols;
    const uint tile = gid - block * tile_cols;
    const uint jbase = tile << 2;
    const uint remain = stride - jbase;
    const uint cols = remain >= 4u ? 4u : remain;
    const uint base0 = block * len + jbase;

    __local GF31 l31[2048];
    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        l31[t] = a31[base0 + off];
        if (cols > 1u) l31[ 512u + t] = a31[base0 + 1u + off];
        if (cols > 2u) l31[1024u + t] = a31[base0 + 2u + off];
        if (cols > 3u) l31[1536u + t] = a31[base0 + 3u + off];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        crt_lds_dit2_31(l31, tw31, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dit2_31(l31 +  512u, tw31, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dit2_31(l31 + 1024u, tw31, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dit2_31(l31 + 1536u, tw31, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        crt_lds_dit4_31(l31, tw31, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dit4_31(l31 +  512u, tw31, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dit4_31(l31 + 1024u, tw31, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dit4_31(l31 + 1536u, tw31, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < radix) {
        crt_lds_dit8_31(l31, tw31, radix, L, stride, jbase, lane);
        if (cols > 1u) crt_lds_dit8_31(l31 +  512u, tw31, radix, L, stride, jbase + 1u, lane);
        if (cols > 2u) crt_lds_dit8_31(l31 + 1024u, tw31, radix, L, stride, jbase + 2u, lane);
        if (cols > 3u) crt_lds_dit8_31(l31 + 1536u, tw31, radix, L, stride, jbase + 3u, lane);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }

    for (uint t = lane; t < radix; t += 64u) {
        const uint off = t * stride;
        a31[base0 + off] = l31[t];
        if (cols > 1u) a31[base0 + 1u + off] = l31[ 512u + t];
        if (cols > 2u) a31[base0 + 2u + off] = l31[1024u + t];
        if (cols > 3u) a31[base0 + 3u + off] = l31[1536u + t];
    }
}



#define DEFINE_MIXED_STAGE_1LDS_61(R) \
__kernel __attribute__((reqd_work_group_size(64,1,1))) \
void gf61_crt_lds_stage_dif_pow2_61_1lds_##R( \
    __global GF* a61, __global const GF* tw61, uint n, uint len, uint radix) \
{ \
    const uint lane = get_local_id(0); \
    (void)radix; \
    if (len < (uint)(R) || (len & ((uint)(R) - 1u)) != 0u) return; \
    const uint stride = len / (uint)(R); \
    const uint j = get_group_id(0) % stride; \
    const uint block = get_group_id(0) / stride; \
    const uint base = block * len + j; \
    if (base >= n) return; \
    __local GF l61[(R)]; \
    for (uint t = lane; t < (uint)(R); t += 64u) l61[t] = a61[base + t * stride]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint L = (uint)(R); \
    while (L >= 8u) { \
        crt_lds_dif8_61(l61, tw61, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L >>= 3; \
    } \
    if (L == 4u) { \
        crt_lds_dif4_61(l61, tw61, (uint)(R), 4u, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } else if (L == 2u) { \
        crt_lds_dif2_61(l61, tw61, (uint)(R), 2u, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (uint t = lane; t < (uint)(R); t += 64u) a61[base + t * stride] = l61[t]; \
} \
__kernel __attribute__((reqd_work_group_size(64,1,1))) \
void gf61_crt_lds_stage_dit_pow2_61_1lds_##R( \
    __global GF* a61, __global const GF* tw61, uint n, uint base_len, uint radix) \
{ \
    const uint lane = get_local_id(0); \
    (void)radix; \
    if (base_len == 0u) return; \
    const uint stride = base_len; \
    const uint len = base_len * (uint)(R); \
    if (len > n) return; \
    const uint j = get_group_id(0) % stride; \
    const uint block = get_group_id(0) / stride; \
    const uint base = block * len + j; \
    __local GF l61[(R)]; \
    for (uint t = lane; t < (uint)(R); t += 64u) l61[t] = a61[base + t * stride]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint tail = (uint)(R); \
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3; \
    uint L = 2u; \
    if (tail == 2u) { \
        crt_lds_dit2_61(l61, tw61, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L <<= 1; \
    } else if (tail == 4u) { \
        crt_lds_dit4_61(l61, tw61, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L <<= 2; \
    } \
    while (L < (uint)(R)) { \
        crt_lds_dit8_61(l61, tw61, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L <<= 3; \
    } \
    for (uint t = lane; t < (uint)(R); t += 64u) a61[base + t * stride] = l61[t]; \
}

#define DEFINE_MIXED_STAGE_1LDS_31(R) \
__kernel __attribute__((reqd_work_group_size(64,1,1))) \
void gf61_crt_lds_stage_dif_pow2_31_1lds_##R( \
    __global GF31* a31, __global const GF31* tw31, uint n, uint len, uint radix) \
{ \
    const uint lane = get_local_id(0); \
    (void)radix; \
    if (len < (uint)(R) || (len & ((uint)(R) - 1u)) != 0u) return; \
    const uint stride = len / (uint)(R); \
    const uint j = get_group_id(0) % stride; \
    const uint block = get_group_id(0) / stride; \
    const uint base = block * len + j; \
    if (base >= n) return; \
    __local GF31 l31[(R)]; \
    for (uint t = lane; t < (uint)(R); t += 64u) l31[t] = a31[base + t * stride]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint L = (uint)(R); \
    while (L >= 8u) { \
        crt_lds_dif8_31(l31, tw31, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L >>= 3; \
    } \
    if (L == 4u) { \
        crt_lds_dif4_31(l31, tw31, (uint)(R), 4u, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } else if (L == 2u) { \
        crt_lds_dif2_31(l31, tw31, (uint)(R), 2u, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (uint t = lane; t < (uint)(R); t += 64u) a31[base + t * stride] = l31[t]; \
} \
__kernel __attribute__((reqd_work_group_size(64,1,1))) \
void gf61_crt_lds_stage_dit_pow2_31_1lds_##R( \
    __global GF31* a31, __global const GF31* tw31, uint n, uint base_len, uint radix) \
{ \
    const uint lane = get_local_id(0); \
    (void)radix; \
    if (base_len == 0u) return; \
    const uint stride = base_len; \
    const uint len = base_len * (uint)(R); \
    if (len > n) return; \
    const uint j = get_group_id(0) % stride; \
    const uint block = get_group_id(0) / stride; \
    const uint base = block * len + j; \
    __local GF31 l31[(R)]; \
    for (uint t = lane; t < (uint)(R); t += 64u) l31[t] = a31[base + t * stride]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint tail = (uint)(R); \
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3; \
    uint L = 2u; \
    if (tail == 2u) { \
        crt_lds_dit2_31(l31, tw31, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L <<= 1; \
    } else if (tail == 4u) { \
        crt_lds_dit4_31(l31, tw31, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L <<= 2; \
    } \
    while (L < (uint)(R)) { \
        crt_lds_dit8_31(l31, tw31, (uint)(R), L, stride, j, lane); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        L <<= 3; \
    } \
    for (uint t = lane; t < (uint)(R); t += 64u) a31[base + t * stride] = l31[t]; \
}

DEFINE_MIXED_STAGE_1LDS_61(8)
DEFINE_MIXED_STAGE_1LDS_61(16)
DEFINE_MIXED_STAGE_1LDS_61(32)
DEFINE_MIXED_STAGE_1LDS_61(64)
DEFINE_MIXED_STAGE_1LDS_61(128)
DEFINE_MIXED_STAGE_1LDS_61(256)
DEFINE_MIXED_STAGE_1LDS_61(512)
DEFINE_MIXED_STAGE_1LDS_61(1024)
DEFINE_MIXED_STAGE_1LDS_31(8)
DEFINE_MIXED_STAGE_1LDS_31(16)
DEFINE_MIXED_STAGE_1LDS_31(32)
DEFINE_MIXED_STAGE_1LDS_31(64)
DEFINE_MIXED_STAGE_1LDS_31(128)
DEFINE_MIXED_STAGE_1LDS_31(256)
DEFINE_MIXED_STAGE_1LDS_31(512)
DEFINE_MIXED_STAGE_1LDS_31(1024)
#undef DEFINE_MIXED_STAGE_1LDS_61
#undef DEFINE_MIXED_STAGE_1LDS_31

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_forward_lds512_to_center(
    __global GF* a61, __global const GF* twf61,
    __global GF31* a31, __global const GF31* twf31,
    uint n, uint target)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    const uint base = get_group_id(0) * 512u;
    __local GF l61[512];
    __local GF31 l31[512];

    if (is61) {
        for (uint t = lane; t < 512u; t += 64u) l61[t] = a61[base + t];
    } else {
        for (uint t = lane; t < 512u; t += 64u) l31[t] = a31[base + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint f_len = 512u;
    while (f_len > target) {
        if (f_len >= 8u && (f_len >> 3) >= target) {
            if (is61) local_stage_dif_radix8_pow2(l61, twf61, 512u, f_len, lane, 64u);
            else      crt_local_stage_dif_radix8_31(l31, twf31, 512u, f_len, lane, 64u);
            f_len >>= 3;
        } else if (f_len >= 4u && (f_len >> 2) >= target) {
            if (is61) local_stage_dif_radix4_pow2(l61, twf61, 512u, f_len, lane, 64u);
            else      crt_local_stage_dif_radix4_31(l31, twf31, 512u, f_len, lane, 64u);
            f_len >>= 2;
        } else {
            if (is61) local_stage_dif_pow2(l61, twf61, 512u, f_len, lane, 64u);
            else      crt_local_stage_dif_pow2_31(l31, twf31, 512u, f_len, lane, 64u);
            f_len >>= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (is61) {
        for (uint t = lane; t < 512u; t += 64u) a61[base + t] = l61[t];
    } else {
        for (uint t = lane; t < 512u; t += 64u) a31[base + t] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_inverse_lds512_from_center(
    __global GF* a61, __global const GF* twi61,
    __global GF31* a31, __global const GF31* twi31,
    uint n, uint target)
{
    (void)n;
    const uint lid = get_local_id(0);
    const uint lane = lid & 63u;
    const uint is61 = (lid < 64u);
    const uint base = get_group_id(0) * 512u;
    __local GF l61[512];
    __local GF31 l31[512];

    if (is61) {
        for (uint t = lane; t < 512u; t += 64u) l61[t] = a61[base + t];
    } else {
        for (uint t = lane; t < 512u; t += 64u) l31[t] = a31[base + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint i_len = target << 1;
    while (i_len <= 512u) {
        if ((i_len << 2) <= 512u) {
            if (is61) local_stage_dit_radix8_pow2(l61, twi61, 512u, i_len, lane, 64u);
            else      crt_local_stage_dit_radix8_31(l31, twi31, 512u, i_len, lane, 64u);
            i_len <<= 3;
        } else if ((i_len << 1) <= 512u) {
            if (is61) local_stage_dit_radix4_pow2(l61, twi61, 512u, i_len, lane, 64u);
            else      crt_local_stage_dit_radix4_31(l31, twi31, 512u, i_len, lane, 64u);
            i_len <<= 2;
        } else {
            if (is61) local_stage_dit_pow2(l61, twi61, 512u, i_len, lane, 64u);
            else      crt_local_stage_dit_pow2_31(l31, twi31, 512u, i_len, lane, 64u);
            i_len <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (is61) {
        for (uint t = lane; t < 512u; t += 64u) a61[base + t] = l61[t];
    } else {
        for (uint t = lane; t < 512u; t += 64u) a31[base + t] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_inverse_bridge_256_to_512(
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint n)
{
    __local GF l61[512];
    __local GF31 l31[512];
    uint lid = get_local_id(0);
    uint block = get_group_id(0);
    uint base = block * 512u;
    for (uint t = lid; t < 512u; t += 128u) { l61[t] = a61[base + t]; l31[t] = a31[base + t]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint t = lid; t < 256u; t += 128u) {
        GF A = l61[t]; GF B = gf_mul(l61[t + 256u], tw61[255u + t]);
        l61[t] = gf_add(A, B);
        l61[t + 256u] = gf_sub(A, B);
        GF31 C = l31[t]; GF31 D = f31_mul(l31[t + 256u], tw31[255u + t]);
        l31[t] = f31_add(C, D);
        l31[t + 256u] = f31_sub(C, D);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint t = lid; t < 512u; t += 128u) { a61[base + t] = l61[t]; a31[base + t] = l31[t]; }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_last_stage_dit_radix4_unweight_wg64(
    __global ulong* digits61, __global uint* digits31,
    __global GF* a61, __global const GF* tw61,
    __global GF31* a31, __global const GF31* tw31,
    uint p, uint lr2_61, uint lr2_31, uint n, uint len, uint log_n)
{
    uint gid = get_global_id(0);
    uint q = len >> 1;
    uint groups = n / (len << 1);
    uint total = groups * q;
    if (gid >= total) return;
    uint j = gid & (q - 1u);
    uint base = (gid - j) << 2;
    uint i0 = base + j;
    uint i1 = i0 + q;
    uint i2 = i0 + len;
    uint i3 = i2 + q;
    GF A = a61[i0]; GF B = gf_mul(a61[i1], tw61[((len >> 1) - 1u) + j]);
    GF C = gf_add(A, B); GF D = gf_sub(A, B);
    GF E = a61[i2]; GF F = gf_mul(a61[i3], tw61[((len >> 1) - 1u) + j]);
    GF G = gf_add(E, F); GF H = gf_sub(E, F);
    GF TG = gf_mul(G, tw61[(len - 1u) + j]);
    GF TH = gf_mul(H, tw61[(len - 1u) + j + q]);
    GF y0 = gf_add(C, TG);
    GF y2 = gf_sub(C, TG);
    GF y1 = gf_add(D, TH);
    GF y3 = gf_sub(D, TH);
    uint ur61_0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    uint ur61_1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    uint ur61_2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    uint ur61_3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));
    GF61_STORE_UNWEIGHT(digits61, i0, y0, ur61_0, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i1, y1, ur61_1, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i2, y2, ur61_2, lr2_61, log_n);
    GF61_STORE_UNWEIGHT(digits61, i3, y3, ur61_3, lr2_61, log_n);

    GF31 A31 = a31[i0]; GF31 B31 = f31_mul(a31[i1], tw31[((len >> 1) - 1u) + j]);
    GF31 C31 = f31_add(A31, B31); GF31 D31 = f31_sub(A31, B31);
    GF31 E31 = a31[i2]; GF31 F31 = f31_mul(a31[i3], tw31[((len >> 1) - 1u) + j]);
    GF31 G31 = f31_add(E31, F31); GF31 H31 = f31_sub(E31, F31);
    GF31 TG31 = f31_mul(G31, tw31[(len - 1u) + j]);
    GF31 TH31 = f31_mul(H31, tw31[(len - 1u) + j + q]);
    GF31 v0 = f31_add(C31, TG31);
    GF31 v2 = f31_sub(C31, TG31);
    GF31 v1 = f31_add(D31, TH31);
    GF31 v3 = f31_sub(D31, TH31);
    uint ur31_0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)(n - 1u));
    uint ur31_1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)(n - 1u));
    uint ur31_2 = (uint)(((ulong)i2 * (ulong)p) & (ulong)(n - 1u));
    uint ur31_3 = (uint)(((ulong)i3 * (ulong)p) & (ulong)(n - 1u));
    digits31[i0] = f31_unweight_digit_logn(v0, ur31_0, lr2_31, log_n);
    digits31[i1] = f31_unweight_digit_logn(v1, ur31_1, lr2_31, log_n);
    digits31[i2] = f31_unweight_digit_logn(v2, ur31_2, lr2_31, log_n);
    digits31[i3] = f31_unweight_digit_logn(v3, ur31_3, lr2_31, log_n);
}


inline void crt_priv_dit_stage_61(__private GF* x, __global const GF* tw,
                                  uint twbase, uint stride, uint q, uint j) {
    for (uint b = 0u; b < 16u; b += (stride << 1)) {
        for (uint s = 0u; s < stride; ++s) {
            uint i0 = b + s;
            uint i1 = i0 + stride;
            GF a = x[i0];
            GF t = gf_mul(x[i1], tw[twbase + j + s * q]);
            x[i0] = gf_add(a, t);
            x[i1] = gf_sub(a, t);
        }
    }
}

inline void crt_priv_dit_stage_31(__private GF31* x, __global const GF31* tw,
                                  uint twbase, uint stride, uint q, uint j) {
    for (uint b = 0u; b < 16u; b += (stride << 1)) {
        for (uint s = 0u; s < stride; ++s) {
            uint i0 = b + s;
            uint i1 = i0 + stride;
            GF31 a = x[i0];
            GF31 t = f31_mul(x[i1], tw[twbase + j + s * q]);
            x[i0] = f31_add(a, t);
            x[i1] = f31_sub(a, t);
        }
    }
}

__kernel void gf61_crt_forward_bridge_1024_to_256(__global GF* a61,
                                                   __global GF31* a31,
                                                   __global const GF* tw61,
                                                   __global const GF31* tw31,
                                                   const uint n) {
    const uint lid = get_local_id(0);
    const uint grp = get_group_id(0);
    const uint base = grp * 1024u;
    __local GF l61[1024];
    __local GF31 l31[1024];

    for (uint t = lid; t < 1024u; t += get_local_size(0)) {
        l61[t] = a61[base + t];
        l31[t] = a31[base + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint t = lid; t < 512u; t += get_local_size(0)) {
        uint i0 = t;
        uint i1 = t + 512u;
        GF u = l61[i0];
        GF v = l61[i1];
        l61[i0] = gf_add(u, v);
        l61[i1] = gf_mul(gf_sub(u, v), tw61[511u + t]);
        GF31 u31 = l31[i0];
        GF31 v31 = l31[i1];
        l31[i0] = f31_add(u31, v31);
        l31[i1] = f31_mul(f31_sub(u31, v31), tw31[511u + t]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint t = lid; t < 512u; t += get_local_size(0)) {
        uint g = t >> 8;
        uint j = t & 255u;
        uint i0 = g * 512u + j;
        uint i1 = i0 + 256u;
        GF u = l61[i0];
        GF v = l61[i1];
        l61[i0] = gf_add(u, v);
        l61[i1] = gf_mul(gf_sub(u, v), tw61[255u + j]);
        GF31 u31 = l31[i0];
        GF31 v31 = l31[i1];
        l31[i0] = f31_add(u31, v31);
        l31[i1] = f31_mul(f31_sub(u31, v31), tw31[255u + j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 1024u; t += get_local_size(0)) {
        a61[base + t] = l61[t];
        a31[base + t] = l31[t];
    }
}

__kernel void gf61_crt_inverse_bridge_256_to_1024(__global GF* a61,
                                                   __global GF31* a31,
                                                   __global const GF* tw61,
                                                   __global const GF31* tw31,
                                                   const uint n) {
    const uint lid = get_local_id(0);
    const uint grp = get_group_id(0);
    const uint base = grp * 1024u;
    __local GF l61[1024];
    __local GF31 l31[1024];

    for (uint t = lid; t < 1024u; t += get_local_size(0)) {
        l61[t] = a61[base + t];
        l31[t] = a31[base + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint t = lid; t < 512u; t += get_local_size(0)) {
        uint g = t >> 8;
        uint j = t & 255u;
        uint i0 = g * 512u + j;
        uint i1 = i0 + 256u;
        GF a = l61[i0];
        GF b = gf_mul(l61[i1], tw61[255u + j]);
        l61[i0] = gf_add(a, b);
        l61[i1] = gf_sub(a, b);
        GF31 a31v = l31[i0];
        GF31 b31v = f31_mul(l31[i1], tw31[255u + j]);
        l31[i0] = f31_add(a31v, b31v);
        l31[i1] = f31_sub(a31v, b31v);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint t = lid; t < 512u; t += get_local_size(0)) {
        uint i0 = t;
        uint i1 = t + 512u;
        GF a = l61[i0];
        GF b = gf_mul(l61[i1], tw61[511u + t]);
        l61[i0] = gf_add(a, b);
        l61[i1] = gf_sub(a, b);
        GF31 a31v = l31[i0];
        GF31 b31v = f31_mul(l31[i1], tw31[511u + t]);
        l31[i0] = f31_add(a31v, b31v);
        l31[i1] = f31_sub(a31v, b31v);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 1024u; t += get_local_size(0)) {
        a61[base + t] = l61[t];
        a31[base + t] = l31[t];
    }
}

__kernel void gf61_crt_last_stage_dit_radix16_unweight(__global const GF* a61,
                                                        __global const GF31* a31,
                                                        __global ulong* digits61,
                                                        __global uint* digits31,
                                                        __global const GF* tw61,
                                                        __global const GF31* tw31,
                                                        const uint n,
                                                        const uint p,
                                                        const uint log_n,
                                                        const uint lr2_61,
                                                        const uint lr2_31) {
    const uint len = n >> 3;
    const uint q = len >> 1;
    const uint gid = get_global_id(0);
    const uint j = gid & (q - 1u);
    const uint base = (gid - j) << 4;

    GF x61[16];
    GF31 x31[16];
    for (uint m = 0u; m < 16u; ++m) {
        uint idx = base + j + m * q;
        x61[m] = a61[idx];
        x31[m] = a31[idx];
    }

    crt_priv_dit_stage_61(x61, tw61, q - 1u, 1u, q, j);
    crt_priv_dit_stage_61(x61, tw61, len - 1u, 2u, q, j);
    crt_priv_dit_stage_61(x61, tw61, (len << 1) - 1u, 4u, q, j);
    crt_priv_dit_stage_61(x61, tw61, (len << 2) - 1u, 8u, q, j);
    crt_priv_dit_stage_31(x31, tw31, q - 1u, 1u, q, j);
    crt_priv_dit_stage_31(x31, tw31, len - 1u, 2u, q, j);
    crt_priv_dit_stage_31(x31, tw31, (len << 1) - 1u, 4u, q, j);
    crt_priv_dit_stage_31(x31, tw31, (len << 2) - 1u, 8u, q, j);

    for (uint m = 0u; m < 16u; ++m) {
        uint idx = base + j + m * q;
        uint r = (uint)(((ulong)idx * (ulong)p) & (ulong)(n - 1u));
        GF61_STORE_UNWEIGHT(digits61, idx, x61[m], r, lr2_61, log_n);
        digits31[idx] = f31_unweight_digit_logn(x31[m], r, lr2_31, log_n);
    }
}


inline GF gf_norm_pair(GF z) {
    return (GF)(norm61(z.s0), norm61(z.s1));
}

inline GF gf_half_scalar_pair(GF z) {
    return (GF)(rshift61(norm61(z.s0), 1u), rshift61(norm61(z.s1), 1u));
}

inline GF gf_quarter_scalar_pair(GF z) {
    return (GF)(rshift61(norm61(z.s0), 2u), rshift61(norm61(z.s1), 2u));
}

inline GF gf_conj_fast(GF z) {
    return (GF)(z.s0, sub61(0ul, z.s1));
}

inline GF gf_neg_conj_fast(GF z) {
    return (GF)(sub61(0ul, z.s0), z.s1);
}

inline GF gf_pack_e_plus_i_o(GF e, GF o) {
    return (GF)(sub61(e.s0, o.s1), add61(e.s1, o.s0));
}

inline GF gf_pack_e_minus_i_o(GF e, GF o) {
    return (GF)(add61(e.s0, o.s1), sub61(e.s1, o.s0));
}

inline GF gf_pack_conj_e_plus_i_conj_o(GF e, GF o) {
    
    return (GF)(add61(e.s0, o.s1), sub61(o.s0, e.s1));
}

inline GF gf_pack_conj_e_minus_i_conj_o(GF e, GF o) {
    
    return (GF)(sub61(e.s0, o.s1), sub61(0ul, add61(e.s1, o.s0)));
}

static inline __attribute__((always_inline)) GF31 f31_half_pair(GF31 z) {
    return (GF31)(f31_lshift30_scalar(z.s0), f31_lshift30_scalar(z.s1));
}

static inline __attribute__((always_inline)) GF31 f31_quarter_pair(GF31 z) {
    return (GF31)(f31_lshift29_scalar(z.s0), f31_lshift29_scalar(z.s1));
}

inline GF31 f31_conj_fast(GF31 z) {
    return (GF31)(z.s0, f31_sub_scalar(0u, z.s1));
}

inline GF31 f31_neg_conj_fast(GF31 z) {
    return (GF31)(f31_sub_scalar(0u, z.s0), z.s1);
}

inline GF31 f31_pack_e_plus_i_o(GF31 e, GF31 o) {
    return (GF31)(f31_sub_scalar(e.s0, o.s1), f31_add_scalar(e.s1, o.s0));
}

inline GF31 f31_pack_e_minus_i_o(GF31 e, GF31 o) {
    return (GF31)(f31_add_scalar(e.s0, o.s1), f31_sub_scalar(e.s1, o.s0));
}

inline GF31 f31_pack_conj_e_plus_i_conj_o(GF31 e, GF31 o) {
    return (GF31)(f31_add_scalar(e.s0, o.s1), f31_sub_scalar(o.s0, e.s1));
}

inline GF31 f31_pack_conj_e_minus_i_conj_o(GF31 e, GF31 o) {
    return (GF31)(f31_sub_scalar(e.s0, o.s1), f31_sub_scalar(0u, f31_add_scalar(e.s1, o.s0)));
}

inline uint f31_weight_scalar_idx(ulong digit, uint index, uint p, uint n, uint lr2) {
    const uint r = (uint)(((ulong)index * (ulong)p) & (ulong)(n - 1u));
    return f31_lshift_scalar(f31_reduce_ulong(digit), shift_from_r31(r, lr2));
}

inline uint f31_weight_scalar_r(ulong digit, uint r, uint lr2) {
    return f31_lshift_scalar(f31_reduce_ulong(digit), shift_from_r31(r, lr2));
}

inline uint f31_unweight_scalar_idx(uint v, uint index, uint p, uint n, uint lr2, uint log_m) {
    const uint r = (uint)(((ulong)index * (ulong)p) & (ulong)(n - 1u));
    const uint s = f31_mod31_small(shift_from_r31(r, lr2) + log_m);
    return f31_lshift_scalar(v, s == 0u ? 0u : 31u - s);
}


static inline __attribute__((always_inline))
uint crt_bitrev_u32(uint x, uint logn);
static inline __attribute__((always_inline))
uint crt_bitrev3_u32_fast(uint x);
static inline __attribute__((always_inline))
uint crt_bitrev4_u32_fast(uint x);
static inline __attribute__((always_inline))
uint crt_bitrev9_u32_fast(uint x);
static inline __attribute__((always_inline))
uint crt_bitrev10_u32_fast(uint x);
static inline __attribute__((always_inline))
uint crt_halfreal_perm_u32(uint x, uint logn, uint flags);
static inline __attribute__((always_inline))
void crt_halfreal_center_eo_f48_61(GF z, GF zneg_conj, __global const GF* restrict twf, const uint n, const uint k, __private GF* Eout, __private GF* Oout);
static inline __attribute__((always_inline))
void crt_halfreal_center_pair_f48_w_61(GF z, GF zneg_conj, GF W, __private GF* out0, __private GF* out1);
static inline __attribute__((always_inline))
GF crt_halfreal_center_one_61(GF z, GF zneg_conj, __global const GF* restrict twf, __global const GF* restrict twi, const uint n, const uint k, const uint flags);
static inline __attribute__((always_inline))
void crt_halfreal_center_eo_f48_31(GF31 z, GF31 zneg_conj, __global const GF31* restrict twf, const uint n, const uint k, __private GF31* Eout, __private GF31* Oout);
static inline __attribute__((always_inline))
void crt_halfreal_center_pair_f48_w_31(GF31 z, GF31 zneg_conj, GF31 W, __private GF31* out0, __private GF31* out1);
static inline __attribute__((always_inline))
GF31 crt_halfreal_center_one_31(GF31 z, GF31 zneg_conj, __global const GF31* restrict twf, __global const GF31* restrict twi, const uint n, const uint k, const uint flags);


static inline __attribute__((always_inline)) uint crt_mixed_inv_pow2_mod_odd(uint pow2_n, uint odd) {
    
    
    const uint r = pow2_n % odd;
    if (odd == 3u) return (r == 2u) ? 2u : 1u;
    if (odd == 9u) {
        
        if (r == 1u) return 1u;
        if (r == 2u) return 5u;
        if (r == 4u) return 7u;
        if (r == 5u) return 2u;
        if (r == 7u) return 4u;
        if (r == 8u) return 8u;
    }
    
    for (uint x = 1u; x < odd; ++x) if (((r * x) % odd) == 1u) return x;
    return 1u;
}

static inline __attribute__((always_inline)) uint crt_mixed_j_from_coord(uint a, uint b, uint odd, uint pow2_n) {
    const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
    const uint bmod = b % odd;
    const uint delta = (a + odd - bmod) % odd;
    const uint t = (delta * inv_m) % odd;
    return b + pow2_n * t;
}
static inline __attribute__((always_inline)) uint crt_mixed_r_from_coord_fast_pre(uint a, uint b, uint odd, uint pow2_n, uint p2, uint podd, uint inv_m) {
    
    
    const uint mask = pow2_n - 1u;
    const uint r2 = (uint)(((ulong)(b & mask) * (ulong)p2) & (ulong)mask);
    const uint ro = (a * podd) % odd;
    const uint r2o = r2 % odd;
    const uint delta = (ro + odd - r2o) % odd;
    const uint t = (delta * inv_m) % odd;
    return r2 + pow2_n * t;
}


static inline GF gf_mul_real_odd(GF z, GF w) {
    const ulong c = w.s0;
    return (GF)(mul61(z.s0, c), mul61(z.s1, c));
}

static inline GF31 f31_mul_real_odd(GF31 z, GF31 w) {
    const uint c = w.s0;
    return (GF31)(f31_mul_scalar(z.s0, c), f31_mul_scalar(z.s1, c));
}


static inline __attribute__((always_inline)) GF gf_mul_real_c61(GF z, ulong c) {
    return (GF)(mul61(z.s0, c), mul61(z.s1, c));
}

static inline __attribute__((always_inline)) GF31 f31_mul_real_c31(GF31 z, uint c) {
    return (GF31)(f31_mul_scalar(z.s0, c), f31_mul_scalar(z.s1, c));
}

static inline __attribute__((always_inline))
void gf_dft3_noscale_61(GF x0, GF x1, GF x2, ulong root3,
                        __private GF* y0, __private GF* y1, __private GF* y2)
{
    const ulong root3_2 = mul61(root3, root3);
    const GF x1w1 = gf_mul_real_c61(x1, root3);
    const GF x2w2 = gf_mul_real_c61(x2, root3_2);
    const GF x1w2 = gf_mul_real_c61(x1, root3_2);
    const GF x2w1 = gf_mul_real_c61(x2, root3);
    *y0 = gf_add(gf_add(x0, x1), x2);
    *y1 = gf_add(gf_add(x0, x1w1), x2w2);
    *y2 = gf_add(gf_add(x0, x1w2), x2w1);
}

static inline __attribute__((always_inline))
void gf_dft3_scaled_61(GF x0, GF x1, GF x2, ulong root3, ulong scale,
                       __private GF* y0, __private GF* y1, __private GF* y2)
{
    GF t0, t1, t2;
    gf_dft3_noscale_61(x0, x1, x2, root3, &t0, &t1, &t2);
    if (scale != 1ul) {
        t0 = gf_mul_real_c61(t0, scale);
        t1 = gf_mul_real_c61(t1, scale);
        t2 = gf_mul_real_c61(t2, scale);
    }
    *y0 = t0; *y1 = t1; *y2 = t2;
}

static inline __attribute__((always_inline))
void gf_dft9_scaled_61(__private GF* x, __private GF* y, ulong root9, ulong scale)
{
    const ulong root3 = mul61(mul61(root9, root9), root9);
    const ulong w1 = root9;
    const ulong w2 = mul61(root9, root9);
    const ulong w4 = mul61(w2, w2);

    GF a00, a01, a02, a10, a11, a12, a20, a21, a22;
    gf_dft3_noscale_61(x[0], x[3], x[6], root3, &a00, &a01, &a02);
    gf_dft3_noscale_61(x[1], x[4], x[7], root3, &a10, &a11, &a12);
    gf_dft3_noscale_61(x[2], x[5], x[8], root3, &a20, &a21, &a22);

    a11 = gf_mul_real_c61(a11, w1);
    a21 = gf_mul_real_c61(a21, w2);
    a12 = gf_mul_real_c61(a12, w2);
    a22 = gf_mul_real_c61(a22, w4);

    gf_dft3_noscale_61(a00, a10, a20, root3, &y[0], &y[3], &y[6]);
    gf_dft3_noscale_61(a01, a11, a21, root3, &y[1], &y[4], &y[7]);
    gf_dft3_noscale_61(a02, a12, a22, root3, &y[2], &y[5], &y[8]);

    if (scale != 1ul) {
        for (uint i = 0u; i < 9u; ++i) y[i] = gf_mul_real_c61(y[i], scale);
    }
}

static inline __attribute__((always_inline))
void f31_dft3_noscale(GF31 x0, GF31 x1, GF31 x2, uint root3,
                      __private GF31* y0, __private GF31* y1, __private GF31* y2)
{
    const uint root3_2 = f31_mul_scalar(root3, root3);
    const GF31 x1w1 = f31_mul_real_c31(x1, root3);
    const GF31 x2w2 = f31_mul_real_c31(x2, root3_2);
    const GF31 x1w2 = f31_mul_real_c31(x1, root3_2);
    const GF31 x2w1 = f31_mul_real_c31(x2, root3);
    *y0 = f31_add(f31_add(x0, x1), x2);
    *y1 = f31_add(f31_add(x0, x1w1), x2w2);
    *y2 = f31_add(f31_add(x0, x1w2), x2w1);
}

static inline __attribute__((always_inline))
void f31_dft3_scaled(GF31 x0, GF31 x1, GF31 x2, uint root3, uint scale,
                     __private GF31* y0, __private GF31* y1, __private GF31* y2)
{
    GF31 t0, t1, t2;
    f31_dft3_noscale(x0, x1, x2, root3, &t0, &t1, &t2);
    if (scale != 1u) {
        t0 = f31_mul_real_c31(t0, scale);
        t1 = f31_mul_real_c31(t1, scale);
        t2 = f31_mul_real_c31(t2, scale);
    }
    *y0 = t0; *y1 = t1; *y2 = t2;
}

static inline __attribute__((always_inline))
void f31_dft9_scaled(__private GF31* x, __private GF31* y, uint root9, uint scale)
{
    const uint root3 = f31_mul_scalar(f31_mul_scalar(root9, root9), root9);
    const uint w1 = root9;
    const uint w2 = f31_mul_scalar(root9, root9);
    const uint w4 = f31_mul_scalar(w2, w2);

    GF31 a00, a01, a02, a10, a11, a12, a20, a21, a22;
    f31_dft3_noscale(x[0], x[3], x[6], root3, &a00, &a01, &a02);
    f31_dft3_noscale(x[1], x[4], x[7], root3, &a10, &a11, &a12);
    f31_dft3_noscale(x[2], x[5], x[8], root3, &a20, &a21, &a22);

    a11 = f31_mul_real_c31(a11, w1);
    a21 = f31_mul_real_c31(a21, w2);
    a12 = f31_mul_real_c31(a12, w2);
    a22 = f31_mul_real_c31(a22, w4);

    f31_dft3_noscale(a00, a10, a20, root3, &y[0], &y[3], &y[6]);
    f31_dft3_noscale(a01, a11, a21, root3, &y[1], &y[4], &y[7]);
    f31_dft3_noscale(a02, a12, a22, root3, &y[2], &y[5], &y[8]);

    if (scale != 1u) {
        for (uint i = 0u; i < 9u; ++i) y[i] = f31_mul_real_c31(y[i], scale);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_61(__global const ulong* restrict digits,
                                   __global GF* restrict a61,
                                   uint n, uint p, uint lr2_61,
                                   uint odd, uint pow2_n)
{
    const uint id = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    const uint storage = odd * row_m;
    if (id >= storage) return;
    const uint row = id / row_m;
    const uint k = id - row * row_m;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(row, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(row, b1, odd, pow2_n);
    const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
    const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
    a61[id] = (GF)(lshift61(digits[j0], shift_from_r_on_the_fly(r0, lr2_61)),
                  lshift61(digits[j1], shift_from_r_on_the_fly(r1, lr2_61)));
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_31(__global const ulong* restrict digits,
                                   __global GF31* restrict a31,
                                   uint n, uint p, uint lr2_31,
                                   uint odd, uint pow2_n)
{
    const uint id = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    const uint storage = odd * row_m;
    if (id >= storage) return;
    const uint row = id / row_m;
    const uint k = id - row * row_m;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(row, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(row, b1, odd, pow2_n);
    const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
    const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
    a31[id] = (GF31)(f31_weight_scalar_r(digits[j0], r0, lr2_31),
                    f31_weight_scalar_r(digits[j1], r1, lr2_31));
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_61(__global const ulong* restrict digits,
                                           __global GF* restrict a61,
                                           __global const GF* restrict mat,
                                           uint n, uint p, uint lr2_61,
                                           uint odd, uint pow2_n)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF x[9];
    for (uint in = 0u; in < 9u; ++in) x[in] = GF_ZERO;
    for (uint in = 0u; in < odd; ++in) {
        const uint j0 = crt_mixed_j_from_coord(in, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(in, b1, odd, pow2_n);
        const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
        const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
        x[in] = (GF)(lshift61(digits[j0], shift_from_r_on_the_fly(r0, lr2_61)),
                    lshift61(digits[j1], shift_from_r_on_the_fly(r1, lr2_61)));
    }

    if (odd == 9u) {
        GF y[9];
        const ulong scale = mat[0].s0;
        ulong root = mat[10].s0;
        if (scale != 1ul) root = mul61(root, 9ul);
        gf_dft9_scaled_61(x, y, root, scale);
        for (uint out = 0u; out < 9u; ++out) a61[out * row_m + k] = y[out];
    } else if (odd == 3u) {
        GF y0, y1, y2;
        const ulong scale = mat[0].s0;
        ulong root = mat[4].s0;
        if (scale != 1ul) root = mul61(root, 3ul);
        gf_dft3_scaled_61(x[0], x[1], x[2], root, scale, &y0, &y1, &y2);
        a61[k] = y0;
        a61[row_m + k] = y1;
        a61[(row_m << 1) + k] = y2;
    } else {
        for (uint out = 0u; out < odd; ++out) {
            GF acc = GF_ZERO;
            for (uint in = 0u; in < odd; ++in) {
                const GF w = mat[out * odd + in];
                acc = gf_add(acc, gf_mul_real_odd(x[in], w));
            }
            a61[out * row_m + k] = acc;
        }
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_31(__global const ulong* restrict digits,
                                           __global GF31* restrict a31,
                                           __global const GF31* restrict mat,
                                           uint n, uint p, uint lr2_31,
                                           uint odd, uint pow2_n)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF31 x[9];
    for (uint in = 0u; in < 9u; ++in) x[in] = (GF31)(0u, 0u);
    for (uint in = 0u; in < odd; ++in) {
        const uint j0 = crt_mixed_j_from_coord(in, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(in, b1, odd, pow2_n);
        const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
        const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
        x[in] = (GF31)(f31_weight_scalar_r(digits[j0], r0, lr2_31),
                      f31_weight_scalar_r(digits[j1], r1, lr2_31));
    }

    if (odd == 9u) {
        GF31 y[9];
        const uint scale = mat[0].s0;
        uint root = mat[10].s0;
        if (scale != 1u) root = f31_mul_scalar(root, 9u);
        f31_dft9_scaled(x, y, root, scale);
        for (uint out = 0u; out < 9u; ++out) a31[out * row_m + k] = y[out];
    } else if (odd == 3u) {
        GF31 y0, y1, y2;
        const uint scale = mat[0].s0;
        uint root = mat[4].s0;
        if (scale != 1u) root = f31_mul_scalar(root, 3u);
        f31_dft3_scaled(x[0], x[1], x[2], root, scale, &y0, &y1, &y2);
        a31[k] = y0;
        a31[row_m + k] = y1;
        a31[(row_m << 1) + k] = y2;
    } else {
        for (uint out = 0u; out < odd; ++out) {
            GF31 acc = (GF31)(0u, 0u);
            for (uint in = 0u; in < odd; ++in) {
                const GF31 w = mat[out * odd + in];
                acc = f31_add(acc, f31_mul_real_odd(x[in], w));
            }
            a31[out * row_m + k] = acc;
        }
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile7_61(__global const ulong* restrict digits,
                                                 __global GF* restrict a61,
                                                 __global const GF* restrict mat,
                                                 uint n, uint p, uint lr2_61,
                                                 uint odd, uint pow2_n)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 7u + kt;

    __local GF l61[7 * 9];

    if (kt < 7u && k < row_m && lane < odd) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, odd, pow2_n);
        const uint p2 = p & (pow2_n - 1u);
        const uint podd = p % odd;
        const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
        const uint r0 = crt_mixed_r_from_coord_fast_pre(lane, b0, odd, pow2_n, p2, podd, inv_m);
        const uint r1 = crt_mixed_r_from_coord_fast_pre(lane, b1, odd, pow2_n, p2, podd, inv_m);
        l61[kt * 9u + lane] = (GF)(lshift61(digits[j0], shift_from_r_on_the_fly(r0, lr2_61)),
                                  lshift61(digits[j1], shift_from_r_on_the_fly(r1, lr2_61)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 7u || k >= row_m || lane >= odd) return;

    GF acc = GF_ZERO;
    const uint lbase = kt * 9u;
    const uint mat_base = lane * odd;
    for (uint in = 0u; in < odd; ++in) {
        acc = gf_add(acc, gf_mul_real_odd(l61[lbase + in], mat[mat_base + in]));
    }
    a61[lane * row_m + k] = acc;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile7_31(__global const ulong* restrict digits,
                                                 __global GF31* restrict a31,
                                                 __global const GF31* restrict mat,
                                                 uint n, uint p, uint lr2_31,
                                                 uint odd, uint pow2_n)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 7u + kt;

    __local GF31 l31[7 * 9];

    if (kt < 7u && k < row_m && lane < odd) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, odd, pow2_n);
        const uint p2 = p & (pow2_n - 1u);
        const uint podd = p % odd;
        const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
        const uint r0 = crt_mixed_r_from_coord_fast_pre(lane, b0, odd, pow2_n, p2, podd, inv_m);
        const uint r1 = crt_mixed_r_from_coord_fast_pre(lane, b1, odd, pow2_n, p2, podd, inv_m);
        l31[kt * 9u + lane] = (GF31)(f31_weight_scalar_r(digits[j0], r0, lr2_31),
                                    f31_weight_scalar_r(digits[j1], r1, lr2_31));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 7u || k >= row_m || lane >= odd) return;

    GF31 acc = (GF31)(0u, 0u);
    const uint lbase = kt * 9u;
    const uint mat_base = lane * odd;
    for (uint in = 0u; in < odd; ++in) {
        acc = f31_add(acc, f31_mul_real_odd(l31[lbase + in], mat[mat_base + in]));
    }
    a31[lane * row_m + k] = acc;
}



__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_61(__global const ulong* restrict digits,
                                                 __global GF* restrict a61,
                                                 __global const GF* restrict mat,
                                                 uint n, uint p, uint lr2_61,
                                                 uint odd, uint pow2_n)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];

    if (kt < 14u && k < row_m && lane < odd) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, odd, pow2_n);
        const uint p2 = p & (pow2_n - 1u);
        const uint podd = p % odd;
        const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
        const uint r0 = crt_mixed_r_from_coord_fast_pre(lane, b0, odd, pow2_n, p2, podd, inv_m);
        const uint r1 = crt_mixed_r_from_coord_fast_pre(lane, b1, odd, pow2_n, p2, podd, inv_m);
        l61[kt * 9u + lane] = (GF)(lshift61(digits[j0], shift_from_r_on_the_fly(r0, lr2_61)),
                                  lshift61(digits[j1], shift_from_r_on_the_fly(r1, lr2_61)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= odd) return;

    GF acc = GF_ZERO;
    const uint lbase = kt * 9u;
    const uint mat_base = lane * odd;
    for (uint in = 0u; in < odd; ++in) {
        acc = gf_add(acc, gf_mul_real_odd(l61[lbase + in], mat[mat_base + in]));
    }
    a61[lane * row_m + k] = acc;
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_31(__global const ulong* restrict digits,
                                                 __global GF31* restrict a31,
                                                 __global const GF31* restrict mat,
                                                 uint n, uint p, uint lr2_31,
                                                 uint odd, uint pow2_n)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF31 l31[14 * 9];

    if (kt < 14u && k < row_m && lane < odd) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, odd, pow2_n);
        const uint p2 = p & (pow2_n - 1u);
        const uint podd = p % odd;
        const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
        const uint r0 = crt_mixed_r_from_coord_fast_pre(lane, b0, odd, pow2_n, p2, podd, inv_m);
        const uint r1 = crt_mixed_r_from_coord_fast_pre(lane, b1, odd, pow2_n, p2, podd, inv_m);
        l31[kt * 9u + lane] = (GF31)(f31_weight_scalar_r(digits[j0], r0, lr2_31),
                                    f31_weight_scalar_r(digits[j1], r1, lr2_31));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= odd) return;

    GF31 acc = (GF31)(0u, 0u);
    const uint lbase = kt * 9u;
    const uint mat_base = lane * odd;
    for (uint in = 0u; in < odd; ++in) {
        acc = f31_add(acc, f31_mul_real_odd(l31[lbase + in], mat[mat_base + in]));
    }
    a31[lane * row_m + k] = acc;
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_61(__global const ulong* restrict digits,
                                                       __global GF* restrict a61,
                                                       __global const GF* restrict mat,
                                                       __global const uchar* restrict shift61,
                                                       uint odd, uint pow2_n)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];

    if (kt < 14u && k < row_m && lane < 9u) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);
        l61[kt * 9u + lane] = (GF)(lshift61(digits[j0], (uint)shift61[j0]),
                                  lshift61(digits[j1], (uint)shift61[j1]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    GF acc = GF_ZERO;
    const uint lbase = kt * 9u;
    const uint mat_base = lane * 9u;
    for (uint in = 0u; in < 9u; ++in) {
        acc = gf_add(acc, gf_mul_real_odd(l61[lbase + in], mat[mat_base + in]));
    }
    a61[lane * row_m + k] = acc;
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_31(__global const ulong* restrict digits,
                                                       __global GF31* restrict a31,
                                                       __global const GF31* restrict mat,
                                                       __global const uchar* restrict shift31,
                                                       uint odd, uint pow2_n)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF31 l31[14 * 9];

    if (kt < 14u && k < row_m && lane < 9u) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);
        l31[kt * 9u + lane] = (GF31)(f31_lshift_scalar(f31_reduce_ulong(digits[j0]), (uint)shift31[j0]),
                                    f31_lshift_scalar(f31_reduce_ulong(digits[j1]), (uint)shift31[j1]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    GF31 acc = (GF31)(0u, 0u);
    const uint lbase = kt * 9u;
    const uint mat_base = lane * 9u;
    for (uint in = 0u; in < 9u; ++in) {
        acc = f31_add(acc, f31_mul_real_odd(l31[lbase + in], mat[mat_base + in]));
    }
    a31[lane * row_m + k] = acc;
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61(__global const ulong* restrict digits,
                                                            __global GF* restrict a61,
                                                            __global const GF* restrict mat,
                                                            __global const uchar* restrict shift61,
                                                            uint odd, uint pow2_n)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];
    __local GF lm61[9 * 9];

    if (lid < 81u) lm61[lid] = mat[lid];
    if (kt < 14u && k < row_m && lane < 9u) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);
        l61[kt * 9u + lane] = (GF)(lshift61(digits[j0], (uint)shift61[j0]),
                                  lshift61(digits[j1], (uint)shift61[j1]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    const uint lbase = kt * 9u;
    const uint mbase = lane * 9u;
    GF acc = GF_ZERO;
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 0u], lm61[mbase + 0u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 1u], lm61[mbase + 1u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 2u], lm61[mbase + 2u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 3u], lm61[mbase + 3u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 4u], lm61[mbase + 4u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 5u], lm61[mbase + 5u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 6u], lm61[mbase + 6u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 7u], lm61[mbase + 7u]));
    acc = gf_add(acc, gf_mul_real_odd(l61[lbase + 8u], lm61[mbase + 8u]));
    a61[lane * row_m + k] = acc;
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_31(__global const ulong* restrict digits,
                                                            __global GF31* restrict a31,
                                                            __global const GF31* restrict mat,
                                                            __global const uchar* restrict shift31,
                                                            uint odd, uint pow2_n)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF31 l31[14 * 9];
    __local GF31 lm31[9 * 9];

    if (lid < 81u) lm31[lid] = mat[lid];
    if (kt < 14u && k < row_m && lane < 9u) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);
        l31[kt * 9u + lane] = (GF31)(f31_lshift_scalar(f31_reduce_ulong(digits[j0]), (uint)shift31[j0]),
                                    f31_lshift_scalar(f31_reduce_ulong(digits[j1]), (uint)shift31[j1]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    const uint lbase = kt * 9u;
    const uint mbase = lane * 9u;
    GF31 acc = (GF31)(0u, 0u);
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 0u], lm31[mbase + 0u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 1u], lm31[mbase + 1u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 2u], lm31[mbase + 2u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 3u], lm31[mbase + 3u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 4u], lm31[mbase + 4u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 5u], lm31[mbase + 5u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 6u], lm31[mbase + 6u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 7u], lm31[mbase + 7u]));
    acc = f31_add(acc, f31_mul_real_odd(l31[lbase + 8u], lm31[mbase + 8u]));
    a31[lane * row_m + k] = acc;
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61x31(__global const ulong* restrict digits,
                                                               __global GF* restrict a61,
                                                               __global GF31* restrict a31,
                                                               __global const GF* restrict mat61,
                                                               __global const GF31* restrict mat31,
                                                               __global const uchar* restrict shift61,
                                                               __global const uchar* restrict shift31,
                                                               uint odd, uint pow2_n)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];
    __local GF31 l31[14 * 9];
    __local GF lm61[9 * 9];
    __local GF31 lm31[9 * 9];

    if (lid < 81u) {
        lm61[lid] = mat61[lid];
        lm31[lid] = mat31[lid];
    }

    if (kt < 14u && k < row_m && lane < 9u) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);
        const ulong d0 = digits[j0];
        const ulong d1 = digits[j1];
        l61[kt * 9u + lane] = (GF)(lshift61(d0, (uint)shift61[j0]),
                                  lshift61(d1, (uint)shift61[j1]));
        l31[kt * 9u + lane] = (GF31)(f31_lshift_scalar(f31_reduce_ulong(d0), (uint)shift31[j0]),
                                    f31_lshift_scalar(f31_reduce_ulong(d1), (uint)shift31[j1]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    const uint lbase = kt * 9u;
    const uint mbase = lane * 9u;
    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 0u], lm61[mbase + 0u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 0u], lm31[mbase + 0u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 1u], lm61[mbase + 1u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 1u], lm31[mbase + 1u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 2u], lm61[mbase + 2u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 2u], lm31[mbase + 2u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 3u], lm61[mbase + 3u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 3u], lm31[mbase + 3u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 4u], lm61[mbase + 4u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 4u], lm31[mbase + 4u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 5u], lm61[mbase + 5u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 5u], lm31[mbase + 5u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 6u], lm61[mbase + 6u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 6u], lm31[mbase + 6u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 7u], lm61[mbase + 7u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 7u], lm31[mbase + 7u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 8u], lm61[mbase + 8u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 8u], lm31[mbase + 8u]));

    const uint out = lane * row_m + k;
    a61[out] = acc61;
    a31[out] = acc31;
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_tile7_61x31(__global const ulong* restrict digits,
                                                    __global GF* restrict a61,
                                                    __global GF31* restrict a31,
                                                    __global const GF* restrict mat61,
                                                    __global const GF31* restrict mat31,
                                                    uint n, uint p, uint lr2_61, uint lr2_31,
                                                    uint odd, uint pow2_n)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 7u + kt;

    __local GF l61[7 * 9];
    __local GF31 l31[7 * 9];

    if (kt < 7u && k < row_m && lane < odd) {
        const uint b0 = k << 1;
        const uint b1 = b0 + 1u;
        const uint j0 = crt_mixed_j_from_coord(lane, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(lane, b1, odd, pow2_n);
        const uint p2 = p & (pow2_n - 1u);
        const uint podd = p % odd;
        const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
        const uint r0 = crt_mixed_r_from_coord_fast_pre(lane, b0, odd, pow2_n, p2, podd, inv_m);
        const uint r1 = crt_mixed_r_from_coord_fast_pre(lane, b1, odd, pow2_n, p2, podd, inv_m);
        const ulong d0 = digits[j0];
        const ulong d1 = digits[j1];
        l61[kt * 9u + lane] = (GF)(lshift61(d0, shift_from_r_on_the_fly(r0, lr2_61)),
                                  lshift61(d1, shift_from_r_on_the_fly(r1, lr2_61)));
        l31[kt * 9u + lane] = (GF31)(f31_weight_scalar_r(d0, r0, lr2_31),
                                    f31_weight_scalar_r(d1, r1, lr2_31));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= odd) return;

    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    const uint lbase = kt * 9u;
    const uint mat_base = lane * odd;
    for (uint in = 0u; in < odd; ++in) {
        acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + in], mat61[mat_base + in]));
        acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + in], mat31[mat_base + in]));
    }
    const uint out = lane * row_m + k;
    a61[out] = acc61;
    a31[out] = acc31;
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_shift_61(__global const ulong* restrict digits,
                                                 __global GF* restrict a61,
                                                 __global const GF* restrict mat,
                                                 __global const uchar* restrict shift61,
                                                 uint odd, uint pow2_n)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF x[9];
    for (uint in = 0u; in < 9u; ++in) x[in] = GF_ZERO;
    for (uint in = 0u; in < odd; ++in) {
        const uint j0 = crt_mixed_j_from_coord(in, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(in, b1, odd, pow2_n);
        x[in] = (GF)(lshift61(digits[j0], (uint)shift61[j0]),
                    lshift61(digits[j1], (uint)shift61[j1]));
    }

    if (odd == 9u) {
        GF y[9];
        const ulong scale = mat[0].s0;
        ulong root = mat[10].s0;
        if (scale != 1ul) root = mul61(root, 9ul);
        gf_dft9_scaled_61(x, y, root, scale);
        for (uint out = 0u; out < 9u; ++out) a61[out * row_m + k] = y[out];
    } else {
        GF y0, y1, y2;
        const ulong scale = mat[0].s0;
        ulong root = mat[4].s0;
        if (scale != 1ul) root = mul61(root, 3ul);
        gf_dft3_scaled_61(x[0], x[1], x[2], root, scale, &y0, &y1, &y2);
        a61[k] = y0;
        a61[row_m + k] = y1;
        a61[(row_m << 1) + k] = y2;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_shift_31(__global const ulong* restrict digits,
                                                 __global GF31* restrict a31,
                                                 __global const GF31* restrict mat,
                                                 __global const uchar* restrict shift31,
                                                 uint odd, uint pow2_n)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF31 x[9];
    for (uint in = 0u; in < 9u; ++in) x[in] = (GF31)(0u, 0u);
    for (uint in = 0u; in < odd; ++in) {
        const uint j0 = crt_mixed_j_from_coord(in, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(in, b1, odd, pow2_n);
        x[in] = (GF31)(f31_lshift_scalar(f31_reduce_ulong(digits[j0]), (uint)shift31[j0]),
                      f31_lshift_scalar(f31_reduce_ulong(digits[j1]), (uint)shift31[j1]));
    }

    if (odd == 9u) {
        GF31 y[9];
        const uint scale = mat[0].s0;
        uint root = mat[10].s0;
        if (scale != 1u) root = f31_mul_scalar(root, 9u);
        f31_dft9_scaled(x, y, root, scale);
        for (uint out = 0u; out < 9u; ++out) a31[out * row_m + k] = y[out];
    } else {
        GF31 y0, y1, y2;
        const uint scale = mat[0].s0;
        uint root = mat[4].s0;
        if (scale != 1u) root = f31_mul_scalar(root, 3u);
        f31_dft3_scaled(x[0], x[1], x[2], root, scale, &y0, &y1, &y2);
        a31[k] = y0;
        a31[row_m + k] = y1;
        a31[(row_m << 1) + k] = y2;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_pack_weight_odd_fwd_both_shift(__global const ulong* restrict digits,
                                                   __global GF* restrict a61,
                                                   __global GF31* restrict a31,
                                                   __global const GF* restrict mat61,
                                                   __global const GF31* restrict mat31,
                                                   __global const uchar* restrict shift61,
                                                   __global const uchar* restrict shift31,
                                                   uint odd, uint pow2_n)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF x61[9];
    GF31 x31[9];
    for (uint in = 0u; in < 9u; ++in) {
        x61[in] = GF_ZERO;
        x31[in] = (GF31)(0u, 0u);
    }
    for (uint in = 0u; in < odd; ++in) {
        const uint j0 = crt_mixed_j_from_coord(in, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(in, b1, odd, pow2_n);
        const ulong d0 = digits[j0];
        const ulong d1 = digits[j1];
        x61[in] = (GF)(lshift61(d0, (uint)shift61[j0]),
                      lshift61(d1, (uint)shift61[j1]));
        x31[in] = (GF31)(f31_lshift_scalar(f31_reduce_ulong(d0), (uint)shift31[j0]),
                        f31_lshift_scalar(f31_reduce_ulong(d1), (uint)shift31[j1]));
    }

    if (odd == 9u) {
        GF y61[9];
        const ulong scale61 = mat61[0].s0;
        ulong root61 = mat61[10].s0;
        if (scale61 != 1ul) root61 = mul61(root61, 9ul);
        gf_dft9_scaled_61(x61, y61, root61, scale61);
        GF31 y31[9];
        const uint scale31 = mat31[0].s0;
        uint root31 = mat31[10].s0;
        if (scale31 != 1u) root31 = f31_mul_scalar(root31, 9u);
        f31_dft9_scaled(x31, y31, root31, scale31);
        for (uint out = 0u; out < 9u; ++out) {
            const uint idx = out * row_m + k;
            a61[idx] = y61[out];
            a31[idx] = y31[out];
        }
    } else {
        GF y610, y611, y612;
        const ulong scale61 = mat61[0].s0;
        ulong root61 = mat61[4].s0;
        if (scale61 != 1ul) root61 = mul61(root61, 3ul);
        gf_dft3_scaled_61(x61[0], x61[1], x61[2], root61, scale61, &y610, &y611, &y612);
        GF31 y310, y311, y312;
        const uint scale31 = mat31[0].s0;
        uint root31 = mat31[4].s0;
        if (scale31 != 1u) root31 = f31_mul_scalar(root31, 3u);
        f31_dft3_scaled(x31[0], x31[1], x31[2], root31, scale31, &y310, &y311, &y312);
        a61[k] = y610; a61[row_m + k] = y611; a61[(row_m << 1) + k] = y612;
        a31[k] = y310; a31[row_m + k] = y311; a31[(row_m << 1) + k] = y312;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_inv_unpack_unweight_61(__global const GF* restrict a61,
                                               __global ulong* restrict digits61,
                                               __global const GF* restrict mat,
                                               uint n, uint p, uint lr2_61,
                                               uint odd, uint pow2_n, uint log_m)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF x[9];
    for (uint in = 0u; in < 9u; ++in) x[in] = GF_ZERO;
    for (uint in = 0u; in < odd; ++in) x[in] = a61[in * row_m + k];

    GF y[9];
    for (uint out = 0u; out < 9u; ++out) y[out] = GF_ZERO;
    if (odd == 9u) {
        const ulong scale = mat[0].s0;
        ulong root = mat[10].s0;
        if (scale != 1ul) root = mul61(root, 9ul);
        gf_dft9_scaled_61(x, y, root, scale);
    } else if (odd == 3u) {
        const ulong scale = mat[0].s0;
        ulong root = mat[4].s0;
        if (scale != 1ul) root = mul61(root, 3ul);
        gf_dft3_scaled_61(x[0], x[1], x[2], root, scale, &y[0], &y[1], &y[2]);
    } else {
        for (uint out = 0u; out < odd; ++out) {
            GF acc = GF_ZERO;
            for (uint in = 0u; in < odd; ++in) {
                const GF w = mat[out * odd + in];
                acc = gf_add(acc, gf_mul_real_odd(x[in], w));
            }
            y[out] = acc;
        }
    }

    for (uint out = 0u; out < odd; ++out) {
        const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);
        const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
        const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
        const GF z = y[out];
        digits61[j0] = rshift61(norm61(z.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
        digits61[j1] = rshift61(norm61(z.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_inv_unpack_unweight_31(__global const GF31* restrict a31,
                                               __global uint* restrict digits31,
                                               __global const GF31* restrict mat,
                                               uint n, uint p, uint lr2_31,
                                               uint odd, uint pow2_n, uint log_m)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF31 x[9];
    for (uint in = 0u; in < 9u; ++in) x[in] = (GF31)(0u, 0u);
    for (uint in = 0u; in < odd; ++in) x[in] = a31[in * row_m + k];

    GF31 y[9];
    for (uint out = 0u; out < 9u; ++out) y[out] = (GF31)(0u, 0u);
    if (odd == 9u) {
        const uint scale = mat[0].s0;
        uint root = mat[10].s0;
        if (scale != 1u) root = f31_mul_scalar(root, 9u);
        f31_dft9_scaled(x, y, root, scale);
    } else if (odd == 3u) {
        const uint scale = mat[0].s0;
        uint root = mat[4].s0;
        if (scale != 1u) root = f31_mul_scalar(root, 3u);
        f31_dft3_scaled(x[0], x[1], x[2], root, scale, &y[0], &y[1], &y[2]);
    } else {
        for (uint out = 0u; out < odd; ++out) {
            GF31 acc = (GF31)(0u, 0u);
            for (uint in = 0u; in < odd; ++in) {
                const GF31 w = mat[out * odd + in];
                acc = f31_add(acc, f31_mul_real_odd(x[in], w));
            }
            y[out] = acc;
        }
    }

    for (uint out = 0u; out < odd; ++out) {
        const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);
        const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
        const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
        const GF31 z = y[out];
        uint s0 = f31_mod31_small(shift_from_r31(r0, lr2_31) + log_m);
        uint s1 = f31_mod31_small(shift_from_r31(r1, lr2_31) + log_m);
        digits31[j0] = f31_lshift_scalar(z.s0, s0 == 0u ? 0u : 31u - s0);
        digits31[j1] = f31_lshift_scalar(z.s1, s1 == 0u ? 0u : 31u - s1);
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi(__global const GF* restrict a61,
                                            __global const GF31* restrict a31,
                                            __global ulong* restrict coeff_lo,
                                            __global uint* restrict coeff_hi,
                                            __global const GF* restrict mat61,
                                            __global const GF31* restrict mat31,
                                            uint n, uint p,
                                            uint lr2_61, uint lr2_31,
                                            uint odd, uint pow2_n, uint log_m)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF x61[9];
    GF31 x31[9];
    for (uint in = 0u; in < 9u; ++in) {
        x61[in] = GF_ZERO;
        x31[in] = (GF31)(0u, 0u);
    }
    for (uint in = 0u; in < odd; ++in) {
        const uint idx = in * row_m + k;
        x61[in] = a61[idx];
        x31[in] = a31[idx];
    }

    GF y61[9];
    GF31 y31[9];
    for (uint out = 0u; out < 9u; ++out) {
        y61[out] = GF_ZERO;
        y31[out] = (GF31)(0u, 0u);
    }

    if (odd == 9u) {
        const ulong scale61 = mat61[0].s0;
        ulong root61 = mat61[10].s0;
        if (scale61 != 1ul) root61 = mul61(root61, 9ul);
        gf_dft9_scaled_61(x61, y61, root61, scale61);

        const uint scale31 = mat31[0].s0;
        uint root31 = mat31[10].s0;
        if (scale31 != 1u) root31 = f31_mul_scalar(root31, 9u);
        f31_dft9_scaled(x31, y31, root31, scale31);
    } else if (odd == 3u) {
        const ulong scale61 = mat61[0].s0;
        ulong root61 = mat61[4].s0;
        if (scale61 != 1ul) root61 = mul61(root61, 3ul);
        gf_dft3_scaled_61(x61[0], x61[1], x61[2], root61, scale61, &y61[0], &y61[1], &y61[2]);

        const uint scale31 = mat31[0].s0;
        uint root31 = mat31[4].s0;
        if (scale31 != 1u) root31 = f31_mul_scalar(root31, 3u);
        f31_dft3_scaled(x31[0], x31[1], x31[2], root31, scale31, &y31[0], &y31[1], &y31[2]);
    } else {
        for (uint out = 0u; out < odd; ++out) {
            GF acc61 = GF_ZERO;
            GF31 acc31 = (GF31)(0u, 0u);
            for (uint in = 0u; in < odd; ++in) {
                acc61 = gf_add(acc61, gf_mul_real_odd(x61[in], mat61[out * odd + in]));
                acc31 = f31_add(acc31, f31_mul_real_odd(x31[in], mat31[out * odd + in]));
            }
            y61[out] = acc61;
            y31[out] = acc31;
        }
    }

    
    const uint p2 = p & (pow2_n - 1u);
    const uint podd = p % odd;
    const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
    for (uint out = 0u; out < odd; ++out) {
        const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);
        const uint r0 = crt_mixed_r_from_coord_fast_pre(out, b0, odd, pow2_n, p2, podd, inv_m);
        const uint r1 = crt_mixed_r_from_coord_fast_pre(out, b1, odd, pow2_n, p2, podd, inv_m);

        const GF z61 = y61[out];
        const GF31 z31 = y31[out];

        const ulong a61_0 = rshift61(norm61(z61.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
        const ulong a61_1 = rshift61(norm61(z61.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);

        const uint s0 = f31_mod31_small(shift_from_r31(r0, lr2_31) + log_m);
        const uint s1 = f31_mod31_small(shift_from_r31(r1, lr2_31) + log_m);
        const uint a31_0 = f31_lshift_scalar(z31.s0, s0 == 0u ? 0u : 31u - s0);
        const uint a31_1 = f31_lshift_scalar(z31.s1, s1 == 0u ? 0u : 31u - s1);

        ulong lo, hi;
        crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
        coeff_lo[j0] = lo;
        coeff_hi[j0] = (uint)hi;
        crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
        coeff_lo[j1] = lo;
        coeff_hi[j1] = (uint)hi;
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi_tile7(__global const GF* restrict a61,
                                                  __global const GF31* restrict a31,
                                                  __global ulong* restrict coeff_lo,
                                                  __global uint* restrict coeff_hi,
                                                  __global const GF* restrict mat61,
                                                  __global const GF31* restrict mat31,
                                                  uint n, uint p,
                                                  uint lr2_61, uint lr2_31,
                                                  uint odd, uint pow2_n, uint log_m)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 7u + kt;

    __local GF l61[7 * 9];
    __local GF31 l31[7 * 9];

    if (kt < 7u && k < row_m && lane < odd) {
        const uint idx = lane * row_m + k;
        l61[kt * 9u + lane] = a61[idx];
        l31[kt * 9u + lane] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 7u || k >= row_m || lane >= odd) return;

    const uint out = lane;
    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    const uint mat_base = out * odd;
    const uint lbase = kt * 9u;

    
    for (uint in = 0u; in < odd; ++in) {
        acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + in], mat61[mat_base + in]));
        acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + in], mat31[mat_base + in]));
    }

    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);

    const uint p2 = p & (pow2_n - 1u);
    const uint podd = p % odd;
    const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
    const uint r0 = crt_mixed_r_from_coord_fast_pre(out, b0, odd, pow2_n, p2, podd, inv_m);
    const uint r1 = crt_mixed_r_from_coord_fast_pre(out, b1, odd, pow2_n, p2, podd, inv_m);

    const ulong a61_0 = rshift61(norm61(acc61.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
    const ulong a61_1 = rshift61(norm61(acc61.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);

    const uint s0 = f31_mod31_small(shift_from_r31(r0, lr2_31) + log_m);
    const uint s1 = f31_mod31_small(shift_from_r31(r1, lr2_31) + log_m);
    const uint a31_0 = f31_lshift_scalar(acc31.s0, s0 == 0u ? 0u : 31u - s0);
    const uint a31_1 = f31_lshift_scalar(acc31.s1, s1 == 0u ? 0u : 31u - s1);

    ulong lo, hi;
    crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
    coeff_lo[j0] = lo;
    coeff_hi[j0] = (uint)hi;
    crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
    coeff_lo[j1] = lo;
    coeff_hi[j1] = (uint)hi;
}



__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi_tile14(__global const GF* restrict a61,
                                                  __global const GF31* restrict a31,
                                                  __global ulong* restrict coeff_lo,
                                                  __global uint* restrict coeff_hi,
                                                  __global const GF* restrict mat61,
                                                  __global const GF31* restrict mat31,
                                                  uint n, uint p,
                                                  uint lr2_61, uint lr2_31,
                                                  uint odd, uint pow2_n, uint log_m)
{
    (void)n;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];
    __local GF31 l31[14 * 9];

    if (kt < 14u && k < row_m && lane < odd) {
        const uint idx = lane * row_m + k;
        l61[kt * 9u + lane] = a61[idx];
        l31[kt * 9u + lane] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= odd) return;

    const uint out = lane;
    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    const uint mat_base = out * odd;
    const uint lbase = kt * 9u;

    
    for (uint in = 0u; in < odd; ++in) {
        acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + in], mat61[mat_base + in]));
        acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + in], mat31[mat_base + in]));
    }

    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);

    const uint p2 = p & (pow2_n - 1u);
    const uint podd = p % odd;
    const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
    const uint r0 = crt_mixed_r_from_coord_fast_pre(out, b0, odd, pow2_n, p2, podd, inv_m);
    const uint r1 = crt_mixed_r_from_coord_fast_pre(out, b1, odd, pow2_n, p2, podd, inv_m);

    const ulong a61_0 = rshift61(norm61(acc61.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
    const ulong a61_1 = rshift61(norm61(acc61.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);

    const uint s0 = f31_mod31_small(shift_from_r31(r0, lr2_31) + log_m);
    const uint s1 = f31_mod31_small(shift_from_r31(r1, lr2_31) + log_m);
    const uint a31_0 = f31_lshift_scalar(acc31.s0, s0 == 0u ? 0u : 31u - s0);
    const uint a31_1 = f31_lshift_scalar(acc31.s1, s1 == 0u ? 0u : 31u - s1);

    ulong lo, hi;
    crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
    coeff_lo[j0] = lo;
    coeff_hi[j0] = (uint)hi;
    crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
    coeff_lo[j1] = lo;
    coeff_hi[j1] = (uint)hi;
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi_tile14_shift(__global const GF* restrict a61,
                                                       __global const GF31* restrict a31,
                                                       __global ulong* restrict coeff_lo,
                                                       __global uint* restrict coeff_hi,
                                                       __global const GF* restrict mat61,
                                                       __global const GF31* restrict mat31,
                                                       __global const uchar* restrict shift61,
                                                       __global const uchar* restrict shift31,
                                                       uint odd, uint pow2_n, uint log_m)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];
    __local GF31 l31[14 * 9];

    if (kt < 14u && k < row_m && lane < 9u) {
        const uint idx = lane * row_m + k;
        l61[kt * 9u + lane] = a61[idx];
        l31[kt * 9u + lane] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    const uint mat_base = lane * 9u;
    const uint lbase = kt * 9u;

    for (uint in = 0u; in < 9u; ++in) {
        acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + in], mat61[mat_base + in]));
        acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + in], mat31[mat_base + in]));
    }

    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);

    const ulong a61_0 = rshift61(norm61(acc61.s0), (uint)shift61[j0] + log_m);
    const ulong a61_1 = rshift61(norm61(acc61.s1), (uint)shift61[j1] + log_m);

    const uint s0 = f31_mod31_small((uint)shift31[j0] + log_m);
    const uint s1 = f31_mod31_small((uint)shift31[j1] + log_m);
    const uint a31_0 = f31_lshift_scalar(acc31.s0, s0 == 0u ? 0u : 31u - s0);
    const uint a31_1 = f31_lshift_scalar(acc31.s1, s1 == 0u ? 0u : 31u - s1);

    ulong lo, hi;
    crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
    coeff_lo[j0] = lo;
    coeff_hi[j0] = (uint)hi;
    crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
    coeff_lo[j1] = lo;
    coeff_hi[j1] = (uint)hi;
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat(__global const GF* restrict a61,
                                                            __global const GF31* restrict a31,
                                                            __global ulong* restrict coeff_lo,
                                                            __global uint* restrict coeff_hi,
                                                            __global const GF* restrict mat61,
                                                            __global const GF31* restrict mat31,
                                                            __global const uchar* restrict shift61,
                                                            __global const uchar* restrict shift31,
                                                            uint odd, uint pow2_n, uint log_m)
{
    (void)odd;
    const uint lid = (uint)get_local_id(0);
    const uint kt = lid / 9u;
    const uint lane = lid - kt * 9u;
    const uint row_m = pow2_n >> 1;
    const uint k = (uint)get_group_id(0) * 14u + kt;

    __local GF l61[14 * 9];
    __local GF31 l31[14 * 9];
    __local GF lm61[9 * 9];
    __local GF31 lm31[9 * 9];

    if (lid < 81u) {
        lm61[lid] = mat61[lid];
        lm31[lid] = mat31[lid];
    }
    if (kt < 14u && k < row_m && lane < 9u) {
        const uint idx = lane * row_m + k;
        l61[kt * 9u + lane] = a61[idx];
        l31[kt * 9u + lane] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (kt >= 14u || k >= row_m || lane >= 9u) return;

    const uint lbase = kt * 9u;
    const uint mbase = lane * 9u;
    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 0u], lm61[mbase + 0u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 0u], lm31[mbase + 0u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 1u], lm61[mbase + 1u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 1u], lm31[mbase + 1u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 2u], lm61[mbase + 2u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 2u], lm31[mbase + 2u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 3u], lm61[mbase + 3u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 3u], lm31[mbase + 3u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 4u], lm61[mbase + 4u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 4u], lm31[mbase + 4u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 5u], lm61[mbase + 5u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 5u], lm31[mbase + 5u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 6u], lm61[mbase + 6u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 6u], lm31[mbase + 6u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 7u], lm61[mbase + 7u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 7u], lm31[mbase + 7u]));
    acc61 = gf_add(acc61, gf_mul_real_odd(l61[lbase + 8u], lm61[mbase + 8u]));
    acc31 = f31_add(acc31, f31_mul_real_odd(l31[lbase + 8u], lm31[mbase + 8u]));

    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(lane, b0, 9u, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(lane, b1, 9u, pow2_n);

    const ulong a61_0 = rshift61(norm61(acc61.s0), (uint)shift61[j0] + log_m);
    const ulong a61_1 = rshift61(norm61(acc61.s1), (uint)shift61[j1] + log_m);

    const uint s0 = f31_mod31_small((uint)shift31[j0] + log_m);
    const uint s1 = f31_mod31_small((uint)shift31[j1] + log_m);
    const uint a31_0 = f31_lshift_scalar(acc31.s0, s0 == 0u ? 0u : 31u - s0);
    const uint a31_1 = f31_lshift_scalar(acc31.s1, s1 == 0u ? 0u : 31u - s1);

    ulong lo, hi;
    crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
    coeff_lo[j0] = lo;
    coeff_hi[j0] = (uint)hi;
    crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
    coeff_lo[j1] = lo;
    coeff_hi[j1] = (uint)hi;
}



__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd9_inv_precrt_garner64_lmat(__global const GF* restrict a61,
                                                   __global const GF31* restrict a31,
                                                   __global const GF* restrict mat61,
                                                   __global const GF31* restrict mat31,
                                                   __global const uchar* restrict shift61,
                                                   __global const uchar* restrict shift31,
                                                   __global const uint* restrict width_mask32,
                                                   uint width_base,
                                                   __global ulong* restrict digits61,
                                                   __global ulong* restrict carry_lo_out,
                                                   __global ulong* restrict carry_hi_out,
                                                   __global uint* restrict pending,
                                                   uint pow2_n, uint log_m,
                                                   uint digit_n, uint segments)
{
    const uint lid = (uint)get_local_id(0);
    const uint seg = (uint)get_group_id(0);
    if (seg >= segments) return;

    const uint row_m = pow2_n >> 1;
    const uint start = seg << 6;
    const uint i = start + lid;

    __local GF l61[32 * 9];
    __local GF31 l31[32 * 9];
    __local GF lm61[9 * 9];
    __local GF31 lm31[9 * 9];
    __local ulong coeff_lo[64];
    __local ulong coeff_hi[64];

    if (lid < 64u) {
        lm61[lid] = mat61[lid];
        lm31[lid] = mat31[lid];
    }
    if (lid < 17u) {
        lm61[64u + lid] = mat61[64u + lid];
        lm31[64u + lid] = mat31[64u + lid];
    }

    if (lid < 32u) {
        const uint i0 = start + (lid << 1);
        const uint base = lid * 9u;
        if (i0 < digit_n) {
            const uint b = i0 & (pow2_n - 1u);
            const uint k = b >> 1;
            l61[base + 0u] = a61[0u * row_m + k];
            l31[base + 0u] = a31[0u * row_m + k];
            l61[base + 1u] = a61[1u * row_m + k];
            l31[base + 1u] = a31[1u * row_m + k];
            l61[base + 2u] = a61[2u * row_m + k];
            l31[base + 2u] = a31[2u * row_m + k];
            l61[base + 3u] = a61[3u * row_m + k];
            l31[base + 3u] = a31[3u * row_m + k];
            l61[base + 4u] = a61[4u * row_m + k];
            l31[base + 4u] = a31[4u * row_m + k];
            l61[base + 5u] = a61[5u * row_m + k];
            l31[base + 5u] = a31[5u * row_m + k];
            l61[base + 6u] = a61[6u * row_m + k];
            l31[base + 6u] = a31[6u * row_m + k];
            l61[base + 7u] = a61[7u * row_m + k];
            l31[base + 7u] = a31[7u * row_m + k];
            l61[base + 8u] = a61[8u * row_m + k];
            l31[base + 8u] = a31[8u * row_m + k];
        } else {
            l61[base + 0u] = GF_ZERO; l31[base + 0u] = (GF31)(0u, 0u);
            l61[base + 1u] = GF_ZERO; l31[base + 1u] = (GF31)(0u, 0u);
            l61[base + 2u] = GF_ZERO; l31[base + 2u] = (GF31)(0u, 0u);
            l61[base + 3u] = GF_ZERO; l31[base + 3u] = (GF31)(0u, 0u);
            l61[base + 4u] = GF_ZERO; l31[base + 4u] = (GF31)(0u, 0u);
            l61[base + 5u] = GF_ZERO; l31[base + 5u] = (GF31)(0u, 0u);
            l61[base + 6u] = GF_ZERO; l31[base + 6u] = (GF31)(0u, 0u);
            l61[base + 7u] = GF_ZERO; l31[base + 7u] = (GF31)(0u, 0u);
            l61[base + 8u] = GF_ZERO; l31[base + 8u] = (GF31)(0u, 0u);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ulong lo = 0ul;
    ulong hi = 0ul;
    if (i < digit_n) {
        const uint pair = lid >> 1;
        const uint component = lid & 1u;
        const uint lane = i % 9u;
        const uint lbase = pair * 9u;
        const uint mbase = lane * 9u;

        ulong acc61 = 0ul;
        uint acc31 = 0u;

        GF v610 = l61[lbase + 0u];
        GF31 v310 = l31[lbase + 0u];
        acc61 = add61_lazy(acc61, mul61(component ? v610.s1 : v610.s0, lm61[mbase + 0u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v310.s1 : v310.s0, lm31[mbase + 0u].s0));
        GF v611 = l61[lbase + 1u];
        GF31 v311 = l31[lbase + 1u];
        acc61 = add61_lazy(acc61, mul61(component ? v611.s1 : v611.s0, lm61[mbase + 1u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v311.s1 : v311.s0, lm31[mbase + 1u].s0));
        GF v612 = l61[lbase + 2u];
        GF31 v312 = l31[lbase + 2u];
        acc61 = add61_lazy(acc61, mul61(component ? v612.s1 : v612.s0, lm61[mbase + 2u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v312.s1 : v312.s0, lm31[mbase + 2u].s0));
        GF v613 = l61[lbase + 3u];
        GF31 v313 = l31[lbase + 3u];
        acc61 = add61_lazy(acc61, mul61(component ? v613.s1 : v613.s0, lm61[mbase + 3u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v313.s1 : v313.s0, lm31[mbase + 3u].s0));
        GF v614 = l61[lbase + 4u];
        GF31 v314 = l31[lbase + 4u];
        acc61 = add61_lazy(acc61, mul61(component ? v614.s1 : v614.s0, lm61[mbase + 4u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v314.s1 : v314.s0, lm31[mbase + 4u].s0));
        GF v615 = l61[lbase + 5u];
        GF31 v315 = l31[lbase + 5u];
        acc61 = add61_lazy(acc61, mul61(component ? v615.s1 : v615.s0, lm61[mbase + 5u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v315.s1 : v315.s0, lm31[mbase + 5u].s0));
        GF v616 = l61[lbase + 6u];
        GF31 v316 = l31[lbase + 6u];
        acc61 = add61_lazy(acc61, mul61(component ? v616.s1 : v616.s0, lm61[mbase + 6u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v316.s1 : v316.s0, lm31[mbase + 6u].s0));
        GF v617 = l61[lbase + 7u];
        GF31 v317 = l31[lbase + 7u];
        acc61 = add61_lazy(acc61, mul61(component ? v617.s1 : v617.s0, lm61[mbase + 7u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v317.s1 : v317.s0, lm31[mbase + 7u].s0));
        GF v618 = l61[lbase + 8u];
        GF31 v318 = l31[lbase + 8u];
        acc61 = add61_lazy(acc61, mul61(component ? v618.s1 : v618.s0, lm61[mbase + 8u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v318.s1 : v318.s0, lm31[mbase + 8u].s0));

        const ulong r61 = rshift61(norm61(acc61), (uint)shift61[i] + log_m);
        const uint s31 = f31_mod31_small((uint)shift31[i] + log_m);
        const uint r31 = f31_lshift_scalar(acc31, s31 == 0u ? 0u : 31u - s31);
        crt_coeff_from_residues(r61, (ulong)r31, &lo, &hi);
    }
    coeff_lo[lid] = lo;
    coeff_hi[lid] = hi;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        const uint mask0 = width_mask32[seg << 1];
        const uint mask1 = ((start + 32u) < digit_n) ? width_mask32[(seg << 1) + 1u] : 0u;
        ulong clo = 0ul;
        ulong chi = 0ul;

        if (width_base == 30u && start + 63u < digit_n) {
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint bit = (mask0 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[j], clo, &c);
                const ulong total_hi = coeff_hi[j] + chi + c;
                if (bit == 0u) {
                    digits61[start + j] = total_lo & 0x3ffffffful;
                    clo = (total_lo >> 30u) | (total_hi << 34u);
                    chi = total_hi >> 30u;
                } else {
                    digits61[start + j] = total_lo & 0x7ffffffful;
                    clo = (total_lo >> 31u) | (total_hi << 33u);
                    chi = total_hi >> 31u;
                }
            }
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint jj = 32u + j;
                const uint bit = (mask1 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[jj], clo, &c);
                const ulong total_hi = coeff_hi[jj] + chi + c;
                if (bit == 0u) {
                    digits61[start + jj] = total_lo & 0x3ffffffful;
                    clo = (total_lo >> 30u) | (total_hi << 34u);
                    chi = total_hi >> 30u;
                } else {
                    digits61[start + jj] = total_lo & 0x7ffffffful;
                    clo = (total_lo >> 31u) | (total_hi << 33u);
                    chi = total_hi >> 31u;
                }
            }
        } else if (width_base == 32u && start + 63u < digit_n) {
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint bit = (mask0 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[j], clo, &c);
                const ulong total_hi = coeff_hi[j] + chi + c;
                if (bit == 0u) {
                    digits61[start + j] = total_lo & 0xfffffffful;
                    clo = (total_lo >> 32u) | (total_hi << 32u);
                    chi = total_hi >> 32u;
                } else {
                    digits61[start + j] = total_lo & 0x1fffffffful;
                    clo = (total_lo >> 33u) | (total_hi << 31u);
                    chi = total_hi >> 33u;
                }
            }
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint jj = 32u + j;
                const uint bit = (mask1 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[jj], clo, &c);
                const ulong total_hi = coeff_hi[jj] + chi + c;
                if (bit == 0u) {
                    digits61[start + jj] = total_lo & 0xfffffffful;
                    clo = (total_lo >> 32u) | (total_hi << 32u);
                    chi = total_hi >> 32u;
                } else {
                    digits61[start + jj] = total_lo & 0x1fffffffful;
                    clo = (total_lo >> 33u) | (total_hi << 31u);
                    chi = total_hi >> 33u;
                }
            }
        } else {
            for (uint j = 0u; j < 64u; ++j) {
                const uint d = start + j;
                if (d >= digit_n) break;
                const uint mask = (j < 32u) ? mask0 : mask1;
                const uint bit = (mask >> (j & 31u)) & 1u;
                ulong total_lo, total_hi;
                crt_add_u128(coeff_lo[j], coeff_hi[j], clo, chi, &total_lo, &total_hi);
                digits61[d] = crt_take_digit_and_shift_u128(total_lo, total_hi, width_base + bit, &clo, &chi);
            }
        }
        const uint next = (seg + 1u < segments) ? (seg + 1u) : 0u;
        carry_lo_out[next] = clo;
        carry_hi_out[next] = chi;
        if ((clo | chi) != 0ul) pending[0] = 1u;
    }
}


__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gf61_crt_mixed_odd9_inv_precrt_garner9seg_lmat(__global const GF* restrict a61,
                                                     __global const GF31* restrict a31,
                                                     __global const GF* restrict mat61,
                                                     __global const GF31* restrict mat31,
                                                     __global const uchar* restrict shift61,
                                                     __global const uchar* restrict shift31,
                                                     __global const uint* restrict width_mask32,
                                                     uint width_base,
                                                     __global ulong* restrict digits61,
                                                     __global ulong* restrict carry_lo_out,
                                                     __global ulong* restrict carry_hi_out,
                                                     __global uint* restrict pending,
                                                     uint pow2_n, uint log_m,
                                                     uint digit_n, uint segments)
{
    const uint lid = (uint)get_local_id(0);
    const uint bseg = (uint)get_group_id(0);
    const uint row_m = pow2_n >> 1;
    const uint segs_per_t = pow2_n >> 6;   // pow2_n / 64, current odd9 path only
    if (bseg >= segs_per_t) return;

    const uint b_start = bseg << 6;
    const uint mmod = pow2_n % 9u;

    __local GF l61[32 * 9];
    __local GF31 l31[32 * 9];
    __local GF lm61[9 * 9];
    __local GF31 lm31[9 * 9];
    __local ulong coeff_lo[9 * 64];
    __local uint coeff_hi[9 * 64];

    for (uint t = lid; t < 81u; t += 256u) {
        lm61[t] = mat61[t];
        lm31[t] = mat31[t];
    }

    for (uint t = lid; t < 32u * 9u; t += 256u) {
        const uint pair = t / 9u;
        const uint row = t - pair * 9u;
        const uint k = (b_start >> 1) + pair;
        l61[t] = a61[row * row_m + k];
        l31[t] = a31[row * row_m + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 9 natural 64-digit segments together.  This keeps the good odd-axis
    // reuse: for each b/k pair, the 9 CRT lanes are computed while the 9 input
    // rows are still in LDS.  The old v43 natural-64 fused kernel lost this reuse.
    for (uint idx = lid; idx < 9u * 64u; idx += 256u) {
        const uint t = idx >> 6;          // CRT/PFA block, natural segment in this t
        const uint q = idx & 63u;         // offset inside the 64-digit segment
        const uint b = b_start + q;
        const uint pair = q >> 1;
        const uint component = q & 1u;
        const uint lane = (b + mmod * t) % 9u;
        const uint lbase = pair * 9u;
        const uint mbase = lane * 9u;
        const uint j = b + pow2_n * t;

        ulong acc61 = 0ul;
        uint acc31 = 0u;

        GF v610 = l61[lbase + 0u];
        GF31 v310 = l31[lbase + 0u];
        acc61 = add61_lazy(acc61, mul61(component ? v610.s1 : v610.s0, lm61[mbase + 0u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v310.s1 : v310.s0, lm31[mbase + 0u].s0));
        GF v611 = l61[lbase + 1u];
        GF31 v311 = l31[lbase + 1u];
        acc61 = add61_lazy(acc61, mul61(component ? v611.s1 : v611.s0, lm61[mbase + 1u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v311.s1 : v311.s0, lm31[mbase + 1u].s0));
        GF v612 = l61[lbase + 2u];
        GF31 v312 = l31[lbase + 2u];
        acc61 = add61_lazy(acc61, mul61(component ? v612.s1 : v612.s0, lm61[mbase + 2u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v312.s1 : v312.s0, lm31[mbase + 2u].s0));
        GF v613 = l61[lbase + 3u];
        GF31 v313 = l31[lbase + 3u];
        acc61 = add61_lazy(acc61, mul61(component ? v613.s1 : v613.s0, lm61[mbase + 3u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v313.s1 : v313.s0, lm31[mbase + 3u].s0));
        GF v614 = l61[lbase + 4u];
        GF31 v314 = l31[lbase + 4u];
        acc61 = add61_lazy(acc61, mul61(component ? v614.s1 : v614.s0, lm61[mbase + 4u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v314.s1 : v314.s0, lm31[mbase + 4u].s0));
        GF v615 = l61[lbase + 5u];
        GF31 v315 = l31[lbase + 5u];
        acc61 = add61_lazy(acc61, mul61(component ? v615.s1 : v615.s0, lm61[mbase + 5u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v315.s1 : v315.s0, lm31[mbase + 5u].s0));
        GF v616 = l61[lbase + 6u];
        GF31 v316 = l31[lbase + 6u];
        acc61 = add61_lazy(acc61, mul61(component ? v616.s1 : v616.s0, lm61[mbase + 6u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v316.s1 : v316.s0, lm31[mbase + 6u].s0));
        GF v617 = l61[lbase + 7u];
        GF31 v317 = l31[lbase + 7u];
        acc61 = add61_lazy(acc61, mul61(component ? v617.s1 : v617.s0, lm61[mbase + 7u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v317.s1 : v317.s0, lm31[mbase + 7u].s0));
        GF v618 = l61[lbase + 8u];
        GF31 v318 = l31[lbase + 8u];
        acc61 = add61_lazy(acc61, mul61(component ? v618.s1 : v618.s0, lm61[mbase + 8u].s0));
        acc31 = f31_add_scalar(acc31, f31_mul_scalar(component ? v318.s1 : v318.s0, lm31[mbase + 8u].s0));

        const ulong r61 = rshift61(norm61(acc61), (uint)shift61[j] + log_m);
        const uint s31 = f31_mod31_small((uint)shift31[j] + log_m);
        const uint r31 = f31_lshift_scalar(acc31, s31 == 0u ? 0u : 31u - s31);
        ulong lo, hi;
        crt_coeff_from_residues(r61, (ulong)r31, &lo, &hi);
        coeff_lo[idx] = lo;
        coeff_hi[idx] = (uint)hi;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 9u) {
        const uint t = lid;
        const uint seg = bseg + t * segs_per_t;
        const uint start_j = b_start + t * pow2_n;
        const uint mask0 = width_mask32[seg << 1];
        const uint mask1 = width_mask32[(seg << 1) + 1u];
        ulong clo = 0ul;
        ulong chi = 0ul;
        const uint base = t << 6;

        if (width_base == 30u && start_j + 63u < digit_n) {
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint bit = (mask0 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[base + j], clo, &c);
                const ulong total_hi = (ulong)coeff_hi[base + j] + chi + c;
                if (bit == 0u) {
                    digits61[start_j + j] = total_lo & 0x3ffffffful;
                    clo = (total_lo >> 30u) | (total_hi << 34u);
                    chi = total_hi >> 30u;
                } else {
                    digits61[start_j + j] = total_lo & 0x7ffffffful;
                    clo = (total_lo >> 31u) | (total_hi << 33u);
                    chi = total_hi >> 31u;
                }
            }
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint jj = 32u + j;
                const uint bit = (mask1 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[base + jj], clo, &c);
                const ulong total_hi = (ulong)coeff_hi[base + jj] + chi + c;
                if (bit == 0u) {
                    digits61[start_j + jj] = total_lo & 0x3ffffffful;
                    clo = (total_lo >> 30u) | (total_hi << 34u);
                    chi = total_hi >> 30u;
                } else {
                    digits61[start_j + jj] = total_lo & 0x7ffffffful;
                    clo = (total_lo >> 31u) | (total_hi << 33u);
                    chi = total_hi >> 31u;
                }
            }
        } else if (width_base == 32u && start_j + 63u < digit_n) {
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint bit = (mask0 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[base + j], clo, &c);
                const ulong total_hi = (ulong)coeff_hi[base + j] + chi + c;
                if (bit == 0u) {
                    digits61[start_j + j] = total_lo & 0xfffffffful;
                    clo = (total_lo >> 32u) | (total_hi << 32u);
                    chi = total_hi >> 32u;
                } else {
                    digits61[start_j + j] = total_lo & 0x1fffffffful;
                    clo = (total_lo >> 33u) | (total_hi << 31u);
                    chi = total_hi >> 33u;
                }
            }
            #pragma unroll 32
            for (uint j = 0u; j < 32u; ++j) {
                const uint jj = 32u + j;
                const uint bit = (mask1 >> j) & 1u;
                ulong c = 0ul;
                const ulong total_lo = crt_add_carry_lo(coeff_lo[base + jj], clo, &c);
                const ulong total_hi = (ulong)coeff_hi[base + jj] + chi + c;
                if (bit == 0u) {
                    digits61[start_j + jj] = total_lo & 0xfffffffful;
                    clo = (total_lo >> 32u) | (total_hi << 32u);
                    chi = total_hi >> 32u;
                } else {
                    digits61[start_j + jj] = total_lo & 0x1fffffffful;
                    clo = (total_lo >> 33u) | (total_hi << 31u);
                    chi = total_hi >> 33u;
                }
            }
        } else {
            for (uint j = 0u; j < 64u; ++j) {
                const uint d = start_j + j;
                if (d >= digit_n) break;
                const uint mask = (j < 32u) ? mask0 : mask1;
                const uint bit = (mask >> (j & 31u)) & 1u;
                ulong total_lo, total_hi;
                crt_add_u128(coeff_lo[base + j], (ulong)coeff_hi[base + j], clo, chi, &total_lo, &total_hi);
                digits61[d] = crt_take_digit_and_shift_u128(total_lo, total_hi, width_base + bit, &clo, &chi);
            }
        }
        const uint next = (seg + 1u < segments) ? (seg + 1u) : 0u;
        carry_lo_out[next] = clo;
        carry_hi_out[next] = chi;
        if ((clo | chi) != 0ul) pending[0] = 1u;
    }
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_odd9_inv_precrt_garner9seg30_pair_lmat(__global const GF* restrict a61,
                                                           __global const GF31* restrict a31,
                                                           __global const GF* restrict mat61,
                                                           __global const GF31* restrict mat31,
                                                           __global const uchar* restrict shift61,
                                                           __global const uchar* restrict shift31,
                                                           __global const uint* restrict width_mask32,
                                                           uint width_base,
                                                           __global ulong* restrict digits61,
                                                           __global ulong* restrict carry_lo_out,
                                                           __global ulong* restrict carry_hi_out,
                                                           __global uint* restrict pending,
                                                           uint pow2_n, uint log_m,
                                                           uint digit_n, uint segments)
{
    (void)width_base;
    (void)digit_n;
    (void)segments;
    const uint lid = (uint)get_local_id(0);
    const uint bseg = (uint)get_group_id(0);
    const uint row_m = pow2_n >> 1;
    const uint segs_per_t = pow2_n >> 6;
    if (bseg >= segs_per_t) return;

    const uint b_start = bseg << 6;
    const uint mmod = pow2_n % 9u;

    __local GF l61[32 * 9];
    __local GF31 l31[32 * 9];
    __local GF lm61[9 * 9];
    __local GF31 lm31[9 * 9];
    __local ulong coeff_lo[9 * 64];
    __local uint coeff_hi[9 * 64];

    for (uint t = lid; t < 81u; t += 128u) {
        lm61[t] = mat61[t];
        lm31[t] = mat31[t];
    }

    for (uint t = lid; t < 32u * 9u; t += 128u) {
        const uint pair = t / 9u;
        const uint row = t - pair * 9u;
        const uint k = (b_start >> 1) + pair;
        l61[t] = a61[row * row_m + k];
        l31[t] = a31[row * row_m + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* v45 fast path for the common digit-width-base=30 case.
       One work item computes the even and odd component for the same pair.
       That keeps the same 9-segment layout as v44, but loads the 9 input rows
       once instead of doing the component 0 and component 1 dot products as
       two independent work items. */
    for (uint idx = lid; idx < 9u * 32u; idx += 128u) {
        const uint t = idx >> 5;          // 0..8
        const uint pair = idx & 31u;      // 0..31
        const uint b0 = b_start + (pair << 1);
        const uint b1 = b0 + 1u;
        const uint lane0 = (b0 + mmod * t) % 9u;
        const uint lane1 = (b1 + mmod * t) % 9u;
        const uint lbase = pair * 9u;
        const uint mbase0 = lane0 * 9u;
        const uint mbase1 = lane1 * 9u;
        const uint j0 = b0 + pow2_n * t;
        const uint j1 = j0 + 1u;

        ulong acc610 = 0ul;
        ulong acc611 = 0ul;
        uint acc310 = 0u;
        uint acc311 = 0u;

        GF v610 = l61[lbase + 0u];
        GF31 v310 = l31[lbase + 0u];
        acc610 = add61_lazy(acc610, mul61(v610.s0, lm61[mbase0 + 0u].s0));
        acc611 = add61_lazy(acc611, mul61(v610.s1, lm61[mbase1 + 0u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v310.s0, lm31[mbase0 + 0u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v310.s1, lm31[mbase1 + 0u].s0));

        GF v611 = l61[lbase + 1u];
        GF31 v311 = l31[lbase + 1u];
        acc610 = add61_lazy(acc610, mul61(v611.s0, lm61[mbase0 + 1u].s0));
        acc611 = add61_lazy(acc611, mul61(v611.s1, lm61[mbase1 + 1u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v311.s0, lm31[mbase0 + 1u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v311.s1, lm31[mbase1 + 1u].s0));

        GF v612 = l61[lbase + 2u];
        GF31 v312 = l31[lbase + 2u];
        acc610 = add61_lazy(acc610, mul61(v612.s0, lm61[mbase0 + 2u].s0));
        acc611 = add61_lazy(acc611, mul61(v612.s1, lm61[mbase1 + 2u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v312.s0, lm31[mbase0 + 2u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v312.s1, lm31[mbase1 + 2u].s0));

        GF v613 = l61[lbase + 3u];
        GF31 v313 = l31[lbase + 3u];
        acc610 = add61_lazy(acc610, mul61(v613.s0, lm61[mbase0 + 3u].s0));
        acc611 = add61_lazy(acc611, mul61(v613.s1, lm61[mbase1 + 3u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v313.s0, lm31[mbase0 + 3u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v313.s1, lm31[mbase1 + 3u].s0));

        GF v614 = l61[lbase + 4u];
        GF31 v314 = l31[lbase + 4u];
        acc610 = add61_lazy(acc610, mul61(v614.s0, lm61[mbase0 + 4u].s0));
        acc611 = add61_lazy(acc611, mul61(v614.s1, lm61[mbase1 + 4u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v314.s0, lm31[mbase0 + 4u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v314.s1, lm31[mbase1 + 4u].s0));

        GF v615 = l61[lbase + 5u];
        GF31 v315 = l31[lbase + 5u];
        acc610 = add61_lazy(acc610, mul61(v615.s0, lm61[mbase0 + 5u].s0));
        acc611 = add61_lazy(acc611, mul61(v615.s1, lm61[mbase1 + 5u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v315.s0, lm31[mbase0 + 5u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v315.s1, lm31[mbase1 + 5u].s0));

        GF v616 = l61[lbase + 6u];
        GF31 v316 = l31[lbase + 6u];
        acc610 = add61_lazy(acc610, mul61(v616.s0, lm61[mbase0 + 6u].s0));
        acc611 = add61_lazy(acc611, mul61(v616.s1, lm61[mbase1 + 6u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v316.s0, lm31[mbase0 + 6u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v316.s1, lm31[mbase1 + 6u].s0));

        GF v617 = l61[lbase + 7u];
        GF31 v317 = l31[lbase + 7u];
        acc610 = add61_lazy(acc610, mul61(v617.s0, lm61[mbase0 + 7u].s0));
        acc611 = add61_lazy(acc611, mul61(v617.s1, lm61[mbase1 + 7u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v317.s0, lm31[mbase0 + 7u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v317.s1, lm31[mbase1 + 7u].s0));

        GF v618 = l61[lbase + 8u];
        GF31 v318 = l31[lbase + 8u];
        acc610 = add61_lazy(acc610, mul61(v618.s0, lm61[mbase0 + 8u].s0));
        acc611 = add61_lazy(acc611, mul61(v618.s1, lm61[mbase1 + 8u].s0));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v318.s0, lm31[mbase0 + 8u].s0));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v318.s1, lm31[mbase1 + 8u].s0));

        const ulong r610 = rshift61(norm61(acc610), (uint)shift61[j0] + log_m);
        const ulong r611 = rshift61(norm61(acc611), (uint)shift61[j1] + log_m);
        const uint s310 = f31_mod31_small((uint)shift31[j0] + log_m);
        const uint s311 = f31_mod31_small((uint)shift31[j1] + log_m);
        const uint r310 = f31_lshift_scalar(acc310, s310 == 0u ? 0u : 31u - s310);
        const uint r311 = f31_lshift_scalar(acc311, s311 == 0u ? 0u : 31u - s311);

        ulong lo0, hi0, lo1, hi1;
        crt_coeff_from_residues(r610, (ulong)r310, &lo0, &hi0);
        crt_coeff_from_residues(r611, (ulong)r311, &lo1, &hi1);
        const uint out = (t << 6) + (pair << 1);
        coeff_lo[out] = lo0;
        coeff_hi[out] = (uint)hi0;
        coeff_lo[out + 1u] = lo1;
        coeff_hi[out + 1u] = (uint)hi1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 9u) {
        const uint t = lid;
        const uint seg = bseg + t * segs_per_t;
        const uint start_j = b_start + t * pow2_n;
        const uint mask0 = width_mask32[seg << 1];
        const uint mask1 = width_mask32[(seg << 1) + 1u];
        ulong clo = 0ul;
        ulong chi = 0ul;
        const uint base = t << 6;

        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            const uint bit = (mask0 >> j) & 1u;
            ulong c = 0ul;
            const ulong total_lo = crt_add_carry_lo(coeff_lo[base + j], clo, &c);
            const ulong total_hi = (ulong)coeff_hi[base + j] + chi + c;
            if (bit == 0u) {
                digits61[start_j + j] = total_lo & 0x3ffffffful;
                clo = (total_lo >> 30u) | (total_hi << 34u);
                chi = total_hi >> 30u;
            } else {
                digits61[start_j + j] = total_lo & 0x7ffffffful;
                clo = (total_lo >> 31u) | (total_hi << 33u);
                chi = total_hi >> 31u;
            }
        }
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            const uint jj = 32u + j;
            const uint bit = (mask1 >> j) & 1u;
            ulong c = 0ul;
            const ulong total_lo = crt_add_carry_lo(coeff_lo[base + jj], clo, &c);
            const ulong total_hi = (ulong)coeff_hi[base + jj] + chi + c;
            if (bit == 0u) {
                digits61[start_j + jj] = total_lo & 0x3ffffffful;
                clo = (total_lo >> 30u) | (total_hi << 34u);
                chi = total_hi >> 30u;
            } else {
                digits61[start_j + jj] = total_lo & 0x7ffffffful;
                clo = (total_lo >> 31u) | (total_hi << 33u);
                chi = total_hi >> 31u;
            }
        }

        const uint next = (seg + 1u < 9u * segs_per_t) ? (seg + 1u) : 0u;
        carry_lo_out[next] = clo;
        carry_hi_out[next] = chi;
        if ((clo | chi) != 0ul) pending[0] = 1u;
    }
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void gf61_crt_mixed_odd9_inv_precrt_garner9seg30_pair_smat(__global const GF* restrict a61,
                                                           __global const GF31* restrict a31,
                                                           __global const GF* restrict mat61,
                                                           __global const GF31* restrict mat31,
                                                           __global const uchar* restrict shift61,
                                                           __global const uchar* restrict shift31,
                                                           __global const uint* restrict width_mask32,
                                                           uint width_base,
                                                           __global ulong* restrict digits61,
                                                           __global ulong* restrict carry_lo_out,
                                                           __global ulong* restrict carry_hi_out,
                                                           __global uint* restrict pending,
                                                           uint pow2_n, uint log_m,
                                                           uint digit_n, uint segments)
{
    (void)width_base;
    (void)digit_n;
    (void)segments;
    const uint lid = (uint)get_local_id(0);
    const uint bseg = (uint)get_group_id(0);
    const uint row_m = pow2_n >> 1;
    const uint segs_per_t = pow2_n >> 6;
    if (bseg >= segs_per_t) return;

    const uint b_start = bseg << 6;
    const uint mmod = pow2_n % 9u;

    __local GF l61[32 * 9];
    __local GF31 l31[32 * 9];
    __local ulong lm61r[9 * 9];
    __local uint lm31r[9 * 9];
    __local ulong coeff_lo[9 * 64];
    __local uint coeff_hi[9 * 64];

    for (uint t = lid; t < 81u; t += 128u) {
        lm61r[t] = mat61[t].s0;
        lm31r[t] = mat31[t].s0;
    }

    for (uint t = lid; t < 32u * 9u; t += 128u) {
        const uint pair = t / 9u;
        const uint row = t - pair * 9u;
        const uint k = (b_start >> 1) + pair;
        l61[t] = a61[row * row_m + k];
        l31[t] = a31[row * row_m + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* v45 fast path for the common digit-width-base=30 case.
       One work item computes the even and odd component for the same pair.
       That keeps the same 9-segment layout as v44, but loads the 9 input rows
       once instead of doing the component 0 and component 1 dot products as
       two independent work items. */
    for (uint idx = lid; idx < 9u * 32u; idx += 128u) {
        const uint t = idx >> 5;          // 0..8
        const uint pair = idx & 31u;      // 0..31
        const uint b0 = b_start + (pair << 1);
        const uint b1 = b0 + 1u;
        const uint lane0 = (b0 + mmod * t) % 9u;
        const uint lane1 = (b1 + mmod * t) % 9u;
        const uint lbase = pair * 9u;
        const uint mbase0 = lane0 * 9u;
        const uint mbase1 = lane1 * 9u;
        const uint j0 = b0 + pow2_n * t;
        const uint j1 = j0 + 1u;

        ulong acc610 = 0ul;
        ulong acc611 = 0ul;
        uint acc310 = 0u;
        uint acc311 = 0u;

        GF v610 = l61[lbase + 0u];
        GF31 v310 = l31[lbase + 0u];
        acc610 = add61_lazy(acc610, mul61(v610.s0, lm61r[mbase0 + 0u]));
        acc611 = add61_lazy(acc611, mul61(v610.s1, lm61r[mbase1 + 0u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v310.s0, lm31r[mbase0 + 0u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v310.s1, lm31r[mbase1 + 0u]));

        GF v611 = l61[lbase + 1u];
        GF31 v311 = l31[lbase + 1u];
        acc610 = add61_lazy(acc610, mul61(v611.s0, lm61r[mbase0 + 1u]));
        acc611 = add61_lazy(acc611, mul61(v611.s1, lm61r[mbase1 + 1u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v311.s0, lm31r[mbase0 + 1u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v311.s1, lm31r[mbase1 + 1u]));

        GF v612 = l61[lbase + 2u];
        GF31 v312 = l31[lbase + 2u];
        acc610 = add61_lazy(acc610, mul61(v612.s0, lm61r[mbase0 + 2u]));
        acc611 = add61_lazy(acc611, mul61(v612.s1, lm61r[mbase1 + 2u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v312.s0, lm31r[mbase0 + 2u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v312.s1, lm31r[mbase1 + 2u]));

        GF v613 = l61[lbase + 3u];
        GF31 v313 = l31[lbase + 3u];
        acc610 = add61_lazy(acc610, mul61(v613.s0, lm61r[mbase0 + 3u]));
        acc611 = add61_lazy(acc611, mul61(v613.s1, lm61r[mbase1 + 3u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v313.s0, lm31r[mbase0 + 3u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v313.s1, lm31r[mbase1 + 3u]));

        GF v614 = l61[lbase + 4u];
        GF31 v314 = l31[lbase + 4u];
        acc610 = add61_lazy(acc610, mul61(v614.s0, lm61r[mbase0 + 4u]));
        acc611 = add61_lazy(acc611, mul61(v614.s1, lm61r[mbase1 + 4u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v314.s0, lm31r[mbase0 + 4u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v314.s1, lm31r[mbase1 + 4u]));

        GF v615 = l61[lbase + 5u];
        GF31 v315 = l31[lbase + 5u];
        acc610 = add61_lazy(acc610, mul61(v615.s0, lm61r[mbase0 + 5u]));
        acc611 = add61_lazy(acc611, mul61(v615.s1, lm61r[mbase1 + 5u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v315.s0, lm31r[mbase0 + 5u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v315.s1, lm31r[mbase1 + 5u]));

        GF v616 = l61[lbase + 6u];
        GF31 v316 = l31[lbase + 6u];
        acc610 = add61_lazy(acc610, mul61(v616.s0, lm61r[mbase0 + 6u]));
        acc611 = add61_lazy(acc611, mul61(v616.s1, lm61r[mbase1 + 6u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v316.s0, lm31r[mbase0 + 6u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v316.s1, lm31r[mbase1 + 6u]));

        GF v617 = l61[lbase + 7u];
        GF31 v317 = l31[lbase + 7u];
        acc610 = add61_lazy(acc610, mul61(v617.s0, lm61r[mbase0 + 7u]));
        acc611 = add61_lazy(acc611, mul61(v617.s1, lm61r[mbase1 + 7u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v317.s0, lm31r[mbase0 + 7u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v317.s1, lm31r[mbase1 + 7u]));

        GF v618 = l61[lbase + 8u];
        GF31 v318 = l31[lbase + 8u];
        acc610 = add61_lazy(acc610, mul61(v618.s0, lm61r[mbase0 + 8u]));
        acc611 = add61_lazy(acc611, mul61(v618.s1, lm61r[mbase1 + 8u]));
        acc310 = f31_add_scalar(acc310, f31_mul_scalar(v318.s0, lm31r[mbase0 + 8u]));
        acc311 = f31_add_scalar(acc311, f31_mul_scalar(v318.s1, lm31r[mbase1 + 8u]));

        const ulong r610 = rshift61(norm61(acc610), (uint)shift61[j0] + log_m);
        const ulong r611 = rshift61(norm61(acc611), (uint)shift61[j1] + log_m);
        const uint s310 = f31_mod31_small((uint)shift31[j0] + log_m);
        const uint s311 = f31_mod31_small((uint)shift31[j1] + log_m);
        const uint r310 = f31_lshift_scalar(acc310, s310 == 0u ? 0u : 31u - s310);
        const uint r311 = f31_lshift_scalar(acc311, s311 == 0u ? 0u : 31u - s311);

        ulong lo0, hi0, lo1, hi1;
        crt_coeff_from_residues(r610, (ulong)r310, &lo0, &hi0);
        crt_coeff_from_residues(r611, (ulong)r311, &lo1, &hi1);
        const uint out = (t << 6) + (pair << 1);
        coeff_lo[out] = lo0;
        coeff_hi[out] = (uint)hi0;
        coeff_lo[out + 1u] = lo1;
        coeff_hi[out + 1u] = (uint)hi1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 9u) {
        const uint t = lid;
        const uint seg = bseg + t * segs_per_t;
        const uint start_j = b_start + t * pow2_n;
        const uint mask0 = width_mask32[seg << 1];
        const uint mask1 = width_mask32[(seg << 1) + 1u];
        ulong clo = 0ul;
        ulong chi = 0ul;
        const uint base = t << 6;

        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            const uint bit = (mask0 >> j) & 1u;
            ulong c = 0ul;
            const ulong total_lo = crt_add_carry_lo(coeff_lo[base + j], clo, &c);
            const ulong total_hi = (ulong)coeff_hi[base + j] + chi + c;
            if (bit == 0u) {
                digits61[start_j + j] = total_lo & 0x3ffffffful;
                clo = (total_lo >> 30u) | (total_hi << 34u);
                chi = total_hi >> 30u;
            } else {
                digits61[start_j + j] = total_lo & 0x7ffffffful;
                clo = (total_lo >> 31u) | (total_hi << 33u);
                chi = total_hi >> 31u;
            }
        }
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            const uint jj = 32u + j;
            const uint bit = (mask1 >> j) & 1u;
            ulong c = 0ul;
            const ulong total_lo = crt_add_carry_lo(coeff_lo[base + jj], clo, &c);
            const ulong total_hi = (ulong)coeff_hi[base + jj] + chi + c;
            if (bit == 0u) {
                digits61[start_j + jj] = total_lo & 0x3ffffffful;
                clo = (total_lo >> 30u) | (total_hi << 34u);
                chi = total_hi >> 30u;
            } else {
                digits61[start_j + jj] = total_lo & 0x7ffffffful;
                clo = (total_lo >> 31u) | (total_hi << 33u);
                chi = total_hi >> 31u;
            }
        }

        const uint next = (seg + 1u < 9u * segs_per_t) ? (seg + 1u) : 0u;
        carry_lo_out[next] = clo;
        carry_hi_out[next] = chi;
        if ((clo | chi) != 0ul) pending[0] = 1u;
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi_outpar(__global const GF* restrict a61,
                                                   __global const GF31* restrict a31,
                                                   __global ulong* restrict coeff_lo,
                                                   __global uint* restrict coeff_hi,
                                                   __global const GF* restrict mat61,
                                                   __global const GF31* restrict mat31,
                                                   uint n, uint p,
                                                   uint lr2_61, uint lr2_31,
                                                   uint odd, uint pow2_n, uint log_m)
{
    const uint id = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    const uint storage = odd * row_m;
    if (id >= storage) return;

    const uint out = id / row_m;
    const uint k = id - out * row_m;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    
    GF acc61 = GF_ZERO;
    GF31 acc31 = (GF31)(0u, 0u);
    const uint mat_base = out * odd;
    for (uint in = 0u; in < odd; ++in) {
        const uint idx = in * row_m + k;
        acc61 = gf_add(acc61, gf_mul_real_odd(a61[idx], mat61[mat_base + in]));
        acc31 = f31_add(acc31, f31_mul_real_odd(a31[idx], mat31[mat_base + in]));
    }

    const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);

    const uint p2 = p & (pow2_n - 1u);
    const uint podd = p % odd;
    const uint inv_m = crt_mixed_inv_pow2_mod_odd(pow2_n, odd);
    const uint r0 = crt_mixed_r_from_coord_fast_pre(out, b0, odd, pow2_n, p2, podd, inv_m);
    const uint r1 = crt_mixed_r_from_coord_fast_pre(out, b1, odd, pow2_n, p2, podd, inv_m);

    const ulong a61_0 = rshift61(norm61(acc61.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
    const ulong a61_1 = rshift61(norm61(acc61.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);

    const uint s0 = f31_mod31_small(shift_from_r31(r0, lr2_31) + log_m);
    const uint s1 = f31_mod31_small(shift_from_r31(r1, lr2_31) + log_m);
    const uint a31_0 = f31_lshift_scalar(acc31.s0, s0 == 0u ? 0u : 31u - s0);
    const uint a31_1 = f31_lshift_scalar(acc31.s1, s1 == 0u ? 0u : 31u - s1);

    ulong lo, hi;
    crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
    coeff_lo[j0] = lo;
    coeff_hi[j0] = (uint)hi;
    crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
    coeff_lo[j1] = lo;
    coeff_hi[j1] = (uint)hi;
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_inv_precrt_coeffhi_shift(__global const GF* restrict a61,
                                                  __global const GF31* restrict a31,
                                                  __global ulong* restrict coeff_lo,
                                                  __global uint* restrict coeff_hi,
                                                  __global const GF* restrict mat61,
                                                  __global const GF31* restrict mat31,
                                                  __global const uchar* restrict shift61,
                                                  __global const uchar* restrict shift31,
                                                  uint odd, uint pow2_n, uint log_m)
{
    const uint k = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    if (k >= row_m) return;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;

    GF x61[9];
    GF31 x31[9];
    for (uint in = 0u; in < 9u; ++in) {
        x61[in] = GF_ZERO;
        x31[in] = (GF31)(0u, 0u);
    }
    for (uint in = 0u; in < odd; ++in) {
        const uint idx = in * row_m + k;
        x61[in] = a61[idx];
        x31[in] = a31[idx];
    }

    GF y61[9];
    GF31 y31[9];
    for (uint out = 0u; out < 9u; ++out) {
        y61[out] = GF_ZERO;
        y31[out] = (GF31)(0u, 0u);
    }

    if (odd == 9u) {
        const ulong scale61 = mat61[0].s0;
        ulong root61 = mat61[10].s0;
        if (scale61 != 1ul) root61 = mul61(root61, 9ul);
        gf_dft9_scaled_61(x61, y61, root61, scale61);

        const uint scale31 = mat31[0].s0;
        uint root31 = mat31[10].s0;
        if (scale31 != 1u) root31 = f31_mul_scalar(root31, 9u);
        f31_dft9_scaled(x31, y31, root31, scale31);
    } else {
        const ulong scale61 = mat61[0].s0;
        ulong root61 = mat61[4].s0;
        if (scale61 != 1ul) root61 = mul61(root61, 3ul);
        gf_dft3_scaled_61(x61[0], x61[1], x61[2], root61, scale61, &y61[0], &y61[1], &y61[2]);

        const uint scale31 = mat31[0].s0;
        uint root31 = mat31[4].s0;
        if (scale31 != 1u) root31 = f31_mul_scalar(root31, 3u);
        f31_dft3_scaled(x31[0], x31[1], x31[2], root31, scale31, &y31[0], &y31[1], &y31[2]);
    }

    for (uint out = 0u; out < odd; ++out) {
        const uint j0 = crt_mixed_j_from_coord(out, b0, odd, pow2_n);
        const uint j1 = crt_mixed_j_from_coord(out, b1, odd, pow2_n);
        const GF z61 = y61[out];
        const GF31 z31 = y31[out];

        const ulong a61_0 = rshift61(norm61(z61.s0), (uint)shift61[j0] + log_m);
        const ulong a61_1 = rshift61(norm61(z61.s1), (uint)shift61[j1] + log_m);

        const uint s0 = f31_mod31_small((uint)shift31[j0] + log_m);
        const uint s1 = f31_mod31_small((uint)shift31[j1] + log_m);
        const uint a31_0 = f31_lshift_scalar(z31.s0, s0 == 0u ? 0u : 31u - s0);
        const uint a31_1 = f31_lshift_scalar(z31.s1, s1 == 0u ? 0u : 31u - s1);

        ulong lo, hi;
        crt_coeff_from_residues(a61_0, (ulong)a31_0, &lo, &hi);
        coeff_lo[j0] = lo;
        coeff_hi[j0] = (uint)hi;
        crt_coeff_from_residues(a61_1, (ulong)a31_1, &lo, &hi);
        coeff_lo[j1] = lo;
        coeff_hi[j1] = (uint)hi;
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_residues_to_coeffhi(__global ulong* restrict coeff_lo_res61,
                                         __global uint* restrict coeff_hi_res31,
                                         uint n)
{
    const uint j = (uint)get_global_id(0);
    if (j >= n) return;
    const ulong r61 = coeff_lo_res61[j];
    const ulong r31 = (ulong)coeff_hi_res31[j];
    ulong lo, hi;
    crt_coeff_from_residues(r61, r31, &lo, &hi);
    coeff_lo_res61[j] = lo;
    coeff_hi_res31[j] = (uint)hi;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_dft_61(__global GF* restrict a61,
                               __global GF* restrict scratch,
                               __global const GF* restrict mat,
                               uint odd, uint row_m)
{
    const uint k = (uint)get_global_id(0);
    if (k >= row_m) return;
    for (uint out = 0u; out < odd; ++out) {
        GF acc = GF_ZERO;
        for (uint in = 0u; in < odd; ++in) {
            const GF x = a61[in * row_m + k];
            const GF w = mat[out * odd + in];
            acc = gf_add(acc, gf_mul_real_odd(x, w));
        }
        scratch[out * row_m + k] = acc;
    }
    for (uint row = 0u; row < odd; ++row) a61[row * row_m + k] = scratch[row * row_m + k];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_odd_dft_31(__global GF31* restrict a31,
                               __global GF31* restrict scratch,
                               __global const GF31* restrict mat,
                               uint odd, uint row_m)
{
    const uint k = (uint)get_global_id(0);
    if (k >= row_m) return;
    for (uint out = 0u; out < odd; ++out) {
        GF31 acc = (GF31)(0u, 0u);
        for (uint in = 0u; in < odd; ++in) {
            const GF31 x = a31[in * row_m + k];
            const GF31 w = mat[out * odd + in];
            acc = f31_add(acc, f31_mul_real_odd(x, w));
        }
        scratch[out * row_m + k] = acc;
    }
    for (uint row = 0u; row < odd; ++row) a31[row * row_m + k] = scratch[row * row_m + k];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_ntt_stage_dif_radix2_61(__global GF* restrict a61,
                                            __global const GF* restrict twiddles,
                                            uint row_m, uint odd,
                                            uint tw_offset, uint len, uint half_len)
{
    const uint gid = (uint)get_global_id(0);
    const uint per_row = row_m >> 1;
    const uint total = odd * per_row;
    if (gid >= total) return;
    const uint row = gid / per_row;
    const uint t = gid - row * per_row;
    const uint block = t / half_len;
    const uint j = t - block * half_len;
    const uint base = row * row_m + block * len;
    const uint i0 = base + j;
    const uint i1 = i0 + half_len;
    const GF u = a61[i0];
    const GF v = a61[i1];
    a61[i0] = gf_add(u, v);
    a61[i1] = gf_mul(gf_sub(u, v), twiddles[tw_offset + j]);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_ntt_stage_dit_radix2_61(__global GF* restrict a61,
                                            __global const GF* restrict twiddles,
                                            uint row_m, uint odd,
                                            uint tw_offset, uint len, uint half_len)
{
    const uint gid = (uint)get_global_id(0);
    const uint per_row = row_m >> 1;
    const uint total = odd * per_row;
    if (gid >= total) return;
    const uint row = gid / per_row;
    const uint t = gid - row * per_row;
    const uint block = t / half_len;
    const uint j = t - block * half_len;
    const uint base = row * row_m + block * len;
    const uint i0 = base + j;
    const uint i1 = i0 + half_len;
    const GF u = a61[i0];
    const GF v = gf_mul(a61[i1], twiddles[tw_offset + j]);
    a61[i0] = gf_add(u, v);
    a61[i1] = gf_sub(u, v);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_ntt_stage_dif_radix2_31(__global GF31* restrict a31,
                                            __global const GF31* restrict twiddles,
                                            uint row_m, uint odd,
                                            uint tw_offset, uint len, uint half_len)
{
    const uint gid = (uint)get_global_id(0);
    const uint per_row = row_m >> 1;
    const uint total = odd * per_row;
    if (gid >= total) return;
    const uint row = gid / per_row;
    const uint t = gid - row * per_row;
    const uint block = t / half_len;
    const uint j = t - block * half_len;
    const uint base = row * row_m + block * len;
    const uint i0 = base + j;
    const uint i1 = i0 + half_len;
    GF31 s, d;
    F31_DIF_MUL(a31[i0], a31[i1], twiddles[tw_offset + j], s, d);
    a31[i0] = s; a31[i1] = d;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_ntt_stage_dit_radix2_31(__global GF31* restrict a31,
                                            __global const GF31* restrict twiddles,
                                            uint row_m, uint odd,
                                            uint tw_offset, uint len, uint half_len)
{
    const uint gid = (uint)get_global_id(0);
    const uint per_row = row_m >> 1;
    const uint total = odd * per_row;
    if (gid >= total) return;
    const uint row = gid / per_row;
    const uint t = gid - row * per_row;
    const uint block = t / half_len;
    const uint j = t - block * half_len;
    const uint base = row * row_m + block * len;
    const uint i0 = base + j;
    const uint i1 = i0 + half_len;
    GF31 s, d;
    F31_DIT_MUL(a31[i0], a31[i1], twiddles[tw_offset + j], s, d);
    a31[i0] = s; a31[i1] = d;
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_center_61(__global GF* restrict a61,
                                       __global const GF* restrict twf61,
                                       __global const GF* restrict twi61,
                                       uint pow2_n, uint odd, uint flags)
{
    const uint row_m = pow2_n >> 1;
    const uint gid = (uint)get_global_id(0);
    const uint row = gid / row_m;
    const uint k = gid - row * row_m;
    if (row >= odd) return;
    const uint km = (row_m - k) & (row_m - 1u);
    if (k > km) return;
    const uint log_m = 31u - (uint)clz(row_m);
    const uint base = row * row_m;
    const uint pk  = base + crt_halfreal_perm_u32(k,  log_m, flags);
    const uint pkm = base + crt_halfreal_perm_u32(km, log_m, flags);
    const GF zk = a61[pk];
    const GF zkm = a61[pkm];
    if ((flags & 32u) && km != k) {
        GF E, O;
        crt_halfreal_center_eo_f48_61(zk, gf_conj_fast(zkm), twf61, pow2_n, k, &E, &O);
        a61[pk] = gf_pack_e_plus_i_o(E, O);
        a61[pkm] = gf_pack_conj_e_plus_i_conj_o(E, O);
    } else {
        a61[pk] = crt_halfreal_center_one_61(zk, gf_conj_fast(zkm), twf61, twi61, pow2_n, k, flags);
        if (km != k) a61[pkm] = crt_halfreal_center_one_61(zkm, gf_conj_fast(zk), twf61, twi61, pow2_n, km, flags);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_center_31(__global GF31* restrict a31,
                                       __global const GF31* restrict twf31,
                                       __global const GF31* restrict twi31,
                                       uint pow2_n, uint odd, uint flags)
{
    const uint row_m = pow2_n >> 1;
    const uint gid = (uint)get_global_id(0);
    const uint row = gid / row_m;
    const uint k = gid - row * row_m;
    if (row >= odd) return;
    const uint km = (row_m - k) & (row_m - 1u);
    if (k > km) return;
    const uint log_m = 31u - (uint)clz(row_m);
    const uint base = row * row_m;
    const uint pk  = base + crt_halfreal_perm_u32(k,  log_m, flags);
    const uint pkm = base + crt_halfreal_perm_u32(km, log_m, flags);
    const GF31 zk = a31[pk];
    const GF31 zkm = a31[pkm];
    if ((flags & 32u) && km != k) {
        GF31 E, O;
        crt_halfreal_center_eo_f48_31(zk, f31_conj_fast(zkm), twf31, pow2_n, k, &E, &O);
        a31[pk] = f31_pack_e_plus_i_o(E, O);
        a31[pkm] = f31_pack_conj_e_plus_i_conj_o(E, O);
    } else {
        a31[pk] = crt_halfreal_center_one_31(zk, f31_conj_fast(zkm), twf31, twi31, pow2_n, k, flags);
        if (km != k) a31[pkm] = crt_halfreal_center_one_31(zkm, f31_conj_fast(zk), twf31, twi31, pow2_n, km, flags);
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_61(__global GF* restrict a61,
                                            __global const GF* restrict twf61,
                                            __global const GF* restrict twi61,
                                            uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const int same = (blockA == blockB);

    __local GF A[512];
    __local GF B[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A[t] = a61[baseA + t];
        if (!same) B[t] = a61[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(A, twf61, 512u, L, lid, 64u);
        if (!same) local_stage_dif_radix8_pow2(B, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    if ((flags & 32u) != 0u) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z = A[t];
                const GF zn = same ? A[tm] : B[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                const GF outK  = gf_pack_e_plus_i_o(E, O);
                const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
                A[t] = outK;
                if (same) {
                    if (tm != t) A[tm] = outKm;
                } else {
                    B[tm] = outKm;
                }
            } else {
                const GF zsmall = B[tm];
                const GF zlarge = A[t];
                GF E, O;
                crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
                const GF outSmall = gf_pack_e_plus_i_o(E, O);
                const GF outLarge = gf_pack_conj_e_plus_i_conj_o(E, O);
                B[tm] = outSmall;
                A[t] = outLarge;
            }
        }
    } else {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z = A[t];
                const GF zn = same ? A[tm] : B[tm];
                A[t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                if (km != k) {
                    const GF outKm = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
                    if (same) A[tm] = outKm;
                    else B[tm] = outKm;
                }
            } else {
                const GF zsmall = B[tm];
                const GF zlarge = A[t];
                B[tm] = crt_halfreal_center_one_61(zsmall, gf_conj_fast(zlarge), twf61, twi61, pow2_n, km, flags);
                A[t]  = crt_halfreal_center_one_61(zlarge, gf_conj_fast(zsmall), twf61, twi61, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(A, twi61, 512u, L, lid, 64u);
        if (!same) local_stage_dit_radix8_pow2(B, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 512u; t += 64u) {
        a61[baseA + t] = A[t];
        if (!same) a61[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_31(__global GF31* restrict a31,
                                            __global const GF31* restrict twf31,
                                            __global const GF31* restrict twi31,
                                            uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const int same = (blockA == blockB);

    __local GF31 A[512];
    __local GF31 B[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A[t] = a31[baseA + t];
        if (!same) B[t] = a31[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(A, twf31, 512u, L, lid, 64u);
        if (!same) crt_local_stage_dif_radix8_31(B, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    if ((flags & 32u) != 0u) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF31 z = A[t];
                const GF31 zn = same ? A[tm] : B[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                const GF31 outK  = f31_pack_e_plus_i_o(E, O);
                const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
                A[t] = outK;
                if (same) {
                    if (tm != t) A[tm] = outKm;
                } else {
                    B[tm] = outKm;
                }
            } else {
                const GF31 zsmall = B[tm];
                const GF31 zlarge = A[t];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
                const GF31 outSmall = f31_pack_e_plus_i_o(E, O);
                const GF31 outLarge = f31_pack_conj_e_plus_i_conj_o(E, O);
                B[tm] = outSmall;
                A[t] = outLarge;
            }
        }
    } else {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF31 z = A[t];
                const GF31 zn = same ? A[tm] : B[tm];
                A[t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                if (km != k) {
                    const GF31 outKm = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
                    if (same) A[tm] = outKm;
                    else B[tm] = outKm;
                }
            } else {
                const GF31 zsmall = B[tm];
                const GF31 zlarge = A[t];
                B[tm] = crt_halfreal_center_one_31(zsmall, f31_conj_fast(zlarge), twf31, twi31, pow2_n, km, flags);
                A[t]  = crt_halfreal_center_one_31(zlarge, f31_conj_fast(zsmall), twf31, twi31, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(A, twi31, 512u, L, lid, 64u);
        if (!same) crt_local_stage_dit_radix8_31(B, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 512u; t += 64u) {
        a31[baseA + t] = A[t];
        if (!same) a31[baseB + t] = B[t];
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_61(__global GF* restrict a61,
                                                 __global const GF* restrict twf61,
                                                 __global const GF* restrict twi61,
                                                 uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const int same = (blockA == blockB);

    __local GF X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);    if (same) {
        if ((flags & 32u) != 0u) {
            for (uint r = 0u; r < 8u; ++r) {
                const uint t = (lid << 3) + r;
                const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (511u - t)
                                           : crt_bitrev9_u32_fast((0u - rt) & 511u);
                if (k > km) continue;
                const GF z = X[t];
                const GF zn = X[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                const GF outK  = gf_pack_e_plus_i_o(E, O);
                const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
                X[t] = outK;
                if (tm != t) X[tm] = outKm;
            }
        } else {
            for (uint r = 0u; r < 8u; ++r) {
                const uint t = (lid << 3) + r;
                const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (511u - t)
                                           : crt_bitrev9_u32_fast((0u - rt) & 511u);
                if (k > km) continue;
                const GF z = X[t];
                const GF zn = X[tm];
                X[t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                if (km != k) X[tm] = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
            local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
        return;
    }    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    if ((flags & 32u) != 0u) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = 511u - t;
            if (k <= km) {
                const GF z = a61[baseA + t];
                const GF zn = X[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                a61[baseA + t] = gf_pack_e_plus_i_o(E, O);
                X[tm] = gf_pack_conj_e_plus_i_conj_o(E, O);
            } else {
                const GF zsmall = X[tm];
                const GF zlarge = a61[baseA + t];
                GF E, O;
                crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
                X[tm] = gf_pack_e_plus_i_o(E, O);
                a61[baseA + t] = gf_pack_conj_e_plus_i_conj_o(E, O);
            }
        }
    } else {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = 511u - t;
            if (k <= km) {
                const GF z = a61[baseA + t];
                const GF zn = X[tm];
                a61[baseA + t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                X[tm] = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
            } else {
                const GF zsmall = X[tm];
                const GF zlarge = a61[baseA + t];
                X[tm] = crt_halfreal_center_one_61(zsmall, gf_conj_fast(zlarge), twf61, twi61, pow2_n, km, flags);
                a61[baseA + t] = crt_halfreal_center_one_61(zlarge, gf_conj_fast(zsmall), twf61, twi61, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_rega_f48_61(__global GF* restrict a61,
                                                          __global const GF* restrict twf61,
                                                          __global const GF* restrict twi61,
                                                          uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const uint same = (pair_id == 0u) || ((pair_id << 1) == blocks);

    __local GF X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);

    if (same) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);
            if (k > km) continue;
            const GF z = X[t];
            const GF zn = X[tm];
            GF E, O;
            crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
            const GF outK  = gf_pack_e_plus_i_o(E, O);
            const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
            X[t] = outK;
            if (tm != t) X[tm] = outKm;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
            local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
        return;
    }

    const uint t0 = lid << 3;
    GF A0 = X[t0 + 0u];
    GF A1 = X[t0 + 1u];
    GF A2 = X[t0 + 2u];
    GF A3 = X[t0 + 3u];
    GF A4 = X[t0 + 4u];
    GF A5 = X[t0 + 5u];
    GF A6 = X[t0 + 6u];
    GF A7 = X[t0 + 7u];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

#define CRT_MIXED_CENTER_REGA_EVEN_61(R, AR) do { \
        const uint t = t0 + (uint)(R); \
        const uint rt = (crt_bitrev3_u32_fast((uint)(R)) << 6) | rev_lid6; \
        const uint k = (rt << h) | kb; \
        const uint tm = 511u - t; \
        const GF zn = X[tm]; \
        GF E, O; \
        crt_halfreal_center_eo_f48_61((AR), gf_conj_fast(zn), twf61, pow2_n, k, &E, &O); \
        (AR) = gf_pack_e_plus_i_o(E, O); \
        X[tm] = gf_pack_conj_e_plus_i_conj_o(E, O); \
    } while (0)

#define CRT_MIXED_CENTER_REGA_ODD_61(R, AR) do { \
        const uint t = t0 + (uint)(R); \
        const uint rt = (crt_bitrev3_u32_fast((uint)(R)) << 6) | rev_lid6; \
        const uint k = (rt << h) | kb; \
        const uint km = (row_m - k) & (row_m - 1u); \
        const uint tm = 511u - t; \
        const GF zsmall = X[tm]; \
        GF E, O; \
        crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast((AR)), twf61, pow2_n, km, &E, &O); \
        X[tm] = gf_pack_e_plus_i_o(E, O); \
        (AR) = gf_pack_conj_e_plus_i_conj_o(E, O); \
    } while (0)

    CRT_MIXED_CENTER_REGA_EVEN_61(0u, A0);
    CRT_MIXED_CENTER_REGA_ODD_61 (1u, A1);
    CRT_MIXED_CENTER_REGA_EVEN_61(2u, A2);
    CRT_MIXED_CENTER_REGA_ODD_61 (3u, A3);
    CRT_MIXED_CENTER_REGA_EVEN_61(4u, A4);
    CRT_MIXED_CENTER_REGA_ODD_61 (5u, A5);
    CRT_MIXED_CENTER_REGA_EVEN_61(6u, A6);
    CRT_MIXED_CENTER_REGA_ODD_61 (7u, A7);

#undef CRT_MIXED_CENTER_REGA_EVEN_61
#undef CRT_MIXED_CENTER_REGA_ODD_61

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseB + t] = X[t];
    barrier(CLK_LOCAL_MEM_FENCE);

    X[t0 + 0u] = A0;
    X[t0 + 1u] = A1;
    X[t0 + 2u] = A2;
    X[t0 + 3u] = A3;
    X[t0 + 4u] = A4;
    X[t0 + 5u] = A5;
    X[t0 + 6u] = A6;
    X[t0 + 7u] = A7;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_rega_twinline_f48_61(__global GF* restrict a61,
                                                          __global const GF* restrict twf61,
                                                          __global const GF* restrict twi61,
                                                          uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const uint same = (pair_id == 0u) || ((pair_id << 1) == blocks);

    __local GF X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);

    if (same) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);
            if (k > km) continue;
            const GF z = X[t];
            const GF zn = X[tm];
            GF E, O;
            crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
            const GF outK  = gf_pack_e_plus_i_o(E, O);
            const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
            X[t] = outK;
            if (tm != t) X[tm] = outKm;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
            local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
        return;
    }

    const uint t0 = lid << 3;
    GF A0 = X[t0 + 0u];
    GF A1 = X[t0 + 1u];
    GF A2 = X[t0 + 2u];
    GF A3 = X[t0 + 3u];
    GF A4 = X[t0 + 4u];
    GF A5 = X[t0 + 5u];
    GF A6 = X[t0 + 6u];
    GF A7 = X[t0 + 7u];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

#define CRT_MIXED_CENTER_REGA_EVEN_61(R, HI, AR) do { \
        const uint t = t0 + (uint)(R); \
        const uint rt = ((uint)(HI) << 6) | rev_lid6; \
        const uint k = (rt << h) | kb; \
        const uint tm = 511u - t; \
        const GF zn = X[tm]; \
        GF out0, out1; \
        crt_halfreal_center_pair_f48_w_61((AR), gf_conj_fast(zn), twf61[((pow2_n >> 1) - 1u) + k], &out0, &out1); \
        (AR) = out0; \
        X[tm] = out1; \
    } while (0)

#define CRT_MIXED_CENTER_REGA_ODD_61(R, HI, AR) do { \
        const uint t = t0 + (uint)(R); \
        const uint rt = ((uint)(HI) << 6) | rev_lid6; \
        const uint k = (rt << h) | kb; \
        const uint tm = 511u - t; \
        const GF zsmall = X[tm]; \
        GF out0, out1; \
        crt_halfreal_center_pair_f48_w_61(zsmall, gf_conj_fast((AR)), gf_neg_conj_fast(twf61[((pow2_n >> 1) - 1u) + k]), &out0, &out1); \
        X[tm] = out0; \
        (AR) = out1; \
    } while (0)

    CRT_MIXED_CENTER_REGA_EVEN_61(0u, 0u, A0);
    CRT_MIXED_CENTER_REGA_ODD_61 (1u, 4u, A1);
    CRT_MIXED_CENTER_REGA_EVEN_61(2u, 2u, A2);
    CRT_MIXED_CENTER_REGA_ODD_61 (3u, 6u, A3);
    CRT_MIXED_CENTER_REGA_EVEN_61(4u, 1u, A4);
    CRT_MIXED_CENTER_REGA_ODD_61 (5u, 5u, A5);
    CRT_MIXED_CENTER_REGA_EVEN_61(6u, 3u, A6);
    CRT_MIXED_CENTER_REGA_ODD_61 (7u, 7u, A7);

#undef CRT_MIXED_CENTER_REGA_EVEN_61
#undef CRT_MIXED_CENTER_REGA_ODD_61

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseB + t] = X[t];
    barrier(CLK_LOCAL_MEM_FENCE);

    X[t0 + 0u] = A0;
    X[t0 + 1u] = A1;
    X[t0 + 2u] = A2;
    X[t0 + 3u] = A3;
    X[t0 + 4u] = A4;
    X[t0 + 5u] = A5;
    X[t0 + 6u] = A6;
    X[t0 + 7u] = A7;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
}


/* v42: GF31-specific regA/twinline LDS512 center.
   Same scheduling idea as the fast GF61 center: keep block A in registers while
   block B is transformed through one LDS buffer, then run the two inverse sides. */
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_rega_twinline_f48_31(__global GF31* restrict a31,
                                                          __global const GF31* restrict twf31,
                                                          __global const GF31* restrict twi31,
                                                          uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const uint same = (pair_id == 0u) || ((pair_id << 1) == blocks);

    __local GF31 X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);

    if (same) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);
            if (k > km) continue;
            const GF31 z = X[t];
            const GF31 zn = X[tm];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
            const GF31 outK  = f31_pack_e_plus_i_o(E, O);
            const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
            X[t] = outK;
            if (tm != t) X[tm] = outKm;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
            crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
        return;
    }

    const uint t0 = lid << 3;
    GF31 A0 = X[t0 + 0u];
    GF31 A1 = X[t0 + 1u];
    GF31 A2 = X[t0 + 2u];
    GF31 A3 = X[t0 + 3u];
    GF31 A4 = X[t0 + 4u];
    GF31 A5 = X[t0 + 5u];
    GF31 A6 = X[t0 + 6u];
    GF31 A7 = X[t0 + 7u];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

#define CRT_MIXED_CENTER_REGA_EVEN_31(R, HI, AR) do { \
        const uint t = t0 + (uint)(R); \
        const uint rt = ((uint)(HI) << 6) | rev_lid6; \
        const uint k = (rt << h) | kb; \
        const uint tm = 511u - t; \
        const GF31 zn = X[tm]; \
        GF31 out0, out1; \
        crt_halfreal_center_pair_f48_w_31((AR), f31_conj_fast(zn), twf31[((pow2_n >> 1) - 1u) + k], &out0, &out1); \
        (AR) = out0; \
        X[tm] = out1; \
    } while (0)

#define CRT_MIXED_CENTER_REGA_ODD_31(R, HI, AR) do { \
        const uint t = t0 + (uint)(R); \
        const uint rt = ((uint)(HI) << 6) | rev_lid6; \
        const uint k = (rt << h) | kb; \
        const uint tm = 511u - t; \
        const GF31 zsmall = X[tm]; \
        GF31 out0, out1; \
        crt_halfreal_center_pair_f48_w_31(zsmall, f31_conj_fast((AR)), f31_neg_conj_fast(twf31[((pow2_n >> 1) - 1u) + k]), &out0, &out1); \
        X[tm] = out0; \
        (AR) = out1; \
    } while (0)

    CRT_MIXED_CENTER_REGA_EVEN_31(0u, 0u, A0);
    CRT_MIXED_CENTER_REGA_ODD_31 (1u, 4u, A1);
    CRT_MIXED_CENTER_REGA_EVEN_31(2u, 2u, A2);
    CRT_MIXED_CENTER_REGA_ODD_31 (3u, 6u, A3);
    CRT_MIXED_CENTER_REGA_EVEN_31(4u, 1u, A4);
    CRT_MIXED_CENTER_REGA_ODD_31 (5u, 5u, A5);
    CRT_MIXED_CENTER_REGA_EVEN_31(6u, 3u, A6);
    CRT_MIXED_CENTER_REGA_ODD_31 (7u, 7u, A7);

#undef CRT_MIXED_CENTER_REGA_EVEN_31
#undef CRT_MIXED_CENTER_REGA_ODD_31

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseB + t] = X[t];
    barrier(CLK_LOCAL_MEM_FENCE);

    X[t0 + 0u] = A0;
    X[t0 + 1u] = A1;
    X[t0 + 2u] = A2;
    X[t0 + 3u] = A3;
    X[t0 + 4u] = A4;
    X[t0 + 5u] = A5;
    X[t0 + 6u] = A6;
    X[t0 + 7u] = A7;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_31(__global GF31* restrict a31,
                                                 __global const GF31* restrict twf31,
                                                 __global const GF31* restrict twi31,
                                                 uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const int same = (blockA == blockB);

    __local GF31 X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);

    if (same) {
        if ((flags & 32u) != 0u) {
            for (uint r = 0u; r < 8u; ++r) {
                const uint t = (lid << 3) + r;
                const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (511u - t)
                                           : crt_bitrev9_u32_fast((0u - rt) & 511u);
                if (k > km) continue;
                const GF31 z = X[t];
                const GF31 zn = X[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                const GF31 outK  = f31_pack_e_plus_i_o(E, O);
                const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
                X[t] = outK;
                if (tm != t) X[tm] = outKm;
            }
        } else {
            for (uint r = 0u; r < 8u; ++r) {
                const uint t = (lid << 3) + r;
                const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (511u - t)
                                           : crt_bitrev9_u32_fast((0u - rt) & 511u);
                if (k > km) continue;
                const GF31 z = X[t];
                const GF31 zn = X[tm];
                X[t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                if (km != k) X[tm] = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
            crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
        return;
    }

    for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    if ((flags & 32u) != 0u) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = 511u - t;
            if (k <= km) {
                const GF31 z = a31[baseA + t];
                const GF31 zn = X[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                a31[baseA + t] = f31_pack_e_plus_i_o(E, O);
                X[tm] = f31_pack_conj_e_plus_i_conj_o(E, O);
            } else {
                const GF31 zsmall = X[tm];
                const GF31 zlarge = a31[baseA + t];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
                X[tm] = f31_pack_e_plus_i_o(E, O);
                a31[baseA + t] = f31_pack_conj_e_plus_i_conj_o(E, O);
            }
        }
    } else {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = 511u - t;
            if (k <= km) {
                const GF31 z = a31[baseA + t];
                const GF31 zn = X[tm];
                a31[baseA + t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                X[tm] = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
            } else {
                const GF31 zsmall = X[tm];
                const GF31 zlarge = a31[baseA + t];
                X[tm] = crt_halfreal_center_one_31(zsmall, f31_conj_fast(zlarge), twf31, twi31, pow2_n, km, flags);
                a31[baseA + t] = crt_halfreal_center_one_31(zlarge, f31_conj_fast(zsmall), twf31, twi31, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_self_61(__global GF* restrict a61,
                                                          __global const GF* restrict twf61,
                                                          __global const GF* restrict twi61,
                                                          uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid >> 1;
    const uint self_id = gid & 1u;
    if (row >= odd) return;
    const uint pair_id = (self_id == 0u) ? 0u : (blocks >> 1u);
    if (pair_id >= pair_blocks || (self_id != 0u && pair_id == 0u)) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);

    __local GF X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = (kb != 0u) ? (511u - t)
                                   : crt_bitrev9_u32_fast((0u - rt) & 511u);
        if (k > km) continue;
        const GF z = X[t];
        const GF zn = X[tm];
        GF E, O;
        crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
        const GF outK = gf_pack_e_plus_i_o(E, O);
        const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
        X[t] = outK;
        if (tm != t) X[tm] = outKm;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_nonself_61(__global GF* restrict a61,
                                                             __global const GF* restrict twf61,
                                                             __global const GF* restrict twi61,
                                                             uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    if (pair_blocks <= 2u) return;
    const uint nonself_blocks = pair_blocks - 2u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / nonself_blocks;
    const uint pair_id = 1u + gid - row * nonself_blocks;
    if (row >= odd || pair_id >= pair_blocks - 1u) return;

    const uint h = log_m - 9u;
    const uint kb = pair_id;
    const uint blockA = crt_bitrev_u32(kb, h);
    const uint blockB = crt_bitrev_u32((0u - kb) & (blocks - 1u), h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);

    __local GF X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(X, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = 511u - t;
        if (k <= km) {
            const GF z = a61[baseA + t];
            const GF zn = X[tm];
            GF E, O;
            crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
            a61[baseA + t] = gf_pack_e_plus_i_o(E, O);
            X[tm] = gf_pack_conj_e_plus_i_conj_o(E, O);
        } else {
            const GF zsmall = X[tm];
            const GF zlarge = a61[baseA + t];
            GF E, O;
            crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
            X[tm] = gf_pack_e_plus_i_o(E, O);
            a61[baseA + t] = gf_pack_conj_e_plus_i_conj_o(E, O);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(X, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_self_31(__global GF31* restrict a31,
                                                          __global const GF31* restrict twf31,
                                                          __global const GF31* restrict twi31,
                                                          uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid >> 1;
    const uint self_id = gid & 1u;
    if (row >= odd) return;
    const uint pair_id = (self_id == 0u) ? 0u : (blocks >> 1u);
    if (pair_id >= pair_blocks || (self_id != 0u && pair_id == 0u)) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);

    __local GF31 X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = (kb != 0u) ? (511u - t)
                                   : crt_bitrev9_u32_fast((0u - rt) & 511u);
        if (k > km) continue;
        const GF31 z = X[t];
        const GF31 zn = X[tm];
        GF31 E, O;
        crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
        const GF31 outK = f31_pack_e_plus_i_o(E, O);
        const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
        X[t] = outK;
        if (tm != t) X[tm] = outKm;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_nonself_31(__global GF31* restrict a31,
                                                             __global const GF31* restrict twf31,
                                                             __global const GF31* restrict twi31,
                                                             uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    if (pair_blocks <= 2u) return;
    const uint nonself_blocks = pair_blocks - 2u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / nonself_blocks;
    const uint pair_id = 1u + gid - row * nonself_blocks;
    if (row >= odd || pair_id >= pair_blocks - 1u) return;

    const uint h = log_m - 9u;
    const uint kb = pair_id;
    const uint blockA = crt_bitrev_u32(kb, h);
    const uint blockB = crt_bitrev_u32((0u - kb) & (blocks - 1u), h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);

    __local GF31 X[512];

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(X, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = 511u - t;
        if (k <= km) {
            const GF31 z = a31[baseA + t];
            const GF31 zn = X[tm];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
            a31[baseA + t] = f31_pack_e_plus_i_o(E, O);
            X[tm] = f31_pack_conj_e_plus_i_conj_o(E, O);
        } else {
            const GF31 zsmall = X[tm];
            const GF31 zlarge = a31[baseA + t];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
            X[tm] = f31_pack_e_plus_i_o(E, O);
            a31[baseA + t] = f31_pack_conj_e_plus_i_conj_o(E, O);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(X, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint t = lid; t < 512u; t += 64u) a31[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds1024_pair_61(__global GF* restrict a61,
                                            __global const GF* restrict twf61,
                                            __global const GF* restrict twi61,
                                            uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 10u) return;

    const uint blocks = row_m >> 10;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 10u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 10);
    const uint baseB = row_base + (blockB << 10);
    const int same = (blockA == blockB);

    __local GF A[1024];
    __local GF B[1024];

    for (uint t = lid; t < 1024u; t += 64u) {
        A[t] = a61[baseA + t];
        if (!same) B[t] = a61[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    uint Lf = 1024u;
    while (Lf >= 8u) {
        local_stage_dif_radix8_pow2(A, twf61, 1024u, Lf, lid, 64u);
        if (!same) local_stage_dif_radix8_pow2(B, twf61, 1024u, Lf, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        Lf >>= 3;
    }
    if (Lf == 2u) {
        local_stage_dif_pow2(A, twf61, 1024u, 2u, lid, 64u);
        if (!same) local_stage_dif_pow2(B, twf61, 1024u, 2u, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    if ((flags & 32u) != 0u) {
        for (uint r = 0u; r < 16u; ++r) {
            const uint t = (lid << 4) + r;
            const uint rt = (crt_bitrev4_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (1023u - t)
                                       : crt_bitrev10_u32_fast((0u - rt) & 1023u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z = A[t];
                const GF zn = same ? A[tm] : B[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                const GF outK  = gf_pack_e_plus_i_o(E, O);
                const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
                A[t] = outK;
                if (same) {
                    if (tm != t) A[tm] = outKm;
                } else {
                    B[tm] = outKm;
                }
            } else {
                const GF zsmall = B[tm];
                const GF zlarge = A[t];
                GF E, O;
                crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
                const GF outSmall = gf_pack_e_plus_i_o(E, O);
                const GF outLarge = gf_pack_conj_e_plus_i_conj_o(E, O);
                B[tm] = outSmall;
                A[t] = outLarge;
            }
        }
    } else {
        for (uint r = 0u; r < 16u; ++r) {
            const uint t = (lid << 4) + r;
            const uint rt = (crt_bitrev4_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (1023u - t)
                                       : crt_bitrev10_u32_fast((0u - rt) & 1023u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z = A[t];
                const GF zn = same ? A[tm] : B[tm];
                A[t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                if (km != k) {
                    const GF outKm = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
                    if (same) A[tm] = outKm;
                    else B[tm] = outKm;
                }
            } else {
                const GF zsmall = B[tm];
                const GF zlarge = A[t];
                B[tm] = crt_halfreal_center_one_61(zsmall, gf_conj_fast(zlarge), twf61, twi61, pow2_n, km, flags);
                A[t]  = crt_halfreal_center_one_61(zlarge, gf_conj_fast(zsmall), twf61, twi61, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    local_stage_dit_pow2(A, twi61, 1024u, 2u, lid, 64u);
    if (!same) local_stage_dit_pow2(B, twi61, 1024u, 2u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint Li = 4u; Li < 1024u; Li <<= 3) {
        local_stage_dit_radix8_pow2(A, twi61, 1024u, Li, lid, 64u);
        if (!same) local_stage_dit_radix8_pow2(B, twi61, 1024u, Li, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 1024u; t += 64u) {
        a61[baseA + t] = A[t];
        if (!same) a61[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds1024_pair_31(__global GF31* restrict a31,
                                            __global const GF31* restrict twf31,
                                            __global const GF31* restrict twi31,
                                            uint pow2_n, uint odd, uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 10u) return;

    const uint blocks = row_m >> 10;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 10u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 10);
    const uint baseB = row_base + (blockB << 10);
    const int same = (blockA == blockB);

    __local GF31 A[1024];
    __local GF31 B[1024];

    for (uint t = lid; t < 1024u; t += 64u) {
        A[t] = a31[baseA + t];
        if (!same) B[t] = a31[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint Lf = 1024u;
    while (Lf >= 8u) {
        crt_local_stage_dif_radix8_31(A, twf31, 1024u, Lf, lid, 64u);
        if (!same) crt_local_stage_dif_radix8_31(B, twf31, 1024u, Lf, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        Lf >>= 3;
    }
    if (Lf == 2u) {
        crt_local_stage_dif_pow2_31(A, twf31, 1024u, 2u, lid, 64u);
        if (!same) crt_local_stage_dif_pow2_31(B, twf31, 1024u, 2u, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    if ((flags & 32u) != 0u) {
        for (uint r = 0u; r < 16u; ++r) {
            const uint t = (lid << 4) + r;
            const uint rt = (crt_bitrev4_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (1023u - t)
                                       : crt_bitrev10_u32_fast((0u - rt) & 1023u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF31 z = A[t];
                const GF31 zn = same ? A[tm] : B[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                const GF31 outK  = f31_pack_e_plus_i_o(E, O);
                const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
                A[t] = outK;
                if (same) {
                    if (tm != t) A[tm] = outKm;
                } else {
                    B[tm] = outKm;
                }
            } else {
                const GF31 zsmall = B[tm];
                const GF31 zlarge = A[t];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
                const GF31 outSmall = f31_pack_e_plus_i_o(E, O);
                const GF31 outLarge = f31_pack_conj_e_plus_i_conj_o(E, O);
                B[tm] = outSmall;
                A[t] = outLarge;
            }
        }
    } else {
        for (uint r = 0u; r < 16u; ++r) {
            const uint t = (lid << 4) + r;
            const uint rt = (crt_bitrev4_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (1023u - t)
                                       : crt_bitrev10_u32_fast((0u - rt) & 1023u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF31 z = A[t];
                const GF31 zn = same ? A[tm] : B[tm];
                A[t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                if (km != k) {
                    const GF31 outKm = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
                    if (same) A[tm] = outKm;
                    else B[tm] = outKm;
                }
            } else {
                const GF31 zsmall = B[tm];
                const GF31 zlarge = A[t];
                B[tm] = crt_halfreal_center_one_31(zsmall, f31_conj_fast(zlarge), twf31, twi31, pow2_n, km, flags);
                A[t]  = crt_halfreal_center_one_31(zlarge, f31_conj_fast(zsmall), twf31, twi31, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_local_stage_dit_pow2_31(A, twi31, 1024u, 2u, lid, 64u);
    if (!same) crt_local_stage_dit_pow2_31(B, twi31, 1024u, 2u, lid, 64u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint Li = 4u; Li < 1024u; Li <<= 3) {
        crt_local_stage_dit_radix8_31(A, twi31, 1024u, Li, lid, 64u);
        if (!same) crt_local_stage_dit_radix8_31(B, twi31, 1024u, Li, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 1024u; t += 64u) {
        a31[baseA + t] = A[t];
        if (!same) a31[baseB + t] = B[t];
    }
}



static inline __attribute__((always_inline)) void crt_mixed_local_dif_pow2_any_61(__local GF* x,
                                                                                 __global const GF* tw,
                                                                                 uint center_len,
                                                                                 uint lane) {
    uint L = center_len;
    while (L >= 8u) {
        local_stage_dif_radix8_pow2(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) { L = 1u; break; }
        L >>= 3;
    }
    if (L == 4u) {
        local_stage_dif_radix4_pow2(x, tw, center_len, 4u, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        local_stage_dif_pow2(x, tw, center_len, 2u, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

static inline __attribute__((always_inline)) void crt_mixed_local_dit_pow2_any_61(__local GF* x,
                                                                                 __global const GF* tw,
                                                                                 uint center_len,
                                                                                 uint lane) {
    uint tail = center_len;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        local_stage_dit_pow2(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        local_stage_dit_radix4_pow2(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < center_len) {
        local_stage_dit_radix8_pow2(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }
}

static inline __attribute__((always_inline)) void crt_mixed_local_dif_pow2_any_31(__local GF31* x,
                                                                                 __global const GF31* tw,
                                                                                 uint center_len,
                                                                                 uint lane) {
    uint L = center_len;
    while (L >= 8u) {
        crt_local_stage_dif_radix8_31(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) { L = 1u; break; }
        L >>= 3;
    }
    if (L == 4u) {
        crt_local_stage_dif_radix4_31(x, tw, center_len, 4u, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (L == 2u) {
        crt_local_stage_dif_pow2_31(x, tw, center_len, 2u, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

static inline __attribute__((always_inline)) void crt_mixed_local_dit_pow2_any_31(__local GF31* x,
                                                                                 __global const GF31* tw,
                                                                                 uint center_len,
                                                                                 uint lane) {
    uint tail = center_len;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) {
        crt_local_stage_dit_pow2_31(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 1;
    } else if (tail == 4u) {
        crt_local_stage_dit_radix4_31(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 2;
    }
    while (L < center_len) {
        crt_local_stage_dit_radix8_31(x, tw, center_len, L, lane, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        L <<= 3;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_61(__global GF* restrict a61,
                                             __global const GF* restrict twf61,
                                             __global const GF* restrict twi61,
                                             uint pow2_n, uint odd, uint flags,
                                             uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - log_c;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const int same = (blockA == blockB);

    __local GF A[1024];
    __local GF B[1024];

    for (uint t = lid; t < center_len; t += 64u) {
        A[t] = a61[baseA + t];
        if (!same) B[t] = a61[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(A, twf61, center_len, lid);
    if (!same) crt_mixed_local_dif_pow2_any_61(B, twf61, center_len, lid);

    const uint c_mask = center_len - 1u;
    if ((flags & 32u) != 0u) {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (c_mask - t)
                                       : crt_bitrev_u32((0u - rt) & c_mask, log_c);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z = A[t];
                const GF zn = same ? A[tm] : B[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                const GF outK  = gf_pack_e_plus_i_o(E, O);
                const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
                A[t] = outK;
                if (same) {
                    if (tm != t) A[tm] = outKm;
                } else {
                    B[tm] = outKm;
                }
            } else {
                const GF zsmall = B[tm];
                const GF zlarge = A[t];
                GF E, O;
                crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
                const GF outSmall = gf_pack_e_plus_i_o(E, O);
                const GF outLarge = gf_pack_conj_e_plus_i_conj_o(E, O);
                B[tm] = outSmall;
                A[t] = outLarge;
            }
        }
    } else {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (c_mask - t)
                                       : crt_bitrev_u32((0u - rt) & c_mask, log_c);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z = A[t];
                const GF zn = same ? A[tm] : B[tm];
                A[t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                if (km != k) {
                    const GF outKm = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
                    if (same) A[tm] = outKm;
                    else B[tm] = outKm;
                }
            } else {
                const GF zsmall = B[tm];
                const GF zlarge = A[t];
                B[tm] = crt_halfreal_center_one_61(zsmall, gf_conj_fast(zlarge), twf61, twi61, pow2_n, km, flags);
                A[t]  = crt_halfreal_center_one_61(zlarge, gf_conj_fast(zsmall), twf61, twi61, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_61(A, twi61, center_len, lid);
    if (!same) crt_mixed_local_dit_pow2_any_61(B, twi61, center_len, lid);

    for (uint t = lid; t < center_len; t += 64u) {
        a61[baseA + t] = A[t];
        if (!same) a61[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_31(__global GF31* restrict a31,
                                             __global const GF31* restrict twf31,
                                             __global const GF31* restrict twi31,
                                             uint pow2_n, uint odd, uint flags,
                                             uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - log_c;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const int same = (blockA == blockB);

    __local GF31 A[1024];
    __local GF31 B[1024];

    for (uint t = lid; t < center_len; t += 64u) {
        A[t] = a31[baseA + t];
        if (!same) B[t] = a31[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_31(A, twf31, center_len, lid);
    if (!same) crt_mixed_local_dif_pow2_any_31(B, twf31, center_len, lid);

    const uint c_mask = center_len - 1u;
    if ((flags & 32u) != 0u) {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (c_mask - t)
                                       : crt_bitrev_u32((0u - rt) & c_mask, log_c);

            if (same && k > km) continue;

            if (k <= km) {
                const GF31 z = A[t];
                const GF31 zn = same ? A[tm] : B[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                const GF31 outK  = f31_pack_e_plus_i_o(E, O);
                const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
                A[t] = outK;
                if (same) {
                    if (tm != t) A[tm] = outKm;
                } else {
                    B[tm] = outKm;
                }
            } else {
                const GF31 zsmall = B[tm];
                const GF31 zlarge = A[t];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
                const GF31 outSmall = f31_pack_e_plus_i_o(E, O);
                const GF31 outLarge = f31_pack_conj_e_plus_i_conj_o(E, O);
                B[tm] = outSmall;
                A[t] = outLarge;
            }
        }
    } else {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (c_mask - t)
                                       : crt_bitrev_u32((0u - rt) & c_mask, log_c);

            if (same && k > km) continue;

            if (k <= km) {
                const GF31 z = A[t];
                const GF31 zn = same ? A[tm] : B[tm];
                A[t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                if (km != k) {
                    const GF31 outKm = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
                    if (same) A[tm] = outKm;
                    else B[tm] = outKm;
                }
            } else {
                const GF31 zsmall = B[tm];
                const GF31 zlarge = A[t];
                B[tm] = crt_halfreal_center_one_31(zsmall, f31_conj_fast(zlarge), twf31, twi31, pow2_n, km, flags);
                A[t]  = crt_halfreal_center_one_31(zlarge, f31_conj_fast(zsmall), twf31, twi31, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_31(A, twi31, center_len, lid);
    if (!same) crt_mixed_local_dit_pow2_any_31(B, twi31, center_len, lid);

    for (uint t = lid; t < center_len; t += 64u) {
        a31[baseA + t] = A[t];
        if (!same) a31[baseB + t] = B[t];
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_1lds_61(__global GF* restrict a61,
                                                  __global const GF* restrict twf61,
                                                  __global const GF* restrict twi61,
                                                  uint pow2_n, uint odd, uint flags,
                                                  uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - log_c;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const int same = (blockA == blockB);

    __local GF X[1024];

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(X, twf61, center_len, lid);

    const uint c_mask = center_len - 1u;

    if (same) {
        if ((flags & 32u) != 0u) {
            for (uint t = lid; t < center_len; t += 64u) {
                const uint rt = crt_bitrev_u32(t, log_c);
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (c_mask - t)
                                           : crt_bitrev_u32((0u - rt) & c_mask, log_c);
                if (k > km) continue;
                const GF z = X[t];
                const GF zn = X[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                const GF outK  = gf_pack_e_plus_i_o(E, O);
                const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
                X[t] = outK;
                if (tm != t) X[tm] = outKm;
            }
        } else {
            for (uint t = lid; t < center_len; t += 64u) {
                const uint rt = crt_bitrev_u32(t, log_c);
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (c_mask - t)
                                           : crt_bitrev_u32((0u - rt) & c_mask, log_c);
                if (k > km) continue;
                const GF z = X[t];
                const GF zn = X[tm];
                X[t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                if (km != k) X[tm] = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        crt_mixed_local_dit_pow2_any_61(X, twi61, center_len, lid);
        for (uint t = lid; t < center_len; t += 64u) a61[baseA + t] = X[t];
        return;
    }    for (uint t = lid; t < center_len; t += 64u) a61[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(X, twf61, center_len, lid);

    if ((flags & 32u) != 0u) {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = c_mask - t;
            if (k <= km) {
                const GF z = a61[baseA + t];
                const GF zn = X[tm];
                GF E, O;
                crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
                a61[baseA + t] = gf_pack_e_plus_i_o(E, O);
                X[tm] = gf_pack_conj_e_plus_i_conj_o(E, O);
            } else {
                const GF zsmall = X[tm];
                const GF zlarge = a61[baseA + t];
                GF E, O;
                crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
                X[tm] = gf_pack_e_plus_i_o(E, O);
                a61[baseA + t] = gf_pack_conj_e_plus_i_conj_o(E, O);
            }
        }
    } else {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = c_mask - t;
            if (k <= km) {
                const GF z = a61[baseA + t];
                const GF zn = X[tm];
                a61[baseA + t] = crt_halfreal_center_one_61(z, gf_conj_fast(zn), twf61, twi61, pow2_n, k, flags);
                X[tm] = crt_halfreal_center_one_61(zn, gf_conj_fast(z), twf61, twi61, pow2_n, km, flags);
            } else {
                const GF zsmall = X[tm];
                const GF zlarge = a61[baseA + t];
                X[tm] = crt_halfreal_center_one_61(zsmall, gf_conj_fast(zlarge), twf61, twi61, pow2_n, km, flags);
                a61[baseA + t] = crt_halfreal_center_one_61(zlarge, gf_conj_fast(zsmall), twf61, twi61, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_61(X, twi61, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a61[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_mixed_local_dit_pow2_any_61(X, twi61, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_1lds_31(__global GF31* restrict a31,
                                                  __global const GF31* restrict twf31,
                                                  __global const GF31* restrict twi31,
                                                  uint pow2_n, uint odd, uint flags,
                                                  uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - log_c;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const int same = (blockA == blockB);

    __local GF31 X[1024];

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_31(X, twf31, center_len, lid);

    const uint c_mask = center_len - 1u;

    if (same) {
        if ((flags & 32u) != 0u) {
            for (uint t = lid; t < center_len; t += 64u) {
                const uint rt = crt_bitrev_u32(t, log_c);
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (c_mask - t)
                                           : crt_bitrev_u32((0u - rt) & c_mask, log_c);
                if (k > km) continue;
                const GF31 z = X[t];
                const GF31 zn = X[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                const GF31 outK  = f31_pack_e_plus_i_o(E, O);
                const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
                X[t] = outK;
                if (tm != t) X[tm] = outKm;
            }
        } else {
            for (uint t = lid; t < center_len; t += 64u) {
                const uint rt = crt_bitrev_u32(t, log_c);
                const uint k = (rt << h) | kb;
                const uint km = (row_m - k) & (row_m - 1u);
                const uint tm = (kb != 0u) ? (c_mask - t)
                                           : crt_bitrev_u32((0u - rt) & c_mask, log_c);
                if (k > km) continue;
                const GF31 z = X[t];
                const GF31 zn = X[tm];
                X[t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                if (km != k) X[tm] = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        crt_mixed_local_dit_pow2_any_31(X, twi31, center_len, lid);
        for (uint t = lid; t < center_len; t += 64u) a31[baseA + t] = X[t];
        return;
    }

    for (uint t = lid; t < center_len; t += 64u) a31[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_31(X, twf31, center_len, lid);

    if ((flags & 32u) != 0u) {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = c_mask - t;
            if (k <= km) {
                const GF31 z = a31[baseA + t];
                const GF31 zn = X[tm];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
                a31[baseA + t] = f31_pack_e_plus_i_o(E, O);
                X[tm] = f31_pack_conj_e_plus_i_conj_o(E, O);
            } else {
                const GF31 zsmall = X[tm];
                const GF31 zlarge = a31[baseA + t];
                GF31 E, O;
                crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
                X[tm] = f31_pack_e_plus_i_o(E, O);
                a31[baseA + t] = f31_pack_conj_e_plus_i_conj_o(E, O);
            }
        }
    } else {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = c_mask - t;
            if (k <= km) {
                const GF31 z = a31[baseA + t];
                const GF31 zn = X[tm];
                a31[baseA + t] = crt_halfreal_center_one_31(z, f31_conj_fast(zn), twf31, twi31, pow2_n, k, flags);
                X[tm] = crt_halfreal_center_one_31(zn, f31_conj_fast(z), twf31, twi31, pow2_n, km, flags);
            } else {
                const GF31 zsmall = X[tm];
                const GF31 zlarge = a31[baseA + t];
                X[tm] = crt_halfreal_center_one_31(zsmall, f31_conj_fast(zlarge), twf31, twi31, pow2_n, km, flags);
                a31[baseA + t] = crt_halfreal_center_one_31(zlarge, f31_conj_fast(zsmall), twf31, twi31, pow2_n, k, flags);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_31(X, twi31, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a31[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_mixed_local_dit_pow2_any_31(X, twi31, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a31[baseA + t] = X[t];
}



__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_self_61(__global GF* restrict a61,
                                                           __global const GF* restrict twf61,
                                                           __global const GF* restrict twi61,
                                                           uint pow2_n, uint odd, uint flags,
                                                           uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint self_pairs = (blocks > 1u) ? 2u : 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / self_pairs;
    const uint self_id = gid - row * self_pairs;
    if (row >= odd || self_id >= self_pairs) return;

    const uint h = log_m - log_c;
    const uint pair_id = (self_id == 0u) ? 0u : (blocks >> 1u);
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint c_mask = center_len - 1u;

    __local GF X[1024];

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(X, twf61, center_len, lid);

    for (uint t = lid; t < center_len; t += 64u) {
        const uint rt = crt_bitrev_u32(t, log_c);
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = (kb != 0u) ? (c_mask - t)
                                   : crt_bitrev_u32((0u - rt) & c_mask, log_c);
        if (k > km) continue;
        const GF z = X[t];
        const GF zn = X[tm];
        GF E, O;
        crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
        const GF outK = gf_pack_e_plus_i_o(E, O);
        const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
        X[t] = outK;
        if (tm != t) X[tm] = outKm;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_61(X, twi61, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_nonself_61(__global GF* restrict a61,
                                                              __global const GF* restrict twf61,
                                                              __global const GF* restrict twi61,
                                                              uint pow2_n, uint odd, uint flags,
                                                              uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    if (pair_blocks <= 2u) return;
    const uint nonself_blocks = pair_blocks - 2u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / nonself_blocks;
    const uint pair_id = 1u + gid - row * nonself_blocks;
    if (row >= odd || pair_id >= pair_blocks - 1u) return;

    const uint h = log_m - log_c;
    const uint kb = pair_id;
    const uint blockA = crt_bitrev_u32(kb, h);
    const uint blockB = crt_bitrev_u32((0u - kb) & (blocks - 1u), h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const uint c_mask = center_len - 1u;

    __local GF X[1024];

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(X, twf61, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a61[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(X, twf61, center_len, lid);

    for (uint t = lid; t < center_len; t += 64u) {
        const uint rt = crt_bitrev_u32(t, log_c);
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = c_mask - t;
        if (k <= km) {
            const GF z = a61[baseA + t];
            const GF zn = X[tm];
            GF E, O;
            crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, pow2_n, k, &E, &O);
            a61[baseA + t] = gf_pack_e_plus_i_o(E, O);
            X[tm] = gf_pack_conj_e_plus_i_conj_o(E, O);
        } else {
            const GF zsmall = X[tm];
            const GF zlarge = a61[baseA + t];
            GF E, O;
            crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, pow2_n, km, &E, &O);
            X[tm] = gf_pack_e_plus_i_o(E, O);
            a61[baseA + t] = gf_pack_conj_e_plus_i_conj_o(E, O);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_61(X, twi61, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a61[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a61[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_mixed_local_dit_pow2_any_61(X, twi61, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a61[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_self_31(__global GF31* restrict a31,
                                                           __global const GF31* restrict twf31,
                                                           __global const GF31* restrict twi31,
                                                           uint pow2_n, uint odd, uint flags,
                                                           uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint self_pairs = (blocks > 1u) ? 2u : 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / self_pairs;
    const uint self_id = gid - row * self_pairs;
    if (row >= odd || self_id >= self_pairs) return;

    const uint h = log_m - log_c;
    const uint pair_id = (self_id == 0u) ? 0u : (blocks >> 1u);
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint c_mask = center_len - 1u;

    __local GF31 X[1024];

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_31(X, twf31, center_len, lid);

    for (uint t = lid; t < center_len; t += 64u) {
        const uint rt = crt_bitrev_u32(t, log_c);
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = (kb != 0u) ? (c_mask - t)
                                   : crt_bitrev_u32((0u - rt) & c_mask, log_c);
        if (k > km) continue;
        const GF31 z = X[t];
        const GF31 zn = X[tm];
        GF31 E, O;
        crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
        const GF31 outK = f31_pack_e_plus_i_o(E, O);
        const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
        X[t] = outK;
        if (tm != t) X[tm] = outKm;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_31(X, twi31, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a31[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_nonself_31(__global GF31* restrict a31,
                                                              __global const GF31* restrict twf31,
                                                              __global const GF31* restrict twi31,
                                                              uint pow2_n, uint odd, uint flags,
                                                              uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 1024u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    if (pair_blocks <= 2u) return;
    const uint nonself_blocks = pair_blocks - 2u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / nonself_blocks;
    const uint pair_id = 1u + gid - row * nonself_blocks;
    if (row >= odd || pair_id >= pair_blocks - 1u) return;

    const uint h = log_m - log_c;
    const uint kb = pair_id;
    const uint blockA = crt_bitrev_u32(kb, h);
    const uint blockB = crt_bitrev_u32((0u - kb) & (blocks - 1u), h);
    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const uint c_mask = center_len - 1u;

    __local GF31 X[1024];

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_31(X, twf31, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a31[baseA + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseB + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_31(X, twf31, center_len, lid);

    for (uint t = lid; t < center_len; t += 64u) {
        const uint rt = crt_bitrev_u32(t, log_c);
        const uint k = (rt << h) | kb;
        const uint km = (row_m - k) & (row_m - 1u);
        const uint tm = c_mask - t;
        if (k <= km) {
            const GF31 z = a31[baseA + t];
            const GF31 zn = X[tm];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, pow2_n, k, &E, &O);
            a31[baseA + t] = f31_pack_e_plus_i_o(E, O);
            X[tm] = f31_pack_conj_e_plus_i_conj_o(E, O);
        } else {
            const GF31 zsmall = X[tm];
            const GF31 zlarge = a31[baseA + t];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, pow2_n, km, &E, &O);
            X[tm] = f31_pack_e_plus_i_o(E, O);
            a31[baseA + t] = f31_pack_conj_e_plus_i_conj_o(E, O);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_31(X, twi31, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a31[baseB + t] = X[t];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint t = lid; t < center_len; t += 64u) X[t] = a31[baseA + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_mixed_local_dit_pow2_any_31(X, twi31, center_len, lid);
    for (uint t = lid; t < center_len; t += 64u) a31[baseA + t] = X[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds_pair_any_61x31(__global GF* restrict a61,
                                                __global GF31* restrict a31,
                                                __global const GF* restrict twf61,
                                                __global const GF* restrict twi61,
                                                __global const GF31* restrict twf31,
                                                __global const GF31* restrict twi31,
                                                uint pow2_n, uint odd,
                                                uint flags61, uint flags31,
                                                uint center_len)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;    if (!gf61_crt_valid_stage_radix(center_len) || center_len < 8u || center_len > 512u || row_m < center_len) return;

    const uint log_m = 31u - (uint)clz(row_m);
    const uint log_c = 31u - (uint)clz(center_len);
    if (log_m < log_c) return;

    const uint blocks = row_m >> log_c;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - log_c;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << log_c);
    const uint baseB = row_base + (blockB << log_c);
    const int same = (blockA == blockB);

    __local GF A61[512];
    __local GF B61[512];
    __local GF31 A31[512];
    __local GF31 B31[512];

    for (uint t = lid; t < center_len; t += 64u) {
        A61[t] = a61[baseA + t];
        A31[t] = a31[baseA + t];
        if (!same) {
            B61[t] = a61[baseB + t];
            B31[t] = a31[baseB + t];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dif_pow2_any_61(A61, twf61, center_len, lid);
    crt_mixed_local_dif_pow2_any_31(A31, twf31, center_len, lid);
    if (!same) {
        crt_mixed_local_dif_pow2_any_61(B61, twf61, center_len, lid);
        crt_mixed_local_dif_pow2_any_31(B31, twf31, center_len, lid);
    }

    const uint c_mask = center_len - 1u;
    if (((flags61 & 32u) != 0u) && ((flags31 & 32u) != 0u)) {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (c_mask - t)
                                       : crt_bitrev_u32((0u - rt) & c_mask, log_c);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z61 = A61[t];
                const GF zn61 = same ? A61[tm] : B61[tm];
                const GF31 z31 = A31[t];
                const GF31 zn31 = same ? A31[tm] : B31[tm];
                GF E61, O61;
                GF31 E31, O31;
                crt_halfreal_center_eo_f48_61(z61, gf_conj_fast(zn61), twf61, pow2_n, k, &E61, &O61);
                crt_halfreal_center_eo_f48_31(z31, f31_conj_fast(zn31), twf31, pow2_n, k, &E31, &O31);
                const GF outK61  = gf_pack_e_plus_i_o(E61, O61);
                const GF outKm61 = gf_pack_conj_e_plus_i_conj_o(E61, O61);
                const GF31 outK31  = f31_pack_e_plus_i_o(E31, O31);
                const GF31 outKm31 = f31_pack_conj_e_plus_i_conj_o(E31, O31);
                A61[t] = outK61;
                A31[t] = outK31;
                if (same) {
                    if (tm != t) { A61[tm] = outKm61; A31[tm] = outKm31; }
                } else {
                    B61[tm] = outKm61;
                    B31[tm] = outKm31;
                }
            } else {
                const GF zsmall61 = B61[tm];
                const GF zlarge61 = A61[t];
                const GF31 zsmall31 = B31[tm];
                const GF31 zlarge31 = A31[t];
                GF E61, O61;
                GF31 E31, O31;
                crt_halfreal_center_eo_f48_61(zsmall61, gf_conj_fast(zlarge61), twf61, pow2_n, km, &E61, &O61);
                crt_halfreal_center_eo_f48_31(zsmall31, f31_conj_fast(zlarge31), twf31, pow2_n, km, &E31, &O31);
                B61[tm] = gf_pack_e_plus_i_o(E61, O61);
                A61[t] = gf_pack_conj_e_plus_i_conj_o(E61, O61);
                B31[tm] = f31_pack_e_plus_i_o(E31, O31);
                A31[t] = f31_pack_conj_e_plus_i_conj_o(E31, O31);
            }
        }
    } else {
        for (uint t = lid; t < center_len; t += 64u) {
            const uint rt = crt_bitrev_u32(t, log_c);
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (c_mask - t)
                                       : crt_bitrev_u32((0u - rt) & c_mask, log_c);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z61 = A61[t];
                const GF zn61 = same ? A61[tm] : B61[tm];
                const GF31 z31 = A31[t];
                const GF31 zn31 = same ? A31[tm] : B31[tm];
                A61[t] = crt_halfreal_center_one_61(z61, gf_conj_fast(zn61), twf61, twi61, pow2_n, k, flags61);
                A31[t] = crt_halfreal_center_one_31(z31, f31_conj_fast(zn31), twf31, twi31, pow2_n, k, flags31);
                if (km != k) {
                    const GF outKm61 = crt_halfreal_center_one_61(zn61, gf_conj_fast(z61), twf61, twi61, pow2_n, km, flags61);
                    const GF31 outKm31 = crt_halfreal_center_one_31(zn31, f31_conj_fast(z31), twf31, twi31, pow2_n, km, flags31);
                    if (same) { A61[tm] = outKm61; A31[tm] = outKm31; }
                    else { B61[tm] = outKm61; B31[tm] = outKm31; }
                }
            } else {
                const GF zsmall61 = B61[tm];
                const GF zlarge61 = A61[t];
                const GF31 zsmall31 = B31[tm];
                const GF31 zlarge31 = A31[t];
                B61[tm] = crt_halfreal_center_one_61(zsmall61, gf_conj_fast(zlarge61), twf61, twi61, pow2_n, km, flags61);
                A61[t]  = crt_halfreal_center_one_61(zlarge61, gf_conj_fast(zsmall61), twf61, twi61, pow2_n, k, flags61);
                B31[tm] = crt_halfreal_center_one_31(zsmall31, f31_conj_fast(zlarge31), twf31, twi31, pow2_n, km, flags31);
                A31[t]  = crt_halfreal_center_one_31(zlarge31, f31_conj_fast(zsmall31), twf31, twi31, pow2_n, k, flags31);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_mixed_local_dit_pow2_any_61(A61, twi61, center_len, lid);
    crt_mixed_local_dit_pow2_any_31(A31, twi31, center_len, lid);
    if (!same) {
        crt_mixed_local_dit_pow2_any_61(B61, twi61, center_len, lid);
        crt_mixed_local_dit_pow2_any_31(B31, twi31, center_len, lid);
    }

    for (uint t = lid; t < center_len; t += 64u) {
        a61[baseA + t] = A61[t];
        a31[baseA + t] = A31[t];
        if (!same) {
            a61[baseB + t] = B61[t];
            a31[baseB + t] = B31[t];
        }
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_halfreal_lds512_pair_61x31(__global GF* restrict a61,
                                               __global GF31* restrict a31,
                                               __global const GF* restrict twf61,
                                               __global const GF* restrict twi61,
                                               __global const GF31* restrict twf31,
                                               __global const GF31* restrict twi31,
                                               uint pow2_n, uint odd,
                                               uint flags61, uint flags31)
{
    const uint lid = (uint)get_local_id(0);
    const uint row_m = pow2_n >> 1;
    const uint log_m = 31u - (uint)clz(row_m);
    if (log_m < 9u) return;

    const uint blocks = row_m >> 9;
    const uint pair_blocks = (blocks >> 1u) + 1u;
    const uint gid = (uint)get_group_id(0);
    const uint row = gid / pair_blocks;
    const uint pair_id = gid - row * pair_blocks;
    if (row >= odd || pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint row_base = row * row_m;
    const uint baseA = row_base + (blockA << 9);
    const uint baseB = row_base + (blockB << 9);
    const int same = (blockA == blockB);

    __local GF A61[512];
    __local GF B61[512];
    __local GF31 A31[512];
    __local GF31 B31[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A61[t] = a61[baseA + t];
        A31[t] = a31[baseA + t];
        if (!same) {
            B61[t] = a61[baseB + t];
            B31[t] = a31[baseB + t];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(A61, twf61, 512u, L, lid, 64u);
        crt_local_stage_dif_radix8_31(A31, twf31, 512u, L, lid, 64u);
        if (!same) {
            local_stage_dif_radix8_pow2(B61, twf61, 512u, L, lid, 64u);
            crt_local_stage_dif_radix8_31(B31, twf31, 512u, L, lid, 64u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    if (((flags61 & 32u) != 0u) && ((flags31 & 32u) != 0u)) {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z61 = A61[t];
                const GF zn61 = same ? A61[tm] : B61[tm];
                const GF31 z31 = A31[t];
                const GF31 zn31 = same ? A31[tm] : B31[tm];
                GF E61, O61;
                GF31 E31, O31;
                crt_halfreal_center_eo_f48_61(z61, gf_conj_fast(zn61), twf61, pow2_n, k, &E61, &O61);
                crt_halfreal_center_eo_f48_31(z31, f31_conj_fast(zn31), twf31, pow2_n, k, &E31, &O31);
                const GF outK61  = gf_pack_e_plus_i_o(E61, O61);
                const GF outKm61 = gf_pack_conj_e_plus_i_conj_o(E61, O61);
                const GF31 outK31  = f31_pack_e_plus_i_o(E31, O31);
                const GF31 outKm31 = f31_pack_conj_e_plus_i_conj_o(E31, O31);
                A61[t] = outK61;
                A31[t] = outK31;
                if (same) {
                    if (tm != t) { A61[tm] = outKm61; A31[tm] = outKm31; }
                } else {
                    B61[tm] = outKm61;
                    B31[tm] = outKm31;
                }
            } else {
                const GF zsmall61 = B61[tm];
                const GF zlarge61 = A61[t];
                const GF31 zsmall31 = B31[tm];
                const GF31 zlarge31 = A31[t];
                GF E61, O61;
                GF31 E31, O31;
                crt_halfreal_center_eo_f48_61(zsmall61, gf_conj_fast(zlarge61), twf61, pow2_n, km, &E61, &O61);
                crt_halfreal_center_eo_f48_31(zsmall31, f31_conj_fast(zlarge31), twf31, pow2_n, km, &E31, &O31);
                B61[tm] = gf_pack_e_plus_i_o(E61, O61);
                A61[t] = gf_pack_conj_e_plus_i_conj_o(E61, O61);
                B31[tm] = f31_pack_e_plus_i_o(E31, O31);
                A31[t] = f31_pack_conj_e_plus_i_conj_o(E31, O31);
            }
        }
    } else {
        for (uint r = 0u; r < 8u; ++r) {
            const uint t = (lid << 3) + r;
            const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
            const uint k = (rt << h) | kb;
            const uint km = (row_m - k) & (row_m - 1u);
            const uint tm = (kb != 0u) ? (511u - t)
                                       : crt_bitrev9_u32_fast((0u - rt) & 511u);

            if (same && k > km) continue;

            if (k <= km) {
                const GF z61 = A61[t];
                const GF zn61 = same ? A61[tm] : B61[tm];
                const GF31 z31 = A31[t];
                const GF31 zn31 = same ? A31[tm] : B31[tm];
                A61[t] = crt_halfreal_center_one_61(z61, gf_conj_fast(zn61), twf61, twi61, pow2_n, k, flags61);
                A31[t] = crt_halfreal_center_one_31(z31, f31_conj_fast(zn31), twf31, twi31, pow2_n, k, flags31);
                if (km != k) {
                    const GF outKm61 = crt_halfreal_center_one_61(zn61, gf_conj_fast(z61), twf61, twi61, pow2_n, km, flags61);
                    const GF31 outKm31 = crt_halfreal_center_one_31(zn31, f31_conj_fast(z31), twf31, twi31, pow2_n, km, flags31);
                    if (same) { A61[tm] = outKm61; A31[tm] = outKm31; }
                    else { B61[tm] = outKm61; B31[tm] = outKm31; }
                }
            } else {
                const GF zsmall61 = B61[tm];
                const GF zlarge61 = A61[t];
                const GF31 zsmall31 = B31[tm];
                const GF31 zlarge31 = A31[t];
                B61[tm] = crt_halfreal_center_one_61(zsmall61, gf_conj_fast(zlarge61), twf61, twi61, pow2_n, km, flags61);
                A61[t]  = crt_halfreal_center_one_61(zlarge61, gf_conj_fast(zsmall61), twf61, twi61, pow2_n, k, flags61);
                B31[tm] = crt_halfreal_center_one_31(zsmall31, f31_conj_fast(zlarge31), twf31, twi31, pow2_n, km, flags31);
                A31[t]  = crt_halfreal_center_one_31(zlarge31, f31_conj_fast(zsmall31), twf31, twi31, pow2_n, k, flags31);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(A61, twi61, 512u, L, lid, 64u);
        crt_local_stage_dit_radix8_31(A31, twi31, 512u, L, lid, 64u);
        if (!same) {
            local_stage_dit_radix8_pow2(B61, twi61, 512u, L, lid, 64u);
            crt_local_stage_dit_radix8_31(B31, twi31, 512u, L, lid, 64u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 512u; t += 64u) {
        a61[baseA + t] = A61[t];
        a31[baseA + t] = A31[t];
        if (!same) {
            a61[baseB + t] = B61[t];
            a31[baseB + t] = B31[t];
        }
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_unpack_unweight_61(__global const GF* restrict a61,
                                       __global ulong* restrict digits61,
                                       uint n, uint p, uint lr2_61,
                                       uint odd, uint pow2_n, uint log_m)
{
    const uint id = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    const uint storage = odd * row_m;
    if (id >= storage) return;
    const uint row = id / row_m;
    const uint k = id - row * row_m;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(row, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(row, b1, odd, pow2_n);
    const GF z = a61[id];
    const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
    const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
    digits61[j0] = rshift61(norm61(z.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
    digits61[j1] = rshift61(norm61(z.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_mixed_unpack_unweight_31(__global const GF31* restrict a31,
                                       __global uint* restrict digits31,
                                       uint n, uint p, uint lr2_31,
                                       uint odd, uint pow2_n, uint log_m)
{
    const uint id = (uint)get_global_id(0);
    const uint row_m = pow2_n >> 1;
    const uint storage = odd * row_m;
    if (id >= storage) return;
    const uint row = id / row_m;
    const uint k = id - row * row_m;
    const uint b0 = k << 1;
    const uint b1 = b0 + 1u;
    const uint j0 = crt_mixed_j_from_coord(row, b0, odd, pow2_n);
    const uint j1 = crt_mixed_j_from_coord(row, b1, odd, pow2_n);
    const GF31 z = a31[id];
    const uint r0 = (uint)(((ulong)j0 * (ulong)p) % (ulong)n);
    const uint r1 = (uint)(((ulong)j1 * (ulong)p) % (ulong)n);
    uint s0 = f31_mod31_small(shift_from_r31(r0, lr2_31) + log_m);
    uint s1 = f31_mod31_small(shift_from_r31(r1, lr2_31) + log_m);
    digits31[j0] = f31_lshift_scalar(z.s0, s0 == 0u ? 0u : 31u - s0);
    digits31[j1] = f31_lshift_scalar(z.s1, s1 == 0u ? 0u : 31u - s1);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_weight_61(__global const ulong* restrict digits,
                                      __global GF* restrict a61,
                                      uint n, uint p, uint lr2_61)
{
    const uint j = (uint)get_global_id(0);
    const uint m = n >> 1;
    if (j >= m) return;
    const uint i0 = j << 1;
    const uint i1 = i0 + 1u;
    const uint mask = n - 1u;
    const uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
    const uint r1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)mask);
    a61[j] = (GF)(lshift61(digits[i0], shift_from_r_on_the_fly(r0, lr2_61)),
                  lshift61(digits[i1], shift_from_r_on_the_fly(r1, lr2_61)));
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_weight_31(__global const ulong* restrict digits,
                                      __global GF31* restrict a31,
                                      uint n, uint p, uint lr2_31)
{
    const uint j = (uint)get_global_id(0);
    const uint m = n >> 1;
    if (j >= m) return;
    const uint i0 = j << 1;
    const uint i1 = i0 + 1u;
    a31[j] = (GF31)(f31_weight_scalar_idx(digits[i0], i0, p, n, lr2_31),
                    f31_weight_scalar_idx(digits[i1], i1, p, n, lr2_31));
}


static inline __attribute__((always_inline)) uint crt_shift61_from_prod(uint prod61)
{
    
    uint s = 62u - prod61;
    s -= 61u & (0u - (uint)(s >= 61u));
    return s;
}

static inline __attribute__((always_inline)) uint crt_shift31_from_prod(uint r, uint prod31)
{
    
    const uint s = f31_mod31_small(32u - prod31);
    return s & (0u - (uint)(r != 0u));
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_lds512_dif_precrt(__global const ulong* restrict digits,
                                               __global GF* restrict a61,
                                               __global const GF* restrict tw61,
                                               __global GF31* restrict a31,
                                               __global const GF31* restrict tw31,
                                               uint n, uint p, uint lr2_61, uint lr2_31)
{
    const uint lane = get_local_id(0);
    const uint m = n >> 1;
    if (m < 512u) return;
    const uint stride = m >> 9;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;

    const uint mask = n - 1u;
    const uint pmod = p & mask;
    const uint idx_base = j0 + lane * stride;
    const uint idx_step = stride << 6;
    const uint i_step = idx_step << 1;
    const uint r_step = (uint)(((ulong)i_step * (ulong)p) & (ulong)mask);

    __local GF   l61[512];
    __local GF31 l31[512];

    
    const uint lr2_61_mod = crt_mod61_u32_fast(lr2_61);
    const uint n61_lr2    = crt_mod61_small(crt_mod61_u32_fast(n) * lr2_61_mod);
    const uint p61_lr2    = crt_mod61_small(crt_mod61_u32_fast(pmod) * lr2_61_mod);
    const uint step61_lr2 = crt_mod61_small(crt_mod61_u32_fast(r_step) * lr2_61_mod);

    const uint lr2_31_mod = f31_mod31_small(lr2_31);
    const uint n31_lr2    = crt_mod31_u32_fast(crt_mod31_u32_fast(n) * lr2_31_mod);
    const uint p31_lr2    = crt_mod31_u32_fast(crt_mod31_u32_fast(pmod) * lr2_31_mod);
    const uint step31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(r_step) * lr2_31_mod);

    uint i0 = idx_base << 1;
    uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
    uint prod0_61 = crt_mod61_small(crt_mod61_u32_fast(r0) * lr2_61_mod);
    uint prod0_31 = crt_mod31_u32_fast(crt_mod31_u32_fast(r0) * lr2_31_mod);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint i1 = i0 + 1u;
        uint r1 = r0 + pmod;
        const uint wrap1 = (uint)(r1 >= n);
        r1 -= n & (0u - wrap1);

        uint prod1_61 = crt_add61_fast(prod0_61, p61_lr2);
        prod1_61 = crt_sub61_if(prod1_61, n61_lr2, wrap1);
        uint prod1_31 = crt_add31_fast(prod0_31, p31_lr2);
        prod1_31 = crt_sub31_if(prod1_31, n31_lr2, wrap1);

        const ulong d0 = digits[i0];
        const ulong d1 = digits[i1];

        l61[t] = (GF)(lshift61(d0, crt_shift61_from_prod(prod0_61)),
                      lshift61(d1, crt_shift61_from_prod(prod1_61)));

        const uint s0 = crt_shift31_from_prod(r0, prod0_31);
        const uint s1 = crt_shift31_from_prod(r1, prod1_31);
        l31[t] = (GF31)(f31_lshift_scalar_nomod(f31_reduce_ulong(d0), s0),
                        f31_lshift_scalar_nomod(f31_reduce_ulong(d1), s1));

        i0 += i_step;
        uint nr0 = r0 + r_step;
        const uint wrap0 = (uint)(nr0 >= n);
        nr0 -= n & (0u - wrap0);
        prod0_61 = crt_sub61_if(crt_add61_fast(prod0_61, step61_lr2), n61_lr2, wrap0);
        prod0_31 = crt_sub31_if(crt_add31_fast(prod0_31, step31_lr2), n31_lr2, wrap0);
        r0 = nr0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dif8_61(l61, tw61, 512u, 512u, stride, j0, lane);
    crt_lds_dif8_31(l31, tw31, 512u, 512u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61(l61, tw61, 512u,  64u, stride, j0, lane);
    crt_lds_dif8_31(l31, tw31, 512u,  64u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61_leaf512(l61, lane);
    crt_lds_dif8_31_leaf512(l31, lane);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        a61[idx] = l61[t];
        a31[idx] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_lds512_dif_61(__global const ulong* restrict digits,
                                          __global GF* restrict a61,
                                          __global const GF* restrict tw61,
                                          uint n, uint p, uint lr2_61)
{
    const uint lane = get_local_id(0);
    const uint m = n >> 1;
    if (m < 512u) return;
    const uint stride = m >> 9;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint mask = n - 1u;
    const uint idx_base = j0 + lane * stride;
    const uint idx_step = stride << 6;
    const uint i_step = idx_step << 1;
    const uint r_step = (uint)(((ulong)i_step * (ulong)p) & (ulong)mask);

    __local GF l61[512];

    
    const uint pmod = p & mask;
    const uint lr2_61_mod = crt_mod61_u32_fast(lr2_61);
    const uint n61_lr2 = crt_mod61_small(crt_mod61_u32_fast(n) * lr2_61_mod);
    const uint p61_lr2 = crt_mod61_small(crt_mod61_u32_fast(pmod) * lr2_61_mod);
    const uint step61_lr2 = crt_mod61_small(crt_mod61_u32_fast(r_step) * lr2_61_mod);

    uint i0 = idx_base << 1;
    uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
    uint prod0_61 = crt_mod61_small(crt_mod61_u32_fast(r0) * lr2_61_mod);
    for (uint t = lane; t < 512u; t += 64u) {
        const uint i1 = i0 + 1u;
        uint r1 = r0 + pmod;
        const uint wrap1 = (uint)(r1 >= n);
        r1 -= n & (0u - wrap1);
        uint prod1_61 = crt_add61_fast(prod0_61, p61_lr2);
        prod1_61 = crt_sub61_if(prod1_61, n61_lr2, wrap1);

        l61[t] = (GF)(lshift61(digits[i0], crt_shift61_from_prod(prod0_61)),
                      lshift61(digits[i1], crt_shift61_from_prod(prod1_61)));

        i0 += i_step;
        uint nr0 = r0 + r_step;
        const uint wrap0 = (uint)(nr0 >= n);
        nr0 -= n & (0u - wrap0);
        prod0_61 = crt_sub61_if(crt_add61_fast(prod0_61, step61_lr2), n61_lr2, wrap0);
        r0 = nr0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dif8_61(l61, tw61, 512u, 512u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61(l61, tw61, 512u,  64u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_61_leaf512(l61, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        a61[idx] = l61[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_lds512_dif_31(__global const ulong* restrict digits,
                                          __global GF31* restrict a31,
                                          __global const GF31* restrict tw31,
                                          uint n, uint p, uint lr2_31)
{
    const uint lane = get_local_id(0);
    const uint m = n >> 1;
    if (m < 512u) return;
    const uint stride = m >> 9;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint mask = n - 1u;
    const uint pmod = p & mask;
    const uint idx_base = j0 + lane * stride;
    const uint idx_step = stride << 6;
    const uint i_step = idx_step << 1;
    const uint r_step = (uint)(((ulong)i_step * (ulong)p) & (ulong)mask);
    const uint lr2_31_mod = f31_mod31_small(lr2_31);

    
    const uint n31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(n) * lr2_31_mod);
    const uint p31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(pmod) * lr2_31_mod);
    const uint step31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(r_step) * lr2_31_mod);

    __local GF31 l31[512];
    uint i0 = idx_base << 1;
    uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
    uint prod0_31 = crt_mod31_u32_fast(crt_mod31_u32_fast(r0) * lr2_31_mod);
    for (uint t = lane; t < 512u; t += 64u) {
        const uint i1 = i0 + 1u;
        uint r1 = r0 + pmod;
        const uint wrap1 = (uint)(r1 >= n);
        r1 -= n & (0u - wrap1);
        uint prod1_31 = crt_add31_fast(prod0_31, p31_lr2);
        prod1_31 = crt_sub31_if(prod1_31, n31_lr2, wrap1);
        const uint s0 = crt_shift31_from_prod(r0, prod0_31);
        const uint s1 = crt_shift31_from_prod(r1, prod1_31);
        l31[t] = (GF31)(f31_lshift_scalar_nomod(f31_reduce_ulong(digits[i0]), s0),
                        f31_lshift_scalar_nomod(f31_reduce_ulong(digits[i1]), s1));

        i0 += i_step;
        uint nr0 = r0 + r_step;
        const uint wrap0 = (uint)(nr0 >= n);
        nr0 -= n & (0u - wrap0);
        prod0_31 = crt_sub31_if(crt_add31_fast(prod0_31, step31_lr2), n31_lr2, wrap0);
        r0 = nr0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dif8_31(l31, tw31, 512u, 512u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_31(l31, tw31, 512u,  64u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dif8_31_leaf512(l31, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        a31[idx] = l31[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_dit512_unpack_61(__global GF* restrict a61,
                                        __global const GF* restrict tw61,
                                        __global ulong* restrict digits,
                                        uint n, uint p, uint lr2_61, uint log_m)
{
    const uint lane = get_local_id(0);
    const uint m = n >> 1;
    if (m < 512u) return;
    const uint stride = m >> 9;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint mask = n - 1u;

    __local GF l61[512];
    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        l61[t] = a61[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dit8_61_leaf512(l61, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u,  16u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u, 128u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        const uint i0 = idx << 1;
        const uint i1 = i0 + 1u;
        const uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
        const uint r1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)mask);
        const GF z = l61[t];
        digits[i0] = rshift61(norm61(z.x), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
        digits[i1] = rshift61(norm61(z.y), shift_from_r_on_the_fly(r1, lr2_61) + log_m);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_dit512_unpack_31(__global GF31* restrict a31,
                                        __global const GF31* restrict tw31,
                                        __global uint* restrict digits31,
                                        uint n, uint p, uint lr2_31, uint log_m)
{
    const uint lane = get_local_id(0);
    const uint m = n >> 1;
    if (m < 512u) return;
    const uint stride = m >> 9;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;

    __local GF31 l31[512];
    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        l31[t] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dit8_31_leaf512(l31, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_31(l31, tw31, 512u,  16u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_31(l31, tw31, 512u, 128u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        const uint i0 = idx << 1;
        const uint i1 = i0 + 1u;
        const GF31 z = l31[t];
        digits31[i0] = f31_unweight_scalar_idx(z.x, i0, p, n, lr2_31, log_m);
        digits31[i1] = f31_unweight_scalar_idx(z.y, i1, p, n, lr2_31, log_m);
    }
}

inline uint f31_unweight_scalar_r(uint v, uint r, uint lr2, uint log_m)
{
    const uint s = f31_mod31_small(shift_from_r31(r, lr2) + log_m);
    return f31_lshift_scalar(v, s == 0u ? 0u : 31u - s);
}
__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_dit512_unpack_precrt(
    __global GF* restrict a61,
    __global const GF* restrict tw61,
    __global GF31* restrict a31,
    __global const GF31* restrict tw31,
    __global ulong* restrict coeff_lo,
    __global uint* restrict coeff_hi,
    uint n, uint p, uint lr2_61, uint lr2_31, uint log_m)
{
    const uint lane = get_local_id(0);
    const uint m = n >> 1;
    if (m < 512u) return;
    const uint stride = m >> 9;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint mask = n - 1u;
    const uint idx_base = j0 + lane * stride;
    const uint idx_step = stride << 6;
    const uint i_step = idx_step << 1;
    const uint r_step = (uint)(((ulong)i_step * (ulong)p) & (ulong)mask);

    __local GF l61[512];
    __local GF31 l31[512];
    for (uint t = lane; t < 512u; t += 64u) {
        const uint idx = j0 + t * stride;
        l61[t] = a61[idx];
        l31[t] = a31[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    crt_lds_dit8_61_leaf512(l61, lane);
    crt_lds_dit8_31_leaf512(l31, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u,  16u, stride, j0, lane);
    crt_lds_dit8_31(l31, tw31, 512u,  16u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    crt_lds_dit8_61(l61, tw61, 512u, 128u, stride, j0, lane);
    crt_lds_dit8_31(l31, tw31, 512u, 128u, stride, j0, lane);
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint pmod = p & mask;
    const uint lr2_61_mod = crt_mod61_u32_fast(lr2_61);
    const uint lr2_31_mod = f31_mod31_small(lr2_31);
    const uint log_m31 = f31_mod31_small(log_m);

    
    const uint n61_lr2 = crt_mod61_small(crt_mod61_u32_fast(n) * lr2_61_mod);
    const uint p61_lr2 = crt_mod61_small(crt_mod61_u32_fast(pmod) * lr2_61_mod);
    const uint step61_lr2 = crt_mod61_small(crt_mod61_u32_fast(r_step) * lr2_61_mod);
    const uint n31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(n) * lr2_31_mod);
    const uint p31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(pmod) * lr2_31_mod);
    const uint step31_lr2 = crt_mod31_u32_fast(crt_mod31_u32_fast(r_step) * lr2_31_mod);

    uint i0 = idx_base << 1;
    uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
    uint prod0_61 = crt_mod61_small(crt_mod61_u32_fast(r0) * lr2_61_mod);
    uint prod0_31 = crt_mod31_u32_fast(crt_mod31_u32_fast(r0) * lr2_31_mod);

    for (uint t = lane; t < 512u; t += 64u) {
        const uint i1 = i0 + 1u;
        uint r1 = r0 + pmod;
        const uint wrap1 = (uint)(r1 >= n);
        r1 -= n & (0u - wrap1);

        uint prod1_61 = crt_add61_fast(prod0_61, p61_lr2);
        prod1_61 = crt_sub61_if(prod1_61, n61_lr2, wrap1);
        uint prod1_31 = crt_add31_fast(prod0_31, p31_lr2);
        prod1_31 = crt_sub31_if(prod1_31, n31_lr2, wrap1);

        const GF z61 = l61[t];
        const GF31 z31 = l31[t];
        const ulong d61_0 = rshift61(norm61(z61.x), crt_shift61_from_prod(prod0_61) + log_m);
        const ulong d61_1 = rshift61(norm61(z61.y), crt_shift61_from_prod(prod1_61) + log_m);
        const uint s31_0 = crt_add31_fast(crt_shift31_from_prod(r0, prod0_31), log_m31);
        const uint s31_1 = crt_add31_fast(crt_shift31_from_prod(r1, prod1_31), log_m31);
        const uint d31_0 = f31_rshift_scalar_nomod(z31.x, s31_0);
        const uint d31_1 = f31_rshift_scalar_nomod(z31.y, s31_1);
        crt_store_coeff_from_tail_residues(coeff_lo, coeff_hi, i0, d61_0, d31_0);
        crt_store_coeff_from_tail_residues(coeff_lo, coeff_hi, i1, d61_1, d31_1);

        i0 += i_step;
        uint nr0 = r0 + r_step;
        const uint wrap0 = (uint)(nr0 >= n);
        nr0 -= n & (0u - wrap0);
        prod0_61 = crt_sub61_if(crt_add61_fast(prod0_61, step61_lr2), n61_lr2, wrap0);
        prod0_31 = crt_sub31_if(crt_add31_fast(prod0_31, step31_lr2), n31_lr2, wrap0);
        r0 = nr0;
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_ldspow2_dif_61(__global const ulong* restrict digits,
                                            __global GF* restrict a,
                                            __global const GF* restrict tw,
                                            uint n, uint p, uint lr2, uint radix)
{
    const uint m = n >> 1;
    if (!gf61_crt_valid_stage_radix(radix) || radix > 1024u || m < radix || (m % radix) != 0u) return;
    const uint stride = m / radix;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint lane = get_local_id(0);
    const uint mask = n - 1u;
    __local GF l[1024];

    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = j0 + t * stride;
        const uint i0 = idx << 1;
        const uint i1 = i0 + 1u;
        const uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
        const uint r1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)mask);
        l[t] = (GF)(lshift61(digits[i0], shift_from_r_on_the_fly(r0, lr2)),
                    lshift61(digits[i1], shift_from_r_on_the_fly(r1, lr2)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) { crt_lds_dif8_61(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L >>= 3; }
    if (L == 4u) { crt_lds_dif4_61(l, tw, radix, 4u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); }
    else if (L == 2u) { crt_lds_dif2_61(l, tw, radix, 2u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); }

    for (uint t = lane; t < radix; t += 64u) a[j0 + t * stride] = l[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_pack_ldspow2_dif_31(__global const ulong* restrict digits,
                                            __global GF31* restrict a,
                                            __global const GF31* restrict tw,
                                            uint n, uint p, uint lr2, uint radix)
{
    const uint m = n >> 1;
    if (!gf61_crt_valid_stage_radix(radix) || radix > 1024u || m < radix || (m % radix) != 0u) return;
    const uint stride = m / radix;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint lane = get_local_id(0);
    __local GF31 l[1024];

    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = j0 + t * stride;
        const uint i0 = idx << 1;
        l[t] = (GF31)(f31_weight_scalar_idx((uint)(digits[i0] % CRT_M31), i0, p, n, lr2),
                      f31_weight_scalar_idx((uint)(digits[i0 + 1u] % CRT_M31), i0 + 1u, p, n, lr2));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint L = radix;
    while (L >= 8u) { crt_lds_dif8_31(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L >>= 3; }
    if (L == 4u) { crt_lds_dif4_31(l, tw, radix, 4u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); }
    else if (L == 2u) { crt_lds_dif2_31(l, tw, radix, 2u, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); }

    for (uint t = lane; t < radix; t += 64u) a[j0 + t * stride] = l[t];
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_ditpow2_unpack_61(__global GF* restrict a,
                                          __global const GF* restrict tw,
                                          __global ulong* restrict digits,
                                          uint n, uint p, uint lr2, uint log_m, uint radix)
{
    const uint m = n >> 1;
    if (!gf61_crt_valid_stage_radix(radix) || radix > 1024u || m < radix || (m % radix) != 0u) return;
    const uint stride = m / radix;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint lane = get_local_id(0);
    __local GF l[1024];

    for (uint t = lane; t < radix; t += 64u) l[t] = a[j0 + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) { crt_lds_dit2_61(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L <<= 1; }
    else if (tail == 4u) { crt_lds_dit4_61(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L <<= 2; }
    while (L < radix) { crt_lds_dit8_61(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L <<= 3; }

    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = j0 + t * stride;
        const uint i0 = idx << 1;
        const uint i1 = i0 + 1u;
        const uint mask = n - 1u;
        const uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
        const uint r1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)mask);
        const GF z = l[t];
        digits[i0] = rshift61(norm61(z.s0), shift_from_r_on_the_fly(r0, lr2) + log_m);
        digits[i1] = rshift61(norm61(z.s1), shift_from_r_on_the_fly(r1, lr2) + log_m);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_ditpow2_unpack_31(__global GF31* restrict a,
                                          __global const GF31* restrict tw,
                                          __global uint* restrict digits,
                                          uint n, uint p, uint lr2, uint log_m, uint radix)
{
    const uint m = n >> 1;
    if (!gf61_crt_valid_stage_radix(radix) || radix > 1024u || m < radix || (m % radix) != 0u) return;
    const uint stride = m / radix;
    const uint j0 = get_group_id(0);
    if (j0 >= stride) return;
    const uint lane = get_local_id(0);
    __local GF31 l[1024];

    for (uint t = lane; t < radix; t += 64u) l[t] = a[j0 + t * stride];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tail = radix;
    while ((tail & 7u) == 0u && tail > 1u) tail >>= 3;
    uint L = 2u;
    if (tail == 2u) { crt_lds_dit2_31(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L <<= 1; }
    else if (tail == 4u) { crt_lds_dit4_31(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L <<= 2; }
    while (L < radix) { crt_lds_dit8_31(l, tw, radix, L, stride, j0, lane); barrier(CLK_LOCAL_MEM_FENCE); L <<= 3; }

    for (uint t = lane; t < radix; t += 64u) {
        const uint idx = j0 + t * stride;
        const uint i0 = idx << 1;
        const uint i1 = i0 + 1u;
        const GF31 z = l[t];
        digits[i0] = f31_unweight_scalar_idx(z.x, i0, p, n, lr2, log_m);
        digits[i1] = f31_unweight_scalar_idx(z.y, i1, p, n, lr2, log_m);
    }
}


static inline __attribute__((always_inline))
void crt_halfreal_center_eo_61(GF z, GF zneg_conj,
                               __global const GF* restrict twf,
                               __global const GF* restrict twi,
                               const uint n,
                               const uint k,
                               const uint flags,
                               __private GF* Eout,
                               __private GF* Oout)
{
    
    
    GF E = gf_half_scalar_pair(gf_add(z, zneg_conj));
    GF D = gf_half_scalar_pair(gf_sub(z, zneg_conj));
    GF O = (flags & 4u) ? gf_mul_i_fast(D) : gf_mul_minus_i_fast(D);

    const uint off = (n >> 1) - 1u + k;
    const GF W  = (flags & 2u) ? twi[off] : twf[off];
    const GF WO = gf_mul(O, W);

#ifndef CRT_HALFREAL_CENTER_DIRECT
#define CRT_HALFREAL_CENTER_DIRECT 1
#endif
#if CRT_HALFREAL_CENTER_DIRECT
    
    
    const GF E0 = E;
    const GF O0 = O;
    E = gf_add(gf_sqr(E0), gf_sqr(WO));
    O = gf_dbl_pair(gf_mul(E0, O0));
#else
    const GF Wi = gf_conj_fast(W);
    const GF X0 = gf_sqr(gf_add(E, WO));
    const GF X1 = gf_sqr(gf_sub(E, WO));
    E = gf_half_scalar_pair(gf_add(X0, X1));
    O = gf_mul(gf_half_scalar_pair(gf_sub(X0, X1)), Wi);
#endif
    *Eout = E;
    *Oout = O;
}


#ifndef CRT_MIXED_F48_TWIN_SYMMETRY_61
#define CRT_MIXED_F48_TWIN_SYMMETRY_61 1
#endif
#ifndef CRT_MIXED_F48_TWIN_SYMMETRY_31
#define CRT_MIXED_F48_TWIN_SYMMETRY_31 1
#endif

static inline __attribute__((always_inline))
GF crt_halfreal_f48_tw_61(__global const GF* restrict twf, const uint n, const uint k)
{
    const uint row_m = n >> 1;
#if CRT_MIXED_F48_TWIN_SYMMETRY_61
    const uint km = (row_m - k) & (row_m - 1u);
    return (k > km) ? gf_neg_conj_fast(twf[(row_m - 1u) + km]) : twf[(row_m - 1u) + k];
#else
    return twf[(row_m - 1u) + k];
#endif
}

static inline __attribute__((always_inline))
GF31 crt_halfreal_f48_tw_31(__global const GF31* restrict twf, const uint n, const uint k)
{
    const uint row_m = n >> 1;
#if CRT_MIXED_F48_TWIN_SYMMETRY_31
    const uint km = (row_m - k) & (row_m - 1u);
    return (k > km) ? f31_neg_conj_fast(twf[(row_m - 1u) + km]) : twf[(row_m - 1u) + k];
#else
    return twf[(row_m - 1u) + k];
#endif
}

#ifndef CRT_MIXED_F48_DELAYED_SCALE_61
#ifdef CRT_MIXED_F48_LATE_SCALE_61
#define CRT_MIXED_F48_DELAYED_SCALE_61 CRT_MIXED_F48_LATE_SCALE_61
#else
#define CRT_MIXED_F48_DELAYED_SCALE_61 0
#endif
#endif
#ifndef CRT_MIXED_F48_DELAYED_SCALE_31
#ifdef CRT_MIXED_F48_LATE_SCALE_31
#define CRT_MIXED_F48_DELAYED_SCALE_31 CRT_MIXED_F48_LATE_SCALE_31
#else
#define CRT_MIXED_F48_DELAYED_SCALE_31 0
#endif
#endif

static inline __attribute__((always_inline))
void crt_halfreal_center_eo_f48_61(GF z, GF zneg_conj,
                                   __global const GF* restrict twf,
                                   const uint n,
                                   const uint k,
                                   __private GF* Eout,
                                   __private GF* Oout)
{
    const GF W = crt_halfreal_f48_tw_61(twf, n, k);
#if CRT_MIXED_F48_DELAYED_SCALE_61
    const GF U = gf_norm_pair(gf_add(z, zneg_conj));
    const GF D = gf_mul_minus_i_fast(gf_norm_pair(gf_sub(z, zneg_conj)));
    const GF WD = gf_mul(D, W);
    *Eout = gf_quarter_scalar_pair(gf_add(gf_sqr(U), gf_sqr(WD)));
    *Oout = gf_half_scalar_pair(gf_mul(U, D));
#else
    GF E = gf_half_scalar_pair(gf_add(z, zneg_conj));
    GF O = gf_mul_minus_i_fast(gf_half_scalar_pair(gf_sub(z, zneg_conj)));
    const GF WO = gf_mul(O, W);
    const GF E0 = E;
    const GF O0 = O;
    E = gf_add(gf_sqr(E0), gf_sqr(WO));
    O = gf_dbl_pair(gf_mul(E0, O0));
    *Eout = E;
    *Oout = O;
#endif
}

static inline __attribute__((always_inline))
void crt_halfreal_center_pair_f48_w_61(GF z, GF zneg_conj, GF W, __private GF* out0, __private GF* out1)
{
#if CRT_MIXED_F48_DELAYED_SCALE_61
    const GF U = gf_norm_pair(gf_add(z, zneg_conj));
    const GF D = gf_mul_minus_i_fast(gf_norm_pair(gf_sub(z, zneg_conj)));
    const GF WD = gf_mul(D, W);
    const GF E = gf_quarter_scalar_pair(gf_add(gf_sqr(U), gf_sqr(WD)));
    const GF O = gf_half_scalar_pair(gf_mul(U, D));
#else
    const GF E0 = gf_half_scalar_pair(gf_add(z, zneg_conj));
    const GF O0 = gf_mul_minus_i_fast(gf_half_scalar_pair(gf_sub(z, zneg_conj)));
    const GF WO = gf_mul(O0, W);
    const GF E = gf_add(gf_sqr(E0), gf_sqr(WO));
    const GF O = gf_dbl_pair(gf_mul(E0, O0));
#endif
    *out0 = gf_pack_e_plus_i_o(E, O);
    *out1 = gf_pack_conj_e_plus_i_conj_o(E, O);
}

static inline __attribute__((always_inline))
void crt_halfreal_center_eo_f48_31(GF31 z, GF31 zneg_conj,
                                   __global const GF31* restrict twf,
                                   const uint n,
                                   const uint k,
                                   __private GF31* Eout,
                                   __private GF31* Oout)
{
    const GF31 W = crt_halfreal_f48_tw_31(twf, n, k);
#if CRT_MIXED_F48_DELAYED_SCALE_31
    const GF31 U = f31_add(z, zneg_conj);
    const GF31 D = f31_mul_minus_i_fast(f31_sub(z, zneg_conj));
    const GF31 WD = f31_mul(D, W);
    *Eout = f31_quarter_pair(f31_add(f31_sqr(U), f31_sqr(WD)));
    *Oout = f31_half_pair(f31_mul(U, D));
#else
    GF31 E = f31_half_pair(f31_add(z, zneg_conj));
    GF31 O = f31_mul_minus_i_fast(f31_half_pair(f31_sub(z, zneg_conj)));
    const GF31 WO = f31_mul(O, W);
    const GF31 E0 = E;
    const GF31 O0 = O;
    E = f31_add(f31_sqr(E0), f31_sqr(WO));
    O = f31_dbl_pair(f31_mul(E0, O0));
    *Eout = E;
    *Oout = O;
#endif
}

static inline __attribute__((always_inline))
void crt_halfreal_center_pair_f48_w_31(GF31 z, GF31 zneg_conj, GF31 W, __private GF31* out0, __private GF31* out1)
{
#if CRT_MIXED_F48_DELAYED_SCALE_31
    const GF31 U = f31_add(z, zneg_conj);
    const GF31 D = f31_mul_minus_i_fast(f31_sub(z, zneg_conj));
    const GF31 WD = f31_mul(D, W);
    const GF31 E = f31_quarter_pair(f31_add(f31_sqr(U), f31_sqr(WD)));
    const GF31 O = f31_half_pair(f31_mul(U, D));
#else
    const GF31 E0 = f31_half_pair(f31_add(z, zneg_conj));
    const GF31 O0 = f31_mul_minus_i_fast(f31_half_pair(f31_sub(z, zneg_conj)));
    const GF31 WO = f31_mul(O0, W);
    const GF31 E = f31_add(f31_sqr(E0), f31_sqr(WO));
    const GF31 O = f31_dbl_pair(f31_mul(E0, O0));
#endif
    *out0 = f31_pack_e_plus_i_o(E, O);
    *out1 = f31_pack_conj_e_plus_i_conj_o(E, O);
}

static inline __attribute__((always_inline))
GF crt_halfreal_center_one_61(GF z, GF zneg_conj,
                              __global const GF* restrict twf,
                              __global const GF* restrict twi,
                              const uint n,
                              const uint k,
                              const uint flags)
{
    GF E, O;
    crt_halfreal_center_eo_61(z, zneg_conj, twf, twi, n, k, flags, &E, &O);
    return (flags & 8u) ? gf_pack_e_minus_i_o(E, O) : gf_pack_e_plus_i_o(E, O);
}

static inline __attribute__((always_inline))
void crt_halfreal_center_eo_31(GF31 z, GF31 zneg_conj,
                               __global const GF31* restrict twf,
                               __global const GF31* restrict twi,
                               const uint n,
                               const uint k,
                               const uint flags,
                               __private GF31* Eout,
                               __private GF31* Oout)
{
    
    
    GF31 E = f31_half_pair(f31_add(z, zneg_conj));
    GF31 D = f31_half_pair(f31_sub(z, zneg_conj));
    GF31 O = (flags & 4u) ? f31_mul_i_fast(D) : f31_mul_minus_i_fast(D);

    const uint off = (n >> 1) - 1u + k;
    const GF31 W  = (flags & 2u) ? twi[off] : twf[off];
    const GF31 WO = f31_mul(O, W);

#if CRT_HALFREAL_CENTER_DIRECT
    const GF31 E0 = E;
    const GF31 O0 = O;
    E = f31_add(f31_sqr(E0), f31_sqr(WO));
    O = f31_dbl_pair(f31_mul(E0, O0));
#else
    const GF31 Wi = f31_conj_fast(W);
    const GF31 X0 = f31_sqr(f31_add(E, WO));
    const GF31 X1 = f31_sqr(f31_sub(E, WO));
    E = f31_half_pair(f31_add(X0, X1));
    O = f31_mul(f31_half_pair(f31_sub(X0, X1)), Wi);
#endif
    *Eout = E;
    *Oout = O;
}

static inline __attribute__((always_inline))
GF31 crt_halfreal_center_one_31(GF31 z, GF31 zneg_conj,
                                __global const GF31* restrict twf,
                                __global const GF31* restrict twi,
                                const uint n,
                                const uint k,
                                const uint flags)
{
    GF31 E, O;
    crt_halfreal_center_eo_31(z, zneg_conj, twf, twi, n, k, flags, &E, &O);
    return (flags & 8u) ? f31_pack_e_minus_i_o(E, O) : f31_pack_e_plus_i_o(E, O);
}


static inline __attribute__((always_inline))
uint crt_radix8_digitrev_u32(uint x, uint logn)
{
    const uint groups = logn / 3u;
    const uint rem = logn - groups * 3u;
    uint y = 0u;

    for (uint i = 0u; i < groups; ++i) {
        y = (y << 3u) | ((x >> (3u * i)) & 7u);
    }
    if (rem != 0u) {
        y = (y << rem) | (x >> (3u * groups));
    }
    return y;
}

static inline __attribute__((always_inline))
uint crt_bitrev_u32(uint x, uint logn)
{
    
    
    if (logn == 0u) return 0u;
    x = ((x & 0x55555555u) << 1u) | ((x >> 1u) & 0x55555555u);
    x = ((x & 0x33333333u) << 2u) | ((x >> 2u) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4u) | ((x >> 4u) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8u) | ((x >> 8u) & 0x00ff00ffu);
    x = (x << 16u) | (x >> 16u);
    return x >> (32u - logn);
}


static inline __attribute__((always_inline))
uint crt_bitrev3_u32_fast(uint x)
{
    
    return ((x & 1u) << 2) | (x & 2u) | ((x & 4u) >> 2);
}

static inline __attribute__((always_inline))
uint crt_bitrev4_u32_fast(uint x)
{
    x &= 15u;
    return ((x & 1u) << 3) | ((x & 2u) << 1) | ((x & 4u) >> 1) | ((x & 8u) >> 3);
}

static inline __attribute__((always_inline))
uint crt_bitrev9_u32_fast(uint x)
{
    
    
    x &= 511u;
    return ((x & 0x001u) << 8) |
           ((x & 0x002u) << 6) |
           ((x & 0x004u) << 4) |
           ((x & 0x008u) << 2) |
            (x & 0x010u)       |
           ((x & 0x020u) >> 2) |
           ((x & 0x040u) >> 4) |
           ((x & 0x080u) >> 6) |
           ((x & 0x100u) >> 8);
}

static inline __attribute__((always_inline))
uint crt_bitrev10_u32_fast(uint x)
{
    x &= 1023u;
    return ((x & 0x001u) << 9) |
           ((x & 0x002u) << 7) |
           ((x & 0x004u) << 5) |
           ((x & 0x008u) << 3) |
           ((x & 0x010u) << 1) |
           ((x & 0x020u) >> 1) |
           ((x & 0x040u) >> 3) |
           ((x & 0x080u) >> 5) |
           ((x & 0x100u) >> 7) |
           ((x & 0x200u) >> 9);
}

static inline __attribute__((always_inline))
uint crt_halfreal_perm_u32(uint x, uint logn, uint flags)
{
    
    
    if (flags & 16u) return crt_bitrev_u32(x, logn);
    if (flags & 1u) return crt_radix8_digitrev_u32(x, logn);
    return x;
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_bitrev_swap_61(__global GF* restrict a61, uint m)
{
    const uint j = (uint)get_global_id(0);
    if (j >= m) return;
    const uint log_m = 31u - (uint)clz(m);
    const uint r = crt_bitrev_u32(j, log_m);
    if (j < r) {
        const GF t = a61[j];
        a61[j] = a61[r];
        a61[r] = t;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_bitrev_swap_31(__global GF31* restrict a31, uint m)
{
    const uint j = (uint)get_global_id(0);
    if (j >= m) return;
    const uint log_m = 31u - (uint)clz(m);
    const uint r = crt_bitrev_u32(j, log_m);
    if (j < r) {
        const GF31 t = a31[j];
        a31[j] = a31[r];
        a31[r] = t;
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_center_61(__global GF* restrict a61,
                                 __global const GF* restrict twf61,
                                 __global const GF* restrict twi61,
                                 uint n,
                                 uint flags)
{
    const uint m = n >> 1;
    const uint k = (uint)get_global_id(0);
    if (k >= m) return;

    const uint km = (m - k) & (m - 1u);
    if (k > km) return;

    const uint log_m = 31u - (uint)clz(m);
    const uint pk  = crt_halfreal_perm_u32(k,  log_m, flags);
    const uint pkm = crt_halfreal_perm_u32(km, log_m, flags);

    const GF zk  = a61[pk];
    const GF zkm = a61[pkm];
    if ((flags & 32u) && km != k) {
        GF E, O;
        crt_halfreal_center_eo_f48_61(zk, gf_conj_fast(zkm), twf61, n, k, &E, &O);
        a61[pk] = gf_pack_e_plus_i_o(E, O);
        a61[pkm] = gf_pack_conj_e_plus_i_conj_o(E, O);
    } else {
        a61[pk] = crt_halfreal_center_one_61(zk, gf_conj_fast(zkm), twf61, twi61, n, k, flags);
        if (km != k) a61[pkm] = crt_halfreal_center_one_61(zkm, gf_conj_fast(zk), twf61, twi61, n, km, flags);
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_center_31(__global GF31* restrict a31,
                                 __global const GF31* restrict twf31,
                                 __global const GF31* restrict twi31,
                                 uint n,
                                 uint flags)
{
    const uint m = n >> 1;
    const uint k = (uint)get_global_id(0);
    if (k >= m) return;

    const uint km = (m - k) & (m - 1u);
    if (k > km) return;

    const uint log_m = 31u - (uint)clz(m);
    const uint pk  = crt_halfreal_perm_u32(k,  log_m, flags);
    const uint pkm = crt_halfreal_perm_u32(km, log_m, flags);

    const GF31 zk  = a31[pk];
    const GF31 zkm = a31[pkm];
    if ((flags & 32u) && km != k) {
        GF31 E, O;
        crt_halfreal_center_eo_f48_31(zk, f31_conj_fast(zkm), twf31, n, k, &E, &O);
        a31[pk] = f31_pack_e_plus_i_o(E, O);
        a31[pkm] = f31_pack_conj_e_plus_i_conj_o(E, O);
    } else {
        a31[pk] = crt_halfreal_center_one_31(zk, f31_conj_fast(zkm), twf31, twi31, n, k, flags);
        if (km != k) a31[pkm] = crt_halfreal_center_one_31(zkm, f31_conj_fast(zk), twf31, twi31, n, km, flags);
    }
}


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_center512_pair_61(__global GF* restrict a61,
                                         __global const GF* restrict twf61,
                                         __global const GF* restrict twi61,
                                         uint n,
                                         uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint m = n >> 1;
    const uint log_m = 31u - (uint)clz(m);
    if (log_m < 9u) return;
    const uint blockA = (uint)get_group_id(0);
    const uint blocks = m >> 9;
    if (blockA >= blocks) return;

    
    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : crt_bitrev_u32(blockA, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);
    if (blockA > blockB) return;

    const uint baseA = blockA << 9;
    const uint baseB = blockB << 9;
    const int same = (blockA == blockB);

    __local GF A[512];
    __local GF B[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A[t] = a61[baseA + t];
        if (!same) B[t] = a61[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (m - k) & (m - 1u);
        
        
        const uint tm = (kb != 0u) ? (511u - t)
                                   : crt_bitrev9_u32_fast((0u - rt) & 511u);

        if (same && k > km) continue;

        if (k <= km) {
            const GF z = A[t];
            const GF zn = same ? A[tm] : B[tm];
            GF E, O;
            crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, n, k, &E, &O);
            const GF outK  = gf_pack_e_plus_i_o(E, O);
            const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
            A[t] = outK;
            if (same) {
                if (tm != t) A[tm] = outKm;
            } else {
                B[tm] = outKm;
            }
        } else {
            const GF zsmall = B[tm];
            const GF zlarge = A[t];
            GF E, O;
            crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, n, km, &E, &O);
            const GF outSmall = gf_pack_e_plus_i_o(E, O);
            const GF outLarge = gf_pack_conj_e_plus_i_conj_o(E, O);
            B[tm] = outSmall;
            A[t] = outLarge;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) {
        a61[baseA + t] = A[t];
        if (!same) a61[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_center512_pair_31(__global GF31* restrict a31,
                                         __global const GF31* restrict twf31,
                                         __global const GF31* restrict twi31,
                                         uint n,
                                         uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint m = n >> 1;
    const uint log_m = 31u - (uint)clz(m);
    if (log_m < 9u) return;
    const uint blockA = (uint)get_group_id(0);
    const uint blocks = m >> 9;
    if (blockA >= blocks) return;

    const uint h = log_m - 9u;
    const uint kb = (h == 0u) ? 0u : crt_bitrev_u32(blockA, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);
    if (blockA > blockB) return;

    const uint baseA = blockA << 9;
    const uint baseB = blockB << 9;
    const int same = (blockA == blockB);

    __local GF31 A[512];
    __local GF31 B[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A[t] = a31[baseA + t];
        if (!same) B[t] = a31[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (m - k) & (m - 1u);
        
        
        const uint tm = (kb != 0u) ? (511u - t)
                                   : crt_bitrev9_u32_fast((0u - rt) & 511u);

        if (same && k > km) continue;

        if (k <= km) {
            const GF31 z = A[t];
            const GF31 zn = same ? A[tm] : B[tm];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, n, k, &E, &O);
            const GF31 outK  = f31_pack_e_plus_i_o(E, O);
            const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
            A[t] = outK;
            if (same) {
                if (tm != t) A[tm] = outKm;
            } else {
                B[tm] = outKm;
            }
        } else {
            const GF31 zsmall = B[tm];
            const GF31 zlarge = A[t];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, n, km, &E, &O);
            const GF31 outSmall = f31_pack_e_plus_i_o(E, O);
            const GF31 outLarge = f31_pack_conj_e_plus_i_conj_o(E, O);
            B[tm] = outSmall;
            A[t] = outLarge;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = lid; t < 512u; t += 64u) {
        a31[baseA + t] = A[t];
        if (!same) a31[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_lds512_pair_61(__global GF* restrict a61,
                                      __global const GF* restrict twf61,
                                      __global const GF* restrict twi61,
                                      uint n,
                                      uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint m = n >> 1;
    const uint log_m = 31u - (uint)clz(m);
    if (log_m < 9u) return;
    const uint pair_id = (uint)get_group_id(0);
    const uint blocks = m >> 9;
    const uint pair_blocks = (blocks >> 1) + 1u;
    if (pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    
    
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint baseA = blockA << 9;
    const uint baseB = blockB << 9;
    const int same = (blockA == blockB);

    __local GF A[512];
    __local GF B[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A[t] = a61[baseA + t];
        if (!same) B[t] = a61[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint L = 512u; L >= 8u; L >>= 3) {
        local_stage_dif_radix8_pow2(A, twf61, 512u, L, lid, 64u);
        if (!same) local_stage_dif_radix8_pow2(B, twf61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (m - k) & (m - 1u);
        
        
        const uint tm = (kb != 0u) ? (511u - t)
                                   : crt_bitrev9_u32_fast((0u - rt) & 511u);

        if (same && k > km) continue;

        if (k <= km) {
            const GF z = A[t];
            const GF zn = same ? A[tm] : B[tm];
            GF E, O;
            crt_halfreal_center_eo_f48_61(z, gf_conj_fast(zn), twf61, n, k, &E, &O);
            const GF outK  = gf_pack_e_plus_i_o(E, O);
            const GF outKm = gf_pack_conj_e_plus_i_conj_o(E, O);
            A[t] = outK;
            if (same) {
                if (tm != t) A[tm] = outKm;
            } else {
                B[tm] = outKm;
            }
        } else {
            
            
            const GF zsmall = B[tm];
            const GF zlarge = A[t];
            GF E, O;
            crt_halfreal_center_eo_f48_61(zsmall, gf_conj_fast(zlarge), twf61, n, km, &E, &O);
            const GF outSmall = gf_pack_e_plus_i_o(E, O);
            const GF outLarge = gf_pack_conj_e_plus_i_conj_o(E, O);
            B[tm] = outSmall;
            A[t] = outLarge;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        local_stage_dit_radix8_pow2(A, twi61, 512u, L, lid, 64u);
        if (!same) local_stage_dit_radix8_pow2(B, twi61, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 512u; t += 64u) {
        a61[baseA + t] = A[t];
        if (!same) a61[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_lds512_pair_31(__global GF31* restrict a31,
                                      __global const GF31* restrict twf31,
                                      __global const GF31* restrict twi31,
                                      uint n,
                                      uint flags)
{
    const uint lid = (uint)get_local_id(0);
    const uint m = n >> 1;
    const uint log_m = 31u - (uint)clz(m);
    if (log_m < 9u) return;
    const uint pair_id = (uint)get_group_id(0);
    const uint blocks = m >> 9;
    const uint pair_blocks = (blocks >> 1) + 1u;
    if (pair_id >= pair_blocks) return;

    const uint h = log_m - 9u;
    
    
    const uint kb = (h == 0u) ? 0u : pair_id;
    const uint blockA = (h == 0u) ? 0u : crt_bitrev_u32(kb, h);
    const uint blockB = (h == 0u) ? 0u : crt_bitrev_u32((0u - kb) & (blocks - 1u), h);

    const uint baseA = blockA << 9;
    const uint baseB = blockB << 9;
    const int same = (blockA == blockB);

    __local GF31 A[512];
    __local GF31 B[512];

    for (uint t = lid; t < 512u; t += 64u) {
        A[t] = a31[baseA + t];
        if (!same) B[t] = a31[baseB + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 512u; L >= 8u; L >>= 3) {
        crt_local_stage_dif_radix8_31(A, twf31, 512u, L, lid, 64u);
        if (!same) crt_local_stage_dif_radix8_31(B, twf31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (L == 8u) break;
    }

    const uint rev_lid6 = crt_bitrev_u32(lid, 6u);
    for (uint r = 0u; r < 8u; ++r) {
        const uint t = (lid << 3) + r;
        const uint rt = (crt_bitrev3_u32_fast(r) << 6) | rev_lid6;
        const uint k = (rt << h) | kb;
        const uint km = (m - k) & (m - 1u);
        
        
        const uint tm = (kb != 0u) ? (511u - t)
                                   : crt_bitrev9_u32_fast((0u - rt) & 511u);

        if (same && k > km) continue;

        if (k <= km) {
            const GF31 z = A[t];
            const GF31 zn = same ? A[tm] : B[tm];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(z, f31_conj_fast(zn), twf31, n, k, &E, &O);
            const GF31 outK  = f31_pack_e_plus_i_o(E, O);
            const GF31 outKm = f31_pack_conj_e_plus_i_conj_o(E, O);
            A[t] = outK;
            if (same) {
                if (tm != t) A[tm] = outKm;
            } else {
                B[tm] = outKm;
            }
        } else {
            const GF31 zsmall = B[tm];
            const GF31 zlarge = A[t];
            GF31 E, O;
            crt_halfreal_center_eo_f48_31(zsmall, f31_conj_fast(zlarge), twf31, n, km, &E, &O);
            const GF31 outSmall = f31_pack_e_plus_i_o(E, O);
            const GF31 outLarge = f31_pack_conj_e_plus_i_conj_o(E, O);
            B[tm] = outSmall;
            A[t] = outLarge;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint L = 2u; (L << 2) <= 512u; L <<= 3) {
        crt_local_stage_dit_radix8_31(A, twi31, 512u, L, lid, 64u);
        if (!same) crt_local_stage_dit_radix8_31(B, twi31, 512u, L, lid, 64u);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint t = lid; t < 512u; t += 64u) {
        a31[baseA + t] = A[t];
        if (!same) a31[baseB + t] = B[t];
    }
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_unpack_unweight_61(__global const GF* restrict a61,
                                          __global ulong* restrict digits61,
                                          uint n, uint p, uint lr2_61, uint log_m)
{
    const uint j = (uint)get_global_id(0);
    const uint m = n >> 1;
    if (j >= m) return;
    const uint i0 = j << 1;
    const uint i1 = i0 + 1u;
    const uint mask = n - 1u;
    const GF z = a61[j];
    const uint r0 = (uint)(((ulong)i0 * (ulong)p) & (ulong)mask);
    const uint r1 = (uint)(((ulong)i1 * (ulong)p) & (ulong)mask);
    digits61[i0] = rshift61(norm61(z.s0), shift_from_r_on_the_fly(r0, lr2_61) + log_m);
    digits61[i1] = rshift61(norm61(z.s1), shift_from_r_on_the_fly(r1, lr2_61) + log_m);
}

__kernel __attribute__((reqd_work_group_size(64,1,1)))
void gf61_crt_halfreal_unpack_unweight_31(__global const GF31* restrict a31,
                                          __global uint* restrict digits31,
                                          uint n, uint p, uint lr2_31, uint log_m)
{
    const uint j = (uint)get_global_id(0);
    const uint m = n >> 1;
    if (j >= m) return;
    const uint i0 = j << 1;
    const uint i1 = i0 + 1u;
    const GF31 z = a31[j];
    digits31[i0] = f31_unweight_scalar_idx(z.s0, i0, p, n, lr2_31, log_m);
    digits31[i1] = f31_unweight_scalar_idx(z.s1, i1, p, n, lr2_31, log_m);
}
