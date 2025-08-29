/*
Copyright 2025, Yves Gallot

marin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#if __OPENCL_VERSION__ >= 120
	#define INLINE	static inline
#else
	#define INLINE
#endif

#if defined(__NV_CL_C_VERSION)
	#define PTX_ASM	1
#endif

#if !defined(N_SZ)
#define N_SZ		65536u
#define W_F1		4611686017353646079ul
#define W_F2		5818851782451133869ul
#define W_F3		10808002860802937880ul
#define W_F4		1418753320236437486ul
#define W_F5		7970496220330062908ul
#define BLK16		16u
#define BLK64		4u
#define BLK40		32u
#define BLK160		8u
#define BLK640		2u
#define CHUNK16		16u
#define CHUNK64		4u
#define CHUNK256	4u
#define CHUNK16_5	8u
#define CHUNK64_5	2u
#define CWM_WG_SZ	32u
#define CWM_WG_SZ2	16u
#define MAX_WG_SZ	256
#endif

typedef uint	sz_t;
typedef uchar	uint_8;
typedef uint	uint32;
typedef int		int32;
typedef ulong	uint64;
typedef uchar4	uint_8_4;
typedef ulong2	uint64_2;
typedef ulong4	uint64_4;

INLINE sz_t div5(const sz_t n) { return mul_hi(n, 858993460u); }	// = n / 5 if n < 2^30

// --- modular arithmetic ---

#define	MOD_P		0xffffffff00000001ul		// 2^64 - 2^32 + 1
#define	MOD_MP64	0xffffffffu					// -p mod (2^64) = 2^32 - 1

// t = 2^64 * hi + lo modulo p. We must have t < p^2.
INLINE uint64 reduce(const uint64 lo, const uint64 hi)
{
	const uint32 hi_hi = (uint32)(hi >> 32), hi_lo = (uint32)(hi);

	// Let X = hi_lo * (2^32 - 1) - hi_hi + lo = hi_hi * 2^96 + hi_lo * 2^64 + lo (mod p)
	// The trick is to add 2^32 - 1 to X (Nick Craig-Wood, ARM-32 assembly code)
	const uint32 d = MOD_MP64 - hi_hi;
	const uint64 s = upsample(hi_lo, d) - hi_lo;	// No carry: 0 <= s <= (2^32 - 1)^2 + 2^32 - 1 = 2^64 - 2^32 < p
#if defined(PTX_ASM)
	uint64 r; uint32 nc, c;
	asm volatile ("add.cc.u64 %0, %1, %2;" : "=l" (r) : "l" (s), "l" (lo));		// r = s + lo
	asm volatile ("addc.u32 %0, 0xffffffff, 0;" : "=r" (nc));							// If no carry then nc = MOD_MP64 else nc = 0
	const uint64 nc64 = upsample(0, nc);
	asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l" (r) : "l" (r), "l" (nc64));	// r -= nc
	asm volatile ("subc.u32 %0, 0, 0;" : "=r" (c));								// If borrow then c = MOD_MP64 else c = 0
	const uint64 c64 = upsample(0, c);
	asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l" (r) : "l" (r), "l" (c64));	// r -= c
#else
	uint64 r = s + lo;		// If carry then r + 2^64 = X + 2^32 - 1. We have r = X (mod p) and r < s < p
	if (r >= s)				// No carry
	{
		// Subtract 2^32 - 1. If the difference is negative then add p (+p = -(-p))
		const uint32 c = (r < MOD_MP64) ? MOD_MP64 : 0;	// borrow
		r -= MOD_MP64; r -= c;
	}
#endif
	return r;
}

INLINE uint64 mod_add(const uint64 lhs, const uint64 rhs) { return lhs + rhs + ((lhs >= MOD_P - rhs) ? MOD_MP64 : 0); }
INLINE uint64 mod_sub(const uint64 lhs, const uint64 rhs) { return lhs - rhs - ((lhs < rhs) ? MOD_MP64 : 0); }
INLINE uint64 mod_mul(const uint64 lhs, const uint64 rhs) { return reduce(lhs * rhs, mul_hi(lhs, rhs)); }
INLINE uint64 mod_sqr(const uint64 lhs) { return mod_mul(lhs, lhs); }
INLINE uint64 mod_muli(const uint64 lhs) { return reduce(lhs << 48, lhs >> (64 - 48)); }	// sqrt(-1) = 2^48 (mod p)
INLINE uint64 mod_half(const uint64 lhs) { return ((lhs % 2 == 0) ? lhs / 2 : ((lhs - 1) / 2 + (MOD_P + 1) / 2)); }

INLINE uint64_2 mod_add2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_add(lhs.s0, rhs.s0), mod_add(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_sub2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_sub(lhs.s0, rhs.s0), mod_sub(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_mul2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_mul(lhs.s0, rhs.s0), mod_mul(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_sqr2(const uint64_2 lhs) { return (uint64_2)(mod_sqr(lhs.s0), mod_sqr(lhs.s1)); }
INLINE uint64_2 mod_muli2(const uint64_2 lhs) { return (uint64_2)(mod_muli(lhs.s0), mod_muli(lhs.s1)); }
INLINE uint64_2 mod_half2(const uint64_2 lhs) { return (uint64_2)(mod_half(lhs.s0), mod_half(lhs.s1)); }

INLINE uint64_4 mod_add4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_add2(lhs.s01, rhs.s01), mod_add2(lhs.s23, rhs.s23)); }
INLINE uint64_4 mod_sub4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_sub2(lhs.s01, rhs.s01), mod_sub2(lhs.s23, rhs.s23)); }
INLINE uint64_4 mod_mul4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_mul2(lhs.s01, rhs.s01), mod_mul2(lhs.s23, rhs.s23)); }
INLINE uint64_4 mod_half4(const uint64_4 lhs) { return (uint64_4)(mod_half2(lhs.s01), mod_half2(lhs.s23)); }

// Add a carry onto the number and return the carry of the first width bits
INLINE uint32 adc(const uint64 lhs, const uint_8 width, uint64 * const carry)
{
	const uint64 s = lhs + *carry;
	const uint64 c = (s < lhs) ? 1 : 0;
	*carry = (s >> width) + (c << (64 - width));
	return (uint32)(s) & ((1u << width) - 1);
}

// Add carry and mul
INLINE uint32 adc_mul(const uint64 lhs, const uint32 a, const uint_8 width, uint64 * const carry)
{
	uint64 c = 0;
	const uint32 d = adc(lhs, width, &c);
	const uint32 r = adc((uint64)(d) * a, width, carry);
	*carry += a * c;
	return r;
}

INLINE uint64_4 adc4(const uint64_4 lhs, const uint_8_4 width, const uint64 carry)
{
	uint64_4 r;
	uint64 c = carry;
	r.s0 = adc(lhs.s0, width.s0, &c);
	r.s1 = adc(lhs.s1, width.s1, &c);
	r.s2 = adc(lhs.s2, width.s2, &c);
	r.s3 = lhs.s3 + c;
	return r;
}

INLINE uint64_4 adc_mul4(const uint64_4 lhs, const uint32 a, const uint_8_4 width, uint64 * const carry)
{
	uint64_4 r;
	r.s0 = adc_mul(lhs.s0, a, width.s0, carry);
	r.s1 = adc_mul(lhs.s1, a, width.s1, carry);
	r.s2 = adc_mul(lhs.s2, a, width.s2, carry);
	r.s3 = adc_mul(lhs.s3, a, width.s3, carry);
	return r;
}

// Subtract a carry and return the carry if borrowing
INLINE uint64 sbc(const uint64 lhs, const uint_8 width, uint32 * const carry)
{
	const bool borrow = (lhs < *carry);
	const uint64 r = lhs - *carry + (borrow ? (1u << width) : 0);
	*carry = borrow ? 1 : 0;
	return r;
}

// --- transform - inline ---

// Radix-2
#define fwd2(x, r) \
{ \
	const uint64 u0 = x.s0, u1 = mod_mul(x.s1, r); \
	x.s0 = mod_add(u0, u1); x.s1 = mod_sub(u0, u1); \
}

#define fwd2_2(x, r) \
{ \
	const uint64_2 u0 = x[0], u1 = mod_mul2(x[1], r); \
	x[0] = mod_add2(u0, u1); x[1] = mod_sub2(u0, u1); \
}

#define fwd2_4(x, r) \
{ \
	const uint64_2 u0 = x[0], u1 = mod_mul2(x[1], r.s0), u2 = x[2], u3 = mod_mul2(x[3], r.s1); \
	x[0] = mod_add2(u0, u1); x[1] = mod_sub2(u0, u1); x[2] = mod_add2(u2, u3); x[3] = mod_sub2(u2, u3); \
}

// Inverse radix-2
#define bck2(x, ri) \
{ \
	const uint64 u0 = x.s0, u1 = x.s1; \
	x.s0 = mod_add(u0, u1); x.s1 = mod_mul(mod_sub(u0, u1), ri); \
}

#define bck2_2(x, ri) \
{ \
	const uint64_2 u0 = x[0], u1 = x[1]; \
	x[0] = mod_add2(u0, u1); x[1] = mod_mul2(mod_sub2(u0, u1), ri); \
}

#define bck2_4(x, ri) \
{ \
	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3]; \
	x[0] = mod_add2(u0, u1); x[1] = mod_mul2(mod_sub2(u0, u1), ri.s0); \
	x[2] = mod_add2(u2, u3); x[3] = mod_mul2(mod_sub2(u2, u3), ri.s1); \
}

// Radix-4
#define fwd4(x, r1, r2) \
{ \
	const uint64 u0 = x[0], u2 = mod_mul(x[2], r1), u1 = mod_mul(x[1], r2.s0), u3 = mod_mul(x[3], r2.s1); \
	const uint64 v0 = mod_add(u0, u2), v2 = mod_sub(u0, u2), v1 = mod_add(u1, u3), v3 = mod_muli(mod_sub(u1, u3)); \
	x[0] = mod_add(v0, v1); x[1] = mod_sub(v0, v1); x[2] = mod_add(v2, v3); x[3] = mod_sub(v2, v3); \
}

#define fwd4_2(x, r1, r2) \
{ \
	const uint64_2 u0 = x[0], u2 = mod_mul2(x[2], r1), u1 = mod_mul2(x[1], r2.s0), u3 = mod_mul2(x[3], r2.s1); \
	const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3)); \
	x[0] = mod_add2(v0, v1); x[1] = mod_sub2(v0, v1); x[2] = mod_add2(v2, v3); x[3] = mod_sub2(v2, v3); \
}

// Inverse radix-4
#define bck4(x, ri1, ri2) \
{ \
	const uint64 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3]; \
	const uint64 v0 = mod_add(u0, u1), v1 = mod_sub(u0, u1), v2 = mod_add(u3, u2), v3 = mod_muli(mod_sub(u3, u2)); \
	x[0] = mod_add(v0, v2); x[2] = mod_mul(mod_sub(v0, v2), ri1); x[1] = mod_mul(mod_add(v1, v3), ri2.s0); x[3] = mod_mul(mod_sub(v1, v3), ri2.s1); \
}

#define bck4_2(x, ri1, ri2) \
{ \
	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3]; \
	const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2)); \
	x[0] = mod_add2(v0, v2); x[2] = mod_mul2(mod_sub2(v0, v2), ri1); x[1] = mod_mul2(mod_add2(v1, v3), ri2.s0); x[3] = mod_mul2(mod_sub2(v1, v3), ri2.s1); \
}

// squarex2 even
#define sqr2_2(x, r) \
{ \
	const uint64 t = mod_mul(mod_sqr(x.s1), r); \
	x.s1 = mod_mul(mod_add(x.s0, x.s0), x.s1); \
	x.s0 = mod_add(mod_sqr(x.s0), t); \
}

// squarex2 odd
#define sqr2n_2(x, r) \
{ \
	const uint64 t = mod_mul(mod_sqr(x.s1), r); \
	x.s1 = mod_mul(mod_add(x.s0, x.s0), x.s1); \
	x.s0 = mod_sub(mod_sqr(x.s0), t); \
}

// mulx2 even
#define mul2_2(x, y, r) \
{ \
	const uint64 t = mod_mul(mod_mul(x.s1, y.s1), r); \
	x.s1 = mod_add(mod_mul(x.s0, y.s1), mod_mul(y.s0, x.s1)); \
	x.s0 = mod_add(mod_mul(x.s0, y.s0), t); \
}

// mulx2 odd
#define mul2n_2(x, y, r) \
{ \
	const uint64 t = mod_mul(mod_mul(x.s1, y.s1), r); \
	x.s1 = mod_add(mod_mul(x.s0, y.s1), mod_mul(y.s0, x.s1)); \
	x.s0 = mod_sub(mod_mul(x.s0, y.s0), t); \
}

// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.
#define butterfly5_2(a0, a1, a2, a3, a4) \
{ \
	const uint64_2 s1 = mod_add2(a1, a4), s2 = mod_sub2(a1, a4), s3 = mod_add2(a3, a2), s4 = mod_sub2(a3, a2); \
	const uint64_2 s5 = mod_add2(s1, s3), s6 = mod_sub2(s1, s3), s7 = mod_add2(s2, s4), s8 = mod_add2(s5, a0); \
	const uint64_2 m0 = s8; \
	const uint64_2 m1 = mod_mul2(s5, W_F1), m2 = mod_mul2(s6, W_F2), m3 = mod_mul2(s2, W_F3), m4 = mod_mul2(s7, W_F4), m5 = mod_mul2(s4, W_F5); \
	const uint64_2 s9 = mod_add2(m0, m1), s10 = mod_add2(s9, m2), s11 = mod_sub2(s9, m2), s12 = mod_sub2(m3, m4); \
	const uint64_2 s13 = mod_add2(m4, m5), s14 = mod_add2(s10, s12), s15 = mod_sub2(s10, s12), s16 = mod_add2(s11, s13); \
	const uint64_2 s17 = mod_sub2(s11, s13); \
	a0 = m0; a1 = s14; a2 = s16; a3 = s17; a4 = s15; \
}

// Radix-5
#define fwd5_2(x, r) \
{ \
	const uint64_2 r2 = mod_sqr2(r), r3 = mod_mul2(r, r2), r4 = mod_sqr2(r2); \
	uint64_2 a0 = x[0], a1 = mod_mul2(x[1], r), a2 = mod_mul2(x[2], r2), a3 = mod_mul2(x[3], r3), a4 = mod_mul2(x[4], r4); \
	butterfly5_2(a0, a1, a2, a3, a4); \
	x[0] = a0; x[1] = a1; x[2] = a2; x[3] = a3; x[4] = a4; \
}

// Inverse radix-5
#define bck5_2(x, ri) \
{ \
	uint64_2 a0 = x[0], a4 = x[1], a3 = x[2], a2 = x[3], a1 = x[4]; \
	butterfly5_2(a0, a1, a2, a3, a4); \
	const uint64_2 ri2 = mod_sqr2(ri), ri3 = mod_mul2(ri, ri2), ri4 = mod_sqr2(ri2); \
	x[0] = a0; x[1] = mod_mul2(a1, ri); x[2] = mod_mul2(a2, ri2); x[3] = mod_mul2(a3, ri3); x[4] = mod_mul2(a4, ri4); \
}

// Transpose of matrices
#define transpose_52(y, x) \
{ \
	y[0] = (uint64_2)(x[0].s0, x[2].s1); \
	y[1] = (uint64_2)(x[0].s1, x[3].s0); \
	y[2] = (uint64_2)(x[1].s0, x[3].s1); \
	y[3] = (uint64_2)(x[1].s1, x[4].s0); \
	y[4] = (uint64_2)(x[2].s0, x[4].s1); \
}

#define transpose_25(y, x) \
{ \
	y[0] = (uint64_2)(x[0].s0, x[1].s0); \
	y[1] = (uint64_2)(x[2].s0, x[3].s0); \
	y[2] = (uint64_2)(x[4].s0, x[0].s1); \
	y[3] = (uint64_2)(x[1].s1, x[2].s1); \
	y[4] = (uint64_2)(x[3].s1, x[4].s1); \
}

INLINE void loadg1(const sz_t n, uint64 * const xl, __global const uint64 * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
INLINE void storeg1(const sz_t n, __global uint64 * restrict const x, const sz_t s, const uint64 * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }

INLINE void loadg2(const sz_t n, uint64_2 * const xl, __global const uint64_2 * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
INLINE void loadl2(const sz_t n, uint64_2 * const xl, __local const uint64_2 * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }
INLINE void storeg2(const sz_t n, __global uint64_2 * restrict const x, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }
INLINE void storel2(const sz_t n, __local uint64_2 * restrict const X, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }

// --- transform - global mem ---

#if N_SZ % 5 != 0

// 2 x Radix-4
__kernel
void forward4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 2]);

	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);
	const uint64 r1 = r2[j]; const uint64_2 r23 = r4[j];
	fwd4_2(xl, r1, r23);
	storeg2(4, &x[k], m, xl);
}

// 2 x Inverse radix-4
__kernel
void backward4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2i = &root[N_SZ];
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 2]);

	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);
	const uint64 r1i = r2i[j]; const uint64_2 r23i = r4i[j];
	bck4_2(xl, r1i, r23i);
	storeg2(4, &x[k], m, xl);
}

// Radix-2
__kernel
void forward_mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r0 = &root[0];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;

	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);
	const uint64 r = r0[j]; fwd2_2(xl, r);
	storeg2(2, &x[k], 1, xl);
}

// Radix-2, square2x2, inverse radix-2
__kernel
void sqr4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r0 = &root[0];
	__global const uint64 * restrict const r0i = &root[N_SZ];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;

	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);
	const uint64 r = r0[j]; fwd2_2(xl, r);
	sqr2_2(xl[0], r); sqr2n_2(xl[1], r);
	const uint64 ri = r0i[j]; bck2_2(xl, ri);
	storeg2(2, &x[k], 1, xl);
}

// Radix-2, mul2x2, inverse radix-2
__kernel
void mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);
	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);
	__global const uint64 * restrict const r0 = &root[0];
	__global const uint64 * restrict const r0i = &root[N_SZ];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;

	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);
	const uint64 r = r0[j]; fwd2_2(xl, r);
	uint64_2 yl[2]; loadg2(2, yl, &y[k], 1);
	mul2_2(xl[0], yl[0], r); mul2n_2(xl[1], yl[1], r);
	const uint64 ri = r0i[j]; bck2_2(xl, ri);
	storeg2(2, &x[k], 1, xl);
}

// 2 x Radix-2
__kernel
void forward_mul4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	const uint64_2 r = r0[j]; fwd2_4(xl, r);
	storeg2(4, &x[k], 1, xl);
}

// 2 x Radix-2, square2x2, inverse radix-2
__kernel
void sqr4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0];
	__global const uint64_2 * restrict const r0i = (__global const uint64_2 *)&root[N_SZ];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	const uint64_2 r = r0[j]; fwd2_4(xl, r);
	sqr2_2(xl[0], r.s0); sqr2n_2(xl[1], r.s0); sqr2_2(xl[2], r.s1); sqr2n_2(xl[3], r.s1);
	const uint64_2 ri = r0i[j]; bck2_4(xl, ri);
	storeg2(4, &x[k], 1, xl);
}

// 2 x Radix-2, mul2x2, inverse radix-2
__kernel
void mul4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);
	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);
	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0];
	__global const uint64_2 * restrict const r0i = (__global const uint64_2 *)&root[N_SZ];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	const uint64_2 r = r0[j]; fwd2_4(xl, r);
	uint64_2 yl[4]; loadg2(4, yl, &y[k], 1);
	mul2_2(xl[0], yl[0], r.s0); mul2n_2(xl[1], yl[1], r.s0); mul2_2(xl[2], yl[2], r.s1); mul2n_2(xl[3], yl[3], r.s1);
	const uint64_2 ri = r0i[j]; bck2_4(xl, ri);
	storeg2(4, &x[k], 1, xl);
}

#else

// 2 x Radix-4, 5 | N_SZ
__kernel
void forward4x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 5 / 2]);

	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), m = 1u << lm, m5 = 5u << lm;
	const sz_t j = id_5 >> lm, k = 3 * 5 * (id_5 & ~(m - 1)) + id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], m5);
	const uint64 r1 = r2[j]; const uint64_2 r23 = r4[j];
	fwd4_2(xl, r1, r23);
	storeg2(4, &x[k], m5, xl);
}

// 2 x Inverse radix-4, 5 | N_SZ
__kernel
void backward4x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2i = &root[N_SZ];
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5 / 2]);

	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), m = 1u << lm, m5 = 5u << lm;
	const sz_t j = id_5 >> lm, k = 3 * 5 * (id_5 & ~(m - 1)) + id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], m5);
	const uint64 r1i = r2i[j]; const uint64_2 r23i = r4i[j];
	bck4_2(xl, r1i, r23i);
	storeg2(4, &x[k], m5, xl);
}

/*

// Radix-2, radix-5
__kernel
void forward_mul10(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]);

	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;
	if (id_mod5 != 4)
	{
		const sz_t id4 = 4 * id_5 + id_mod5, j = id4, k = 5 * id4;

		uint64_2 xl[5], xt[5]; loadg2(5, xl, &x[k], 1); transpose_52(xt, xl);
		const uint64 r_2 = r2[j]; for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r_2);
		const uint64_2 r_5 = r5[j]; fwd5_2(xt, r_5);
		transpose_25(xl, xt); storeg2(5, &x[k], 1, xl);
	}
}

// Radix-2, radix-5, square, inverse radix-5, inverse radix-2
__kernel
void sqr10(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64 * restrict const r2i = &root[N_SZ];
	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]);
	__global const uint64_2 * restrict const r5i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5]);

	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;
	if (id_mod5 != 4)
	{
		const sz_t id4 = 4 * id_5 + id_mod5, j = id4, k = 5 * id4;

		uint64_2 xl[5], xt[5]; loadg2(5, xl, &x[k], 1); transpose_52(xt, xl);
		const uint64 r = r2[j]; for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r);
		const uint64_2 r_5 = r5[j]; fwd5_2(xt, r_5);
		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_sqr2(xt[i]);
		const uint64_2 r_5i = r5i[j]; bck5_2(xt, r_5i);
		const uint64 ri = r2i[j]; for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], ri);
		transpose_25(xl, xt); storeg2(5, &x[k], 1, xl);
	}
}

// Radix-2, radix-5, mul, inverse radix-5, inverse radix-2
__kernel
void mul10(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);
	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64 * restrict const r2i = &root[N_SZ];
	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]);
	__global const uint64_2 * restrict const r5i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5]);

	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;
	if (id_mod5 != 4)
	{
		const sz_t id4 = 4 * id_5 + id_mod5, j = id4, k = 5 * id4;

		uint64_2 xl[5], xt[5]; loadg2(5, xl, &x[k], 1); transpose_52(xt, xl);
		const uint64 r = r2[j]; for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r);
		const uint64_2 r_5 = r5[j]; fwd5_2(xt, r_5);
		uint64_2 yl[5], yt[5]; loadg2(5, yl, &y[k], 1); transpose_52(yt, yl);
		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_mul2(xt[i], yt[i]);
		const uint64_2 r_5i = r5i[j]; bck5_2(xt, r_5i);
		const uint64 ri = r2i[j]; for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], ri);
		transpose_25(xl, xt); storeg2(5, &x[k], 1, xl);
	}
}
*/

#endif

// --- transform - local mem ---

INLINE void forward_4i(const sz_t ml, __local uint64_2 * restrict const X,
	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1, const uint64_2 r23)
{
	uint64_2 xl[4]; loadg2(4, xl, x, mg);
	fwd4_2(xl, r1, r23);
	storel2(4, X, ml, xl);
}

INLINE void forward_4(const sz_t ml, __local uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	fwd4_2(xl, r1, r23);
	storel2(4, X, ml, xl);
}

INLINE void forward_4o(const sz_t mg, __global uint64_2 * restrict const x,
	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	fwd4_2(xl, r1, r23);
	storeg2(4, x, mg, xl);
}

INLINE void backward_4i(const sz_t ml, __local uint64_2 * restrict const X,
	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1i, const uint64_2 r23i)
{
	uint64_2 xl[4]; loadg2(4, xl, x, mg);
	bck4_2(xl, r1i, r23i);
	storel2(4, X, ml, xl);
}

INLINE void backward_4(const sz_t ml, __local uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	bck4_2(xl, r1i, r23i);
	storel2(4, X, ml, xl);
}

INLINE void backward_4o(const sz_t mg, __global uint64_2 * restrict const x,
	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	bck4_2(xl, r1i, r23i);
	storeg2(4, x, mg, xl);
}

INLINE void forward_4x2o(__global uint64_2 * restrict const x, __local uint64_2 * restrict const X, const uint64_2 r)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	fwd2_4(xl, r);
	storeg2(4, x, 1, xl);
}

INLINE void forward_10o(__global uint64_2 * restrict const x, __local uint64_2 * restrict const X, const uint64 r2, const uint64_2 r5, const bool cond)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	if (cond)
	{
		uint64_2 xl[5], xt[5]; loadl2(5, xl, X, 1); transpose_52(xt, xl);
		for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r2);
		fwd5_2(xt, r5);
		transpose_25(xl, xt); storeg2(5, x, 1, xl);
	}
}

INLINE void square_4x2(__local uint64_2 * restrict const X, const uint64_2 r, const uint64_2 ri)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	fwd2_4(xl, r);
	sqr2_2(xl[0], r.s0); sqr2n_2(xl[1], r.s0); sqr2_2(xl[2], r.s1); sqr2n_2(xl[3], r.s1);
	bck2_4(xl, ri);
	storel2(4, X, 1, xl);
}

INLINE void mul_4x2(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y, const uint64_2 r, const uint64_2 ri)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	fwd2_4(xl, r);
	uint64_2 yl[4]; loadg2(4, yl, y, 1);
	mul2_2(xl[0], yl[0], r.s0); mul2n_2(xl[1], yl[1], r.s0); mul2_2(xl[2], yl[2], r.s1); mul2n_2(xl[3], yl[3], r.s1);
	bck2_4(xl, ri);
	storel2(4, X, 1, xl);
}

INLINE void square_10(__local uint64_2 * restrict const X, const uint64 r2, const uint64 r2i, const uint64_2 r5, const uint64_2 r5i, const bool cond)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	if (cond)
	{
		uint64_2 xl[5], xt[5]; loadl2(5, xl, X, 1); transpose_52(xt, xl);
		for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r2);
		fwd5_2(xt, r5);
		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_sqr2(xt[i]);
		bck5_2(xt, r5i);
		for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], r2i);
		transpose_25(xl, xt); storel2(5, X, 1, xl);
	}
}

INLINE void mul_10(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y,
	const uint64 r2, const uint64 r2i, const uint64_2 r5, const uint64_2 r5i, const bool cond)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	if (cond)
	{
		uint64_2 xl[5], xt[5]; loadl2(5, xl, X, 1); transpose_52(xt, xl);
		for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r2);
		fwd5_2(xt, r5);
		uint64_2 yl[5], yt[5]; loadg2(5, yl, y, 1); transpose_52(yt, yl);
		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_mul2(xt[i], yt[i]);
		bck5_2(xt, r5i);
		for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], r2i);
		transpose_25(xl, xt); storel2(5, X, 1, xl);
	}
}

#if N_SZ % 5 != 0

#define DECLARE_VAR_REG() \
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]); \
	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0]; \
	__global const uint64_2 * restrict const r0i = (__global const uint64_2 *)&root[N_SZ]; \
	__global const uint64 * restrict const r2 = &root[0]; \
	__global const uint64 * restrict const r2i = &root[N_SZ]; \
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 2]); \
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 2]); \
	const sz_t id = (sz_t)get_global_id(0);

#define DECLARE_VAR(B_N, CHUNK_N) \
	DECLARE_VAR_REG(); \
	\
	__local uint64_2 X[4 * B_N * CHUNK_N]; \
	\
	/* thread_idx < B_N */ \
	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \
	const sz_t i = local_id, chunk_idx = i % CHUNK_N, thread_idx = i / CHUNK_N, block_idx = group_id * CHUNK_N + chunk_idx; \
	__local uint64_2 * const Xi = &X[chunk_idx]; \
	\
	const sz_t block_idx_m = block_idx >> lm, idx_m = block_idx_m * B_N + thread_idx; \
	const sz_t block_idx_mm = block_idx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = block_idx + block_idx_mm * (B_N * 3 - 1) + idx_mm, ko = block_idx - block_idx_mm + idx_mm * 4; \
	const sz_t j = idx_m;

#if MAX_WG_SZ >= 16 / 4 * CHUNK16
#define ATTR_FB_16x2() \
	__attribute__((reqd_work_group_size(16 / 4 * CHUNK16, 1, 1)))
#else
#define ATTR_FB_16x2()
#endif

// 2 x Radix-4, radix-4
__kernel
ATTR_FB_16x2()
void forward16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(16 / 4, CHUNK16);

	forward_4i(4 * CHUNK16, &X[i], 4u << lm, &x[ki], r2[j / 4], r4[j / 4]);
	forward_4o(1u << lm, &x[ko], 1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], r2[j], r4[j]);
}

// 2 x Inverse radix-4, radix-4
__kernel
ATTR_FB_16x2()
void backward16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(16 / 4, CHUNK16);

	backward_4i(1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);
	backward_4o(4u << lm, &x[ki], 4 * CHUNK16, &X[i], r2i[j / 4], r4i[j / 4]);
}

#if MAX_WG_SZ >= 64 / 4 * CHUNK64
#define ATTR_FB_64x2() \
	__attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))
#else
#define ATTR_FB_64x2()
#endif

__kernel
ATTR_FB_64x2()
void forward64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(64 / 4, CHUNK64);

	forward_4i(16 * CHUNK64, &X[i], 16u << lm, &x[ki], r2[j / 16], r4[j / 16]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	forward_4(4 * CHUNK64, &Xi[CHUNK64 * i4], r2[j / 4], r4[j / 4]);
	forward_4o(1u << lm, &x[ko], 1 * CHUNK64, &Xi[CHUNK64 * 4 * thread_idx], r2[j], r4[j]);
}

__kernel
ATTR_FB_64x2()
void backward64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(64 / 4, CHUNK64);

	backward_4i(1 * CHUNK64, &Xi[CHUNK64 * 4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	backward_4(4 * CHUNK64, &Xi[CHUNK64 * i4], r2i[j / 4], r4i[j / 4]);
	backward_4o(16u << lm, &x[ki], 16 * CHUNK64, &X[i], r2i[j / 16], r4i[j / 16]);
}

#if MAX_WG_SZ >= 256 / 4 * CHUNK256
#define ATTR_FB_256x2() \
	__attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))
#else
#define ATTR_FB_256x2()
#endif

__kernel
ATTR_FB_256x2()
void forward256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(256 / 4, CHUNK256);

	forward_4i(64 * CHUNK256, &X[i], 64u << lm, &x[ki], r2[j / 64], r4[j / 64]);
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);
	forward_4(16 * CHUNK256, &Xi[CHUNK256 * i16], r2[j / 16], r4[j / 16]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	forward_4(4 * CHUNK256, &Xi[CHUNK256 * i4], r2[j / 4], r4[j / 4]);
	forward_4o(1u << lm, &x[ko], 1 * CHUNK256, &Xi[CHUNK256 * 4 * thread_idx], r2[j], r4[j]);
}

__kernel
ATTR_FB_256x2()
void backward256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(256 / 4, CHUNK256);

	backward_4i(1 * CHUNK256, &Xi[CHUNK256 * 4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	backward_4(4 * CHUNK256, &Xi[CHUNK256 * i4], r2i[j / 4], r4i[j / 4]);
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);
	backward_4(16 * CHUNK256, &Xi[CHUNK256 * i16], r2i[j / 16], r4i[j / 16]);
	backward_4o(64u << lm, &x[ki], 64 * CHUNK256, &X[i], r2i[j / 64], r4i[j / 64]);
}

#if MAX_WG_SZ >= 1024 / 4
#define ATTR_FB_1024x2() \
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#else
#define ATTR_FB_1024x2()
#endif

__kernel
ATTR_FB_1024x2()
void forward1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(1024 / 4, 1);

	forward_4i(256, &X[i], 256u << lm, &x[ki], r2[j / 256], r4[j / 256]);
	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64);
	forward_4(64, &Xi[i64], r2[j / 64], r4[j / 64]);
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);
	forward_4(16, &Xi[i16], r2[j / 16], r4[j / 16]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	forward_4(4, &Xi[i4], r2[j / 4], r4[j / 4]);
	forward_4o(1u << lm, &x[ko], 1, &Xi[4 * thread_idx], r2[j], r4[j]);
}

__kernel
ATTR_FB_1024x2()
void backward1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR(1024 / 4, 1);

	backward_4i(1, &Xi[4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	backward_4(4, &Xi[i4], r2i[j / 4], r4i[j / 4]);
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);
	backward_4(16, &Xi[i16], r2i[j / 16], r4i[j / 16]);
	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64);
	backward_4(64, &Xi[i64], r2i[j / 64], r4i[j / 64]);
	backward_4o(256u << lm, &x[ki], 256, &X[i], r2i[j / 256], r4i[j / 256]);
}

////////////////////////////////////

#define DECLARE_VAR_16x2() \
	__local uint64_2 X[16 * BLK16]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, k = 4 * id, i = k % (16 * BLK16); \
	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % (16 * BLK16);

#if MAX_WG_SZ >= 16 / 4 * BLK16
#define ATTR_16x2() \
	__attribute__((reqd_work_group_size(16 / 4 * BLK16, 1, 1)))
#else
#define ATTR_16x2()
#endif

// 2 x Radix-4, 2 x radix-2
__kernel
ATTR_16x2()
void forward_mul16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_16x2();

	forward_4i(2, &X[i4], 2, &x[k4], r2[j4], r4[j4]);
	forward_4x2o(&x[k], &X[i], r0[j]);
}

// 2 x Radix-4, square4, inverse radix-4
__kernel
ATTR_16x2()
void sqr16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_16x2();

	forward_4i(2, &X[i4], 2, &x[k4], r2[j4], r4[j4]);
	square_4x2(&X[i], r0[j], r0i[j]);
	backward_4o(2, &x[k4], 2, &X[i4], r2i[j4], r4i[j4]);
}

// 2 x Radix-4, mul4, inverse radix-4
__kernel
ATTR_16x2()
void mul16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_16x2();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(2, &X[i4], 2, &x[k4], r2[j4], r4[j4]);
	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);
	backward_4o(2, &x[k4], 2, &X[i4], r2i[j4], r4i[j4]);
}

#define DECLARE_VAR_64x2() \
	__local uint64_2 X[64 * BLK64]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, k = 4 * id, i = k % (64 * BLK64); \
	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % (64 * BLK64); \
	const sz_t j16 = id / 8, k16 = 4 * (id & ~(8 - 1)) + (id % 8), i16 = k16 % (64 * BLK64);

#if MAX_WG_SZ >= 64 / 4 * BLK64
#define ATTR_64x2() \
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#else
#define ATTR_64x2()
#endif

__kernel
ATTR_64x2()
void forward_mul64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_64x2();

	forward_4i(8, &X[i16], 8, &x[k16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	forward_4x2o(&x[k], &X[i], r0[j]);
}

__kernel
ATTR_64x2()
void sqr64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_64x2();

	forward_4i(8, &X[i16], 8, &x[k16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	square_4x2(&X[i], r0[j], r0i[j]);
	backward_4(2, &X[i4], r2i[j4], r4i[j4]);
	backward_4o(8, &x[k16], 8, &X[i16], r2i[j16], r4i[j16]);
}

__kernel
ATTR_64x2()
void mul64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_64x2();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(8, &X[i16], 8, &x[k16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);
	backward_4(2, &X[i4], r2i[j4], r4i[j4]);
	backward_4o(8, &x[k16], 8, &X[i16], r2i[j16], r4i[j16]);
}

#define DECLARE_VAR_256x2() \
	__local uint64_2 X[256]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, k = 4 * id, i = k % 256; \
	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % 256; \
	const sz_t j16 = id / 8, k16 = 4 * (id & ~(8 - 1)) + (id % 8), i16 = k16 % 256; \
	const sz_t j64 = id / 32, k64 = 4 * (id & ~(32 - 1)) + (id % 32), i64 = k64 % 256;

#if MAX_WG_SZ >= 256 / 4
#define ATTR_256x2() \
	__attribute__((reqd_work_group_size(256 / 4, 1, 1)))
#else
#define ATTR_256x2()
#endif

__kernel
ATTR_256x2()
void forward_mul256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_256x2();

	forward_4i(32, &X[i64], 32, &x[k64], r2[j64], r4[j64]);
	forward_4(8, &X[i16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	forward_4x2o(&x[k], &X[i], r0[j]);
}

__kernel
ATTR_256x2()
void sqr256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_256x2();

	forward_4i(32, &X[i64], 32, &x[k64], r2[j64], r4[j64]);
	forward_4(8, &X[i16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	square_4x2(&X[i], r0[j], r0i[j]);
	backward_4(2, &X[i4], r2i[j4], r4i[j4]);
	backward_4(8, &X[i16], r2i[j16], r4i[j16]);
	backward_4o(32, &x[k64], 32, &X[i64], r2i[j64], r4i[j64]);
}

__kernel
ATTR_256x2()
void mul256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_256x2();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(32, &X[i64], 32, &x[k64], r2[j64], r4[j64]);
	forward_4(8, &X[i16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);
	backward_4(2, &X[i4], r2i[j4], r4i[j4]);
	backward_4(8, &X[i16], r2i[j16], r4i[j16]);
	backward_4o(32, &x[k64], 32, &X[i64], r2i[j64], r4i[j64]);
}

#define DECLARE_VAR_1024x2() \
	__local uint64_2 X[1024]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, k = 4 * id, i = k % 1024; \
	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % 1024; \
	const sz_t j16 = id / 8, k16 = 4 * (id & ~(8 - 1)) + (id % 8), i16 = k16 % 1024; \
	const sz_t j64 = id / 32, k64 = 4 * (id & ~(32 - 1)) + (id % 32), i64 = k64 % 1024; \
	const sz_t j256 = id / 128, k256 = 4 * (id & ~(128 - 1)) + (id % 128), i256 = k256 % 1024;

#if MAX_WG_SZ >= 1024 / 4
#define ATTR_1024x2() \
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#else
#define ATTR_1024x2()
#endif

__kernel
ATTR_1024x2()
void forward_mul1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_1024x2();

	forward_4i(128, &X[i256], 128, &x[k256], r2[j256], r4[j256]);
	forward_4(32, &X[i64], r2[j64], r4[j64]);
	forward_4(8, &X[i16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	forward_4x2o(&x[k], &X[i], r0[j]);
}

__kernel
ATTR_1024x2()
void sqr1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_1024x2();

	forward_4i(128, &X[i256], 128, &x[k256], r2[j256], r4[j256]);
	forward_4(32, &X[i64], r2[j64], r4[j64]);
	forward_4(8, &X[i16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	square_4x2(&X[i], r0[j], r0i[j]);
	backward_4(2, &X[i4], r2i[j4], r4i[j4]);
	backward_4(8, &X[i16], r2i[j16], r4i[j16]);
	backward_4(32, &X[i64], r2i[j64], r4i[j64]);
	backward_4o(128, &x[k256], 128, &X[i256], r2i[j256], r4i[j256]);
}

__kernel
ATTR_1024x2()
void mul1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_1024x2();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(128, &X[i256], 128, &x[k256], r2[j256], r4[j256]);
	forward_4(32, &X[i64], r2[j64], r4[j64]);
	forward_4(8, &X[i16], r2[j16], r4[j16]);
	forward_4(2, &X[i4], r2[j4], r4[j4]);
	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);
	backward_4(2, &X[i4], r2i[j4], r4i[j4]);
	backward_4(8, &X[i16], r2i[j16], r4i[j16]);
	backward_4(32, &X[i64], r2i[j64], r4i[j64]);
	backward_4o(128, &x[k256], 128, &X[i256], r2i[j256], r4i[j256]);
}

#else

#define DECLARE_VAR_REG_5() \
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]); \
	__global const uint64 * restrict const r2 = &root[0]; \
	__global const uint64 * restrict const r2i = &root[N_SZ]; \
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 5 / 2]); \
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5 / 2]); \
	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;

#define DECLARE_VAR5(B_N, CHUNK_N) \
	DECLARE_VAR_REG_5(); \
	\
	__local uint64_2 X[5 * 4 * B_N * CHUNK_N]; \
	\
	/* thread_idx < B_N */ \
	const sz_t local_id = id_5 % (B_N * CHUNK_N), group_id = id_5 / (B_N * CHUNK_N); \
	const sz_t i = local_id, chunk_idx = i % CHUNK_N, thread_idx = i / CHUNK_N, block_idx = group_id * CHUNK_N + chunk_idx; \
	__local uint64_2 * const Xi = &X[5 * chunk_idx]; \
	\
	const sz_t block_idx_m = block_idx >> lm, idx_m = block_idx_m * B_N + thread_idx; \
	const sz_t block_idx_mm = block_idx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = 5 * (block_idx + block_idx_mm * (B_N * 3 - 1) + idx_mm) + id_mod5; \
	const sz_t ko = 5 * (block_idx - block_idx_mm + idx_mm * 4) + id_mod5; \
	const sz_t j = idx_m;

#if MAX_WG_SZ >= 5 * 16 / 4 * CHUNK16_5
#define ATTR_FB_16x2_5() \
	__attribute__((reqd_work_group_size(5 * 16 / 4 * CHUNK16_5, 1, 1)))
#else
#define ATTR_FB_16x2_5()
#endif

// 2 x Radix-4, radix-4, 5 | N_SZ
__kernel
ATTR_FB_16x2_5()
void forward16x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR5(16 / 4, CHUNK16_5);

	forward_4i(20 * CHUNK16_5, &X[5 * i + id_mod5], 20u << lm, &x[ki], r2[j / 4], r4[j / 4]);
	forward_4o(5u << lm, &x[ko], 5 * CHUNK16_5, &Xi[5 * CHUNK16_5 * 4 * thread_idx + id_mod5], r2[j], r4[j]);
}

// 2 x Inverse radix-4, radix-4, 5 | N_SZ
__kernel
ATTR_FB_16x2_5()
void backward16x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR5(16 / 4, CHUNK16_5);

	backward_4i(5 * CHUNK16_5, &Xi[5 * CHUNK16_5 * 4 * thread_idx + id_mod5], 5u << lm, &x[ko], r2i[j], r4i[j]);
	backward_4o(20u << lm, &x[ki], 20 * CHUNK16_5, &X[5 * i + id_mod5], r2i[j / 4], r4i[j / 4]);
}

#if MAX_WG_SZ >= 5 * 64 / 4 * CHUNK64_5
#define ATTR_FB_64x2_5() \
	__attribute__((reqd_work_group_size(5 * 64 / 4 * CHUNK64_5, 1, 1)))
#else
#define ATTR_FB_64x2_5()
#endif

__kernel
ATTR_FB_64x2_5()
void forward64x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR5(64 / 4, CHUNK64_5);

	forward_4i(80 * CHUNK64_5, &X[5 * i + id_mod5], 80u << lm, &x[ki], r2[j / 16], r4[j / 16]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	forward_4(20 * CHUNK64_5, &Xi[5 * CHUNK64_5 * i4 + id_mod5], r2[j / 4], r4[j / 4]);
	forward_4o(5u << lm, &x[ko], 5 * CHUNK64_5, &Xi[5 * CHUNK64_5 * 4 * thread_idx + id_mod5], r2[j], r4[j]);
}

__kernel
ATTR_FB_64x2_5()
void backward64x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR5(64 / 4, CHUNK64_5);

	backward_4i(5 * CHUNK64_5, &Xi[5 * CHUNK64_5 * 4 * thread_idx + id_mod5], 5u << lm, &x[ko], r2i[j], r4i[j]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	backward_4(20 * CHUNK64_5, &Xi[5 * CHUNK64_5 * i4 + id_mod5], r2i[j / 4], r4i[j / 4]);
	backward_4o(80u << lm, &x[ki], 80 * CHUNK64_5, &X[5 * i + id_mod5], r2i[j / 16], r4i[j / 16]);
}

#if MAX_WG_SZ >= 5 * 256 / 4
#define ATTR_FB_256x2_5() \
	__attribute__((reqd_work_group_size(5 * 256 / 4, 1, 1)))
#else
#define ATTR_FB_256x2_5()
#endif

__kernel
ATTR_FB_256x2_5()
void forward256x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR5(256 / 4, 1);

	forward_4i(320, &X[5 * i + id_mod5], 320u << lm, &x[ki], r2[j / 64], r4[j / 64]);
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);
	forward_4(80, &Xi[5 * i16 + id_mod5], r2[j / 16], r4[j / 16]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	forward_4(20, &Xi[5 * i4 + id_mod5], r2[j / 4], r4[j / 4]);
	forward_4o(5u << lm, &x[ko], 5, &Xi[5 * (4 * thread_idx) + id_mod5], r2[j], r4[j]);
}

__kernel
ATTR_FB_256x2_5()
void backward256x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)
{
	DECLARE_VAR5(256 / 4, 1);

	backward_4i(5, &Xi[5 * 4 * thread_idx + id_mod5], 5u << lm, &x[ko], r2i[j], r4i[j]);
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);
	backward_4(20, &Xi[5 * i4 + id_mod5], r2i[j / 4], r4i[j / 4]);
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);
	backward_4(80, &Xi[5 * i16 + id_mod5], r2i[j / 16], r4i[j / 16]);
	backward_4o(320u << lm, &x[ki], 320, &X[5 * i + id_mod5], r2i[j / 64], r4i[j / 64]);
}

////////////////////////////////////

#define DECLARE_VAR_REG5() \
	DECLARE_VAR_REG_5(); \
	__global const uint64 * restrict const r0 = &root[0]; \
	__global const uint64 * restrict const r0i = &root[N_SZ]; \
	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]); \
	__global const uint64_2 * restrict const r5i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5]); \
	const sz_t local_id = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);

#define WGSIZE40	(40 / 8 * BLK40)

#define DECLARE_VAR_40() \
	__local uint64_2 X[20 * BLK40]; \
	\
	DECLARE_VAR_REG5(); \
	const sz_t lid4 = local_id, id4 = 4 * WGSIZE40 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \
	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, k1 = 5 * t1 + id_mod5, i1 = 5 * (t1 % (20 * BLK40 / 5)) + id_mod5;

#if MAX_WG_SZ >= WGSIZE40
#define ATTR_40() \
	__attribute__((reqd_work_group_size(WGSIZE40, 1, 1)))
#else
#define ATTR_40()
#endif

// Radix-4, radix-2, radix-5
__kernel
ATTR_40()
void forward_mul40(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_40();

	forward_4i(1 * 5, &X[i1], 1 * 5, &x[k1], r2[j1], r4[j1]);
	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE40 / 5);
}

// Radix-4, radix-2, radix-5, square, inverse radix-5, inverse radix-2, inverse radix-4
__kernel
ATTR_40()
void sqr40(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_40();

	forward_4i(1 * 5, &X[i1], 1 * 5, &x[k1], r2[j1], r4[j1]);
	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE40 / 5);
	backward_4o(1 * 5, &x[k1], 1 * 5, &X[i1], r2i[j1], r4i[j1]);
}

// Radix-4, radix-2, radix-5, mul, inverse radix-5, inverse radix-2, inverse radix-4
__kernel
ATTR_40()
void mul40(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_40();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(1 * 5, &X[i1], 1 * 5, &x[k1], r2[j1], r4[j1]);
	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE40 / 5);
	backward_4o(1 * 5, &x[k1], 1 * 5, &X[i1], r2i[j1], r4i[j1]);
}

#define WGSIZE160	(160 / 8 * BLK160)

#define DECLARE_VAR_160() \
	__local uint64_2 X[80 * BLK160]; \
	\
	DECLARE_VAR_REG5(); \
	const sz_t lid4 = local_id, id4 = 4 * WGSIZE160 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \
	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, /*k1 = 5 * t1 + id_mod5,*/ i1 = 5 * (t1 % (80 * BLK160 / 5)) + id_mod5; \
	const sz_t j4 = id_5 / 4, t4 = 4 * (id_5 & ~(4 - 1)) + (id_5 % 4), k4 = 5 * t4 + id_mod5, i4 = 5 * (t4 % (80 * BLK160 / 5)) + id_mod5;

#if MAX_WG_SZ >= WGSIZE160
#define ATTR_160() \
	__attribute__((reqd_work_group_size(WGSIZE160, 1, 1)))
#else
#define ATTR_160()
#endif

__kernel
ATTR_160()
void forward_mul160(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_160();

	forward_4i(4 * 5, &X[i4], 4 * 5, &x[k4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE160 / 5);
}

__kernel
ATTR_160()
void sqr160(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_160();

	forward_4i(4 * 5, &X[i4], 4 * 5, &x[k4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE160 / 5);
	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);
	backward_4o(4 * 5, &x[k4], 4 * 5, &X[i4], r2i[j4], r4i[j4]);
}

__kernel
ATTR_160()
void mul160(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_160();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(4 * 5, &X[i4], 4 * 5, &x[k4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE160 / 5);
	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);
	backward_4o(4 * 5, &x[k4], 4 * 5, &X[i4], r2i[j4], r4i[j4]);
}

#define WGSIZE640	(640 / 8 * BLK640)

#define DECLARE_VAR_640() \
	__local uint64_2 X[320 * BLK640]; \
	\
	DECLARE_VAR_REG5(); \
	const sz_t lid4 = local_id, id4 = 4 * WGSIZE640 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \
	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, /*k1 = 5 * t1 + id_mod5,*/ i1 = 5 * (t1 % (320 * BLK640 / 5)) + id_mod5; \
	const sz_t j4 = id_5 / 4, t4 = 4 * (id_5 & ~(4 - 1)) + (id_5 % 4), /*k4 = 5 * t4 + id_mod5,*/ i4 = 5 * (t4 % (320 * BLK640 / 5)) + id_mod5; \
	const sz_t j16 = id_5 / 16, t16 = 4 * (id_5 & ~(16 - 1)) + (id_5 % 16), k16 = 5 * t16 + id_mod5, i16 = 5 * (t16 % (320 * BLK640 / 5)) + id_mod5;

#if MAX_WG_SZ >= WGSIZE640
#define ATTR_640() \
	__attribute__((reqd_work_group_size(WGSIZE640, 1, 1)))
#else
#define ATTR_640()
#endif

__kernel
ATTR_640()
void forward_mul640(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_640();

	forward_4i(16 * 5, &X[i16], 16 * 5, &x[k16], r2[j16], r4[j16]);
	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE640 / 5);
}

__kernel
ATTR_640()
void sqr640(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_640();

	forward_4i(16 * 5, &X[i16], 16 * 5, &x[k16], r2[j16], r4[j16]);
	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE640 / 5);
	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);
	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);
	backward_4o(16 * 5, &x[k16], 16 * 5, &X[i16], r2i[j16], r4i[j16]);
}

__kernel
ATTR_640()
void mul640(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_640();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(16 * 5, &X[i16], 16 * 5, &x[k16], r2[j16], r4[j16]);
	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE640 / 5);
	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);
	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);
	backward_4o(16 * 5, &x[k16], 16 * 5, &X[i16], r2i[j16], r4i[j16]);
}

#define WGSIZE2560	(2560 / 8)

#define DECLARE_VAR_2560() \
	__local uint64_2 X[1280]; \
	\
	DECLARE_VAR_REG5(); \
	const sz_t lid4 = local_id, id4 = 4 * WGSIZE2560 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \
	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, /*k1 = 5 * t1 + id_mod5,*/ i1 = 5 * (t1 % (1280 / 5)) + id_mod5; \
	const sz_t j4 = id_5 / 4, t4 = 4 * (id_5 & ~(4 - 1)) + (id_5 % 4), /*k4 = 5 * t4 + id_mod5,*/ i4 = 5 * (t4 % (1280 / 5)) + id_mod5; \
	const sz_t j16 = id_5 / 16, t16 = 4 * (id_5 & ~(16 - 1)) + (id_5 % 16), /*k16 = 5 * t16 + id_mod5,*/ i16 = 5 * (t16 % (1280 / 5)) + id_mod5; \
	const sz_t j64 = id_5 / 64, t64 = 4 * (id_5 & ~(64 - 1)) + (id_5 % 64), k64 = 5 * t64 + id_mod5, i64 = local_id;	// 5 * (t64 % (1280 / 5)) + id_mod5;

#if MAX_WG_SZ >= WGSIZE2560
#define ATTR_2560() \
	__attribute__((reqd_work_group_size(WGSIZE2560, 1, 1)))
#else
#define ATTR_2560()
#endif

__kernel
ATTR_2560()
void forward_mul2560(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_2560();

	forward_4i(64 * 5, &X[i64], 64 * 5, &x[k64], r2[j64], r4[j64]);
	forward_4(16 * 5, &X[i16], r2[j16], r4[j16]);
	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE2560 / 5);
}

__kernel
ATTR_2560()
void sqr2560(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_2560();

	forward_4i(64 * 5, &X[i64], 64 * 5, &x[k64], r2[j64], r4[j64]);
	forward_4(16 * 5, &X[i16], r2[j16], r4[j16]);
	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE2560 / 5);
	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);
	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);
	backward_4(16 * 5, &X[i16], r2i[j16], r4i[j16]);
	backward_4o(64 * 5, &x[k64], 64 * 5, &X[i64], r2i[j64], r4i[j64]);
}

__kernel
ATTR_2560()
void mul2560(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_2560();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(64 * 5, &X[i64], 64 * 5, &x[k64], r2[j64], r4[j64]);
	forward_4(16 * 5, &X[i16], r2[j16], r4[j16]);
	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);
	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);
	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE2560 / 5);
	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);
	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);
	backward_4(16 * 5, &X[i16], r2i[j16], r4i[j16]);
	backward_4o(64 * 5, &x[k64], 64 * 5, &X[i64], r2i[j64], r4i[j64]);
}

#endif

// --- carry ---

#if defined(CWM_WG_SZ)

// Unweight, carry, mul by a, weight (pass 1)
__kernel
void carry_weight_mul_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,
	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const uint32 a, const sz_t offset)
{
	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);
	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);
	__global const uint64_4 * restrict const weight4i_n = (__global const uint64_4 *)(&weight[N_SZ]);
	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;
	__local uint64 cl[CWM_WG_SZ];

	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ;

	const uint_8_4 wd = width4[gid];

	uint64 c = 0;
	uint64_4 u = mod_mul4(x[gid], weight4i_n[gid]);
	u = adc_mul4(u, a, wd, &c);

	cl[lid] = c;

	barrier(CLK_LOCAL_MEM_FENCE);

	u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);
	x[gid] = mod_mul4(u, weight4[gid]);

	if (lid == CWM_WG_SZ - 1)
	{
		carry[(gid != N_SZ / 4 - 1) ? gid / CWM_WG_SZ + 1 : 0] = c;
	}
}

// Unweight, carry, mul by a, weight (pass 2)
__kernel
void carry_weight_mul_p2(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,
	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const sz_t offset)
{
	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);
	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);
	__global const uint64_4 * restrict const weight4i = (__global const uint64_4 *)(&weight[2 * N_SZ]);
	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;

	const sz_t gid = (sz_t)get_global_id(0), id = CWM_WG_SZ * gid;

	const uint_8_4 wd = width4[id];

	uint64_4 u = mod_mul4(x[id], weight4i[id]);
	u = adc4(u, wd, carry[gid]);
	x[id] = mod_mul4(u, weight4[id]);
}

#endif
#if defined(CWM_WG_SZ2)

#define N_SZ_8	(N_SZ / 8)

// Inverse radix-2, unweight, carry, mul by a, weight, radix-2 (pass 1)
__kernel
void carry_weight_mul2_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,
	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const uint32 a, const sz_t offset)
{
	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);
	__global uint64_2 * restrict const carry2 = (__global uint64_2 *)carry;
	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);
	__global const uint64_4 * restrict const weight4i_n = (__global const uint64_4 *)(&weight[N_SZ]);
	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;
	__local uint64_2 cl[CWM_WG_SZ2];

	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ2;

	const uint_8_4 wd0 = width4[gid + 0 * N_SZ_8], wd1 = width4[gid + 1 * N_SZ_8];

	uint64 c1_0 = 0, c1_1 = 0;
	const uint64_4 x0 = x[gid + 0 * N_SZ_8], x1 = x[gid + 1 * N_SZ_8];
	uint64_4 u0 = mod_add4(x0, x1), u1 = mod_sub4(x0, x1);
	u0 = mod_mul4(u0, weight4i_n[gid + 0 * N_SZ_8]);
	u1 = mod_mul4(u1, weight4i_n[gid + 1 * N_SZ_8]);
	u0 = adc_mul4(u0, a, wd0, &c1_0);
	u1 = adc_mul4(u1, a, wd1, &c1_1);

	cl[lid] = (uint64_2)(c1_0, c1_1);

	barrier(CLK_LOCAL_MEM_FENCE);

	const uint64_2 c2 = (lid == 0) ? (uint64_2)(0, 0) : cl[lid - 1];
	u0 = adc4(u0, wd0, c2.s0);
	u1 = adc4(u1, wd1, c2.s1);
	u0 = mod_mul4(u0, weight4[gid + 0 * N_SZ_8]);
	u1 = mod_mul4(u1, weight4[gid + 1 * N_SZ_8]);
	x[gid + 0 * N_SZ_8] = mod_add4(u0, u1);
	x[gid + 1 * N_SZ_8] = mod_sub4(u0, u1);

	if (lid == CWM_WG_SZ2 - 1)
	{
		const uint64_2 c1 = (gid != N_SZ_8 - 1) ? (uint64_2)(c1_0, c1_1) : (uint64_2)(c1_1, c1_0);
		carry2[(gid != N_SZ_8 - 1) ? gid / CWM_WG_SZ2 + 1 : 0] = c1;
	}
}

// Inverse radix-2, unweight, carry, mul by a, weight, radix-2 (pass 2)
__kernel
void carry_weight_mul2_p2(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,
	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const sz_t offset)
{
	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);
	__global const uint64_2 * restrict const carry2 = (__global const uint64_2 *)carry;
	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);
	__global const uint64_4 * restrict const weight4i = (__global const uint64_4 *)(&weight[2 * N_SZ]);
	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;

	const sz_t gid = (sz_t)get_global_id(0), id = CWM_WG_SZ2 * gid;

	const uint_8_4 wd0 = width4[id + 0 * N_SZ_8], wd1 = width4[id + 1 * N_SZ_8];

	const uint64_4 x0 = x[id + 0 * N_SZ_8], x1 = x[id + 1 * N_SZ_8];
	uint64_4 u0 = mod_half4(mod_add4(x0, x1)), u1 = mod_half4(mod_sub4(x0, x1));
	u0 = mod_mul4(u0, weight4i[id + 0 * N_SZ_8]);
	u1 = mod_mul4(u1, weight4i[id + 1 * N_SZ_8]);
	const uint64_2 c = carry2[gid];
	u0 = adc4(u0, wd0, c.s0);
	u1 = adc4(u1, wd1, c.s1);
	u0 = mod_mul4(u0, weight4[id + 0 * N_SZ_8]);
	u1 = mod_mul4(u1, weight4[id + 1 * N_SZ_8]);
	x[id + 0 * N_SZ_8] = mod_add4(u0, u1);
	x[id + 1 * N_SZ_8] = mod_sub4(u0, u1);
}

#endif

// --- misc ---

__kernel
void copy(__global uint64 * restrict const reg, const sz_t offset_y, const sz_t offset_x)
{
	const sz_t gid = (sz_t)get_global_id(0);
	reg[offset_y + gid] = reg[offset_x + gid];
}

__kernel
void subtract(__global uint64 * restrict const reg, __global const uint64 * restrict const weight,
	__global const uint_8 * restrict const width, const sz_t offset, const uint32 a)
{
	__global uint64 * restrict const x = &reg[offset];
	__global const uint64 * restrict const weighti = &weight[2 * N_SZ];

	uint32 c = a;
	while (c != 0)
	{
		// Unweight, sub with carry, weight
		for (size_t k = 0; k < N_SZ; ++k)
		{
			x[k] = mod_mul(sbc(mod_mul(x[k], weighti[k]), width[k], &c), weight[k]);
			if (c == 0) return;
		}
	}
}

#define N_SZ_2	(N_SZ / 2)

__kernel
void subtract2(__global uint64 * restrict const reg, __global const uint64 * restrict const weight,
	__global const uint_8 * restrict const width, const sz_t offset, const uint32 a)
{
	__global uint64 * restrict const x = &reg[offset];
	__global const uint64 * restrict const weighti = &weight[2 * N_SZ];

	uint32 c = a;
	while (c != 0)
	{
		// Inverse radix-2, unweight, sub with carry, weight, radix-2
		for (size_t k = 0; k < N_SZ_2; ++k)
		{
			const uint64 u0 = x[k + 0 * N_SZ_2], u1 = x[k + 1 * N_SZ_2];
			const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));
			const uint64 v0n = mod_mul(sbc(mod_mul(v0, weighti[k + 0 * N_SZ_2]), width[k + 0 * N_SZ_2], &c), weight[k + 0 * N_SZ_2]);
			x[k + 0 * N_SZ_2] = mod_add(v0n, v1); x[k + 1 * N_SZ_2] = mod_sub(v0n, v1);
			if (c == 0) return;
		}
		for (size_t k = 0; k < N_SZ_2; ++k)
		{
			const uint64 u0 = x[k + 0 * N_SZ_2], u1 = x[k + 1 * N_SZ_2];
			const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));
			const uint64 v1n = mod_mul(sbc(mod_mul(v1, weighti[k + 1 * N_SZ_2]), width[k + 1 * N_SZ_2], &c), weight[k + 1 * N_SZ_2]);
			x[k + 0 * N_SZ_2] = mod_add(v0, v1n); x[k + 1 * N_SZ_2] = mod_sub(v0, v1n);
			if (c == 0) return;
		}
	}
}
