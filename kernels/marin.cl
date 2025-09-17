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
#define LN_SZ_S5	14
#define INV_N_2		18446181119461294081ul
#define W_F1		4611686017353646079ul
#define W_F2		5818851782451133869ul
#define W_F3		10808002860802937880ul
#define W_F4		1418753320236437486ul
#define W_F5		7970496220330062908ul
#define BLK16		16u
#define BLK32		8u
#define BLK64		4u
#define BLK128		2u
#define BLK256		1u
#define BLK512		1u
#define CHUNK16		16u
#define CHUNK20		16u
#define CHUNK64		8u
#define CHUNK80		8u
#define CHUNK256	4u
#define CHUNK320	2u
#define CWM_WG_SZ	256u
#define MAX_WG_SZ	256u
#endif

#define sz_t		uint
#define uint_8		uchar
#ifndef uint32
#define uint32		uint
#endif
#ifndef int32
#define int32		int
#endif
#ifndef uint64
#define uint64		ulong
#endif
#define uint_8_4	uchar4
#define uint64_2	ulong2
#define uint64_4	ulong4

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
	asm volatile ("addc.u32 %0, 0xffffffff, 0;" : "=r" (nc));					// If no carry then nc = MOD_MP64 else nc = 0
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

INLINE uint64_2 mod_add2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_add(lhs.s0, rhs.s0), mod_add(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_sub2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_sub(lhs.s0, rhs.s0), mod_sub(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_mul2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_mul(lhs.s0, rhs.s0), mod_mul(lhs.s1, rhs.s1)); }
INLINE uint64_2 mod_sqr2(const uint64_2 lhs) { return (uint64_2)(mod_sqr(lhs.s0), mod_sqr(lhs.s1)); }
INLINE uint64_2 mod_muli2(const uint64_2 lhs) { return (uint64_2)(mod_muli(lhs.s0), mod_muli(lhs.s1)); }

INLINE uint64_4 mod_mul4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_mul2(lhs.s01, rhs.s01), mod_mul2(lhs.s23, rhs.s23)); }

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

// Radix-4
INLINE void fwd4(uint64_2 * const x, const uint64 r1, const uint64_2 r23)
{
	const uint64_2 u0 = x[0], u2 = mod_mul2(x[2], r1), u1 = mod_mul2(x[1], r23.s0), u3 = mod_mul2(x[3], r23.s1);
	const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3));
	x[0] = mod_add2(v0, v1); x[1] = mod_sub2(v0, v1); x[2] = mod_add2(v2, v3); x[3] = mod_sub2(v2, v3);
}

// Inverse radix-4
INLINE void bck4(uint64_2 * const x, const uint64 r1i, const uint64_2 r23i)
{
	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3];
	const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2));
	x[0] = mod_add2(v0, v2); x[2] = mod_mul2(mod_sub2(v0, v2), r1i); x[1] = mod_mul2(mod_add2(v1, v3), r23i.s0); x[3] = mod_mul2(mod_sub2(v1, v3), r23i.s1);
}

// Radix-4, first stage
INLINE void fwd4_0(uint64_2 * const x)
{
	const uint64_2 u0 = x[0], u2 = x[2], u1 = x[1], u3 = x[3];
	const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3));
	x[0] = mod_add2(v0, v1); x[1] = mod_sub2(v0, v1); x[2] = mod_add2(v2, v3); x[3] = mod_sub2(v2, v3);
}

// Inverse radix-4, first stage
INLINE void bck4_0(uint64_2 * const x)
{
	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3];
	const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2));
	x[0] = mod_add2(v0, v2); x[2] = mod_sub2(v0, v2); x[1] = mod_add2(v1, v3); x[3] = mod_sub2(v1, v3);
}

// 2 x radix-2
INLINE void fwd22(uint64_2 * const x, const uint64_2 r)
{
	const uint64_2 u0 = x[0], u1 = mod_mul2(x[1], r);
	x[0] = mod_add2(u0, u1); x[1] = mod_sub2(u0, u1);
}

// 2 x inverse radix-2
INLINE void bck22(uint64_2 * const x, const uint64_2 ri)
{
	const uint64_2 u0 = x[0], u1 = x[1];
	x[0] = mod_add2(u0, u1); x[1] = mod_mul2(mod_sub2(u0, u1), ri);
}

// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.
#define butterfly5(a0, a1, a2, a3, a4) \
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

// Radix-5, first stage
INLINE void fwd5_0(uint64_2 * const x)
{
	uint64_2 a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], a4 = x[4];
	butterfly5(a0, a1, a2, a3, a4);
	x[0] = a0; x[1] = a1; x[2] = a2; x[3] = a3; x[4] = a4;
}
	
// Inverse radix-5, first stage
INLINE void bck5_0(uint64_2 * const x)
{
	uint64_2 a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], a4 = x[4];
	butterfly5(a0, a1, a2, a3, a4);
	x[0] = a0; x[4] = a1; x[3] = a2; x[2] = a3; x[1] = a4;
}

// 2 x Radix-2, sqr, inverse radix-2
INLINE void sqr22(uint64_2 * const x, const uint64 r)
{
	const uint64_2 sx0 = mod_sqr2(x[0]), sx1 = mod_sqr2(x[1]);
	x[0].s1 = mod_mul(x[0].s1, mod_add(x[0].s0, x[0].s0)); x[0].s0 = mod_add(sx0.s0, mod_mul(sx0.s1, r));
	x[1].s1 = mod_mul(x[1].s1, mod_add(x[1].s0, x[1].s0)); x[1].s0 = mod_sub(sx1.s0, mod_mul(sx1.s1, r));
}

// 2 x Radix-2, mul, inverse radix-2
INLINE void mul22(uint64_2 * const x, const uint64_2 * const y, const uint64 r)
{
	const uint64_2 xy0 = mod_mul2(x[0], y[0]), xy1 = mod_mul2(x[1], y[1]);
	x[0].s1 = mod_add(mod_mul(x[0].s0, y[0].s1), mod_mul(x[0].s1, y[0].s0)); x[0].s0 = mod_add(xy0.s0, mod_mul(xy0.s1, r));
	x[1].s1 = mod_add(mod_mul(x[1].s0, y[1].s1), mod_mul(x[1].s1, y[1].s0)); x[1].s0 = mod_sub(xy1.s0, mod_mul(xy1.s1, r));
}

INLINE void sqr_4x1(uint64_2 * const xl, const uint64 r, const uint64 ri)
{
	fwd22(xl, r);
	sqr22(xl, r);
	bck22(xl, ri);
}

INLINE void mul_4x1(uint64_2 * const xl, const uint64_2 * const yl, const uint64 r, const uint64 ri)
{
	fwd22(xl, r);
	mul22(xl, yl, r);
	bck22(xl, ri);
}

INLINE void sqr_4(uint64_2 * const xl, const uint64_2 r, const uint64_2 ri)
{
	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);
	sqr22(&xl[0], r.s0); sqr22(&xl[2], r.s1);
	bck22(&xl[0], ri.s0); bck22(&xl[2], ri.s1);
}

INLINE void mul_4(uint64_2 * const xl, const uint64_2 * const yl, const uint64_2 r, const uint64_2 ri)
{
	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);
	mul22(&xl[0], &yl[0], r.s0); mul22(&xl[2], &yl[2], r.s1);
	bck22(&xl[0], ri.s0); bck22(&xl[2], ri.s1);
}

INLINE void sqr_8(uint64_2 * const xl, const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)
{
	fwd4(xl, r1, r23);
	sqr22(&xl[0], r23.s0);
	sqr22(&xl[2], mod_muli(r23.s0));
	bck4(xl, r1i, r23i);
}

INLINE void mul_8(uint64_2 * const xl, const uint64_2 * const yl, const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)
{
	fwd4(xl, r1, r23);
	mul22(&xl[0], &yl[0], r23.s0);
	mul22(&xl[2], &yl[2], mod_muli(r23.s0));
	bck4(xl, r1i, r23i);
}

INLINE void loadg2(const sz_t n, uint64_2 * const xl, __global const uint64_2 * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }
INLINE void loadl2(const sz_t n, uint64_2 * const xl, __local const uint64_2 * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }
INLINE void storeg2(const sz_t n, __global uint64_2 * restrict const x, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }
INLINE void storel2(const sz_t n, __local uint64_2 * restrict const X, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }

// --- transform - global mem ---

// Radix-4
/*__kernel
void forward4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t s, const uint32 lm)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);

	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);
	fwd4(xl, r2[sj], r4[sj]);
	storeg2(4, &x[k], m, xl);
}

// Inverse radix-4
__kernel
void backward4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t s, const uint32 lm)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2i = &root[N_SZ / 2];
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]);

	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);
	bck4(xl, r2i[sj], r4i[sj]);
	storeg2(4, &x[k], m, xl);
}*/

#if (N_SZ % 5 != 0) && (N_SZ <= 32)

// Radix-4, first stage
__kernel
void forward4_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);

	const sz_t id = (sz_t)get_global_id(0), k = id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], N_SZ / 8);
	fwd4_0(xl);
	storeg2(4, &x[k], N_SZ / 8, xl);
}

// Inverse radix-4, first stage
__kernel
void backward4_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);

	const sz_t id = (sz_t)get_global_id(0), k = id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], N_SZ / 8);
	bck4_0(xl);
	storeg2(4, &x[k], N_SZ / 8, xl);
}

#endif
#if (N_SZ == 40)

// Radix-5, first stage
__kernel
void forward5_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);

	const sz_t id = (sz_t)get_global_id(0), k = id;

	if (k < N_SZ / 10)
	{
		uint64_2 xl[5]; loadg2(5, xl, &x[k], N_SZ / 10);
		fwd5_0(xl);
		storeg2(5, &x[k], N_SZ / 10, xl);
	}
}

// Inverse radix-5, first stage
__kernel
void backward5_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);

	const sz_t id = (sz_t)get_global_id(0), k = id;

	if (k < N_SZ / 10)
	{
		uint64_2 xl[5]; loadg2(5, xl, &x[k], N_SZ / 10);
		bck5_0(xl);
		storeg2(5, &x[k], N_SZ / 10, xl);
	}
}

#endif
#if (N_SZ == 4)

// Radix-4
__kernel
void forward_mul4x1(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;

	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);
	fwd22(xl, r2[N_SZ / 4 + j]);
	storeg2(2, &x[k], 1, xl);
}

// Radix-4, square, inverse radix-4
__kernel
void sqr4x1(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64 * restrict const r2i = &root[N_SZ / 2];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;

	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);
	sqr_4x1(xl, r2[N_SZ / 4 + j], r2i[N_SZ / 4 + j]);
	storeg2(2, &x[k], 1, xl);
}

// Radix-4, mul, inverse radix-4
__kernel
void mul4x1(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);
	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64 * restrict const r2i = &root[N_SZ / 2];

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;

	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);
	uint64_2 yl[2]; loadg2(2, yl, &y[k], 1);
	mul_4x1(xl, yl, r2[N_SZ / 4 + j], r2i[N_SZ / 4 + j]);
	storeg2(2, &x[k], 1, xl);
}

#endif
#if (N_SZ >= 16) && (N_SZ <= 80)

// 2 x Radix-4
__kernel
void forward_mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64_2 * restrict const r2 = (__global const uint64_2 *)(&root[0]);

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	const uint64_2 r = r2[N_SZ / 8 + j];
	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);
	storeg2(4, &x[k], 1, xl);
}

// 2 x Radix-4, square, inverse radix-4
__kernel
void sqr4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64_2 * restrict const r2 = (__global const uint64_2 *)(&root[0]);
	__global const uint64_2 * restrict const r2i = (__global const uint64_2 *)(&root[N_SZ / 2]);

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	sqr_4(xl, r2[N_SZ / 8 + j], r2i[N_SZ / 8 + j]);
	storeg2(4, &x[k], 1, xl);
}

// 2 x Radix-4, mul, inverse radix-4
__kernel
void mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);
	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);
	__global const uint64_2 * restrict const r2 = (__global const uint64_2 *)(&root[0]);
	__global const uint64_2 * restrict const r2i = (__global const uint64_2 *)(&root[N_SZ / 2]);

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	uint64_2 yl[4]; loadg2(4, yl, &y[k], 1);
	mul_4(xl, yl, r2[N_SZ / 8 + j], r2i[N_SZ / 8 + j]);
	storeg2(4, &x[k], 1, xl);
}

#endif
#if (N_SZ >= 8) && (N_SZ <= 160)

// Radix-8
__kernel
void forward_mul8(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	fwd4(xl, r2[N_SZ / 8 + j], r4[N_SZ / 8 + j]);
	storeg2(4, &x[k], 1, xl);
}

// Radix-8, square, inverse radix-8
__kernel
void sqr8(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64 * restrict const r2i = &root[N_SZ / 2];
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]);

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	const uint64 r1 = r2[N_SZ / 8 + j]; const uint64_2 r23 = r4[N_SZ / 8 + j];
	const uint64 r1i = r2i[N_SZ / 8 + j]; const uint64_2 r23i = r4i[N_SZ / 8 + j];
	sqr_8(xl, r1, r23, r1i, r23i);
	storeg2(4, &x[k], 1, xl);
}

// Radix-8, mul, inverse radix-8
__kernel
void mul8(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)
{
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);
	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);
	__global const uint64 * restrict const r2 = &root[0];
	__global const uint64 * restrict const r2i = &root[N_SZ / 2];
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]);

	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;

	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);
	uint64_2 yl[4]; loadg2(4, yl, &y[k], 1);
	const uint64 r1 = r2[N_SZ / 8 + j]; const uint64_2 r23 = r4[N_SZ / 8 + j];
	const uint64 r1i = r2i[N_SZ / 8 + j]; const uint64_2 r23i = r4i[N_SZ / 8 + j];
	mul_8(xl, yl, r1, r23, r1i, r23i);
	storeg2(4, &x[k], 1, xl);
}

#endif

// --- transform - local mem ---

INLINE void forward_4(const sz_t m, __local uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, m);
	fwd4(xl, r1, r23);
	storel2(4, X, m, xl);
}

INLINE void forward_4i(const sz_t ml, __local uint64_2 * restrict const X,
	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1, const uint64_2 r23)
{
	uint64_2 xl[4]; loadg2(4, xl, x, mg);
	fwd4(xl, r1, r23);
	storel2(4, X, ml, xl);
}

INLINE void forward_4o(const sz_t mg, __global uint64_2 * restrict const x,
	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	fwd4(xl, r1, r23);
	storeg2(4, x, mg, xl);
}

INLINE void backward_4(const sz_t m, __local uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, m);
	bck4(xl, r1i, r23i);
	storel2(4, X, m, xl);
}

INLINE void backward_4i(const sz_t ml, __local uint64_2 * restrict const X,
	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1i, const uint64_2 r23i)
{
	uint64_2 xl[4]; loadg2(4, xl, x, mg);
	bck4(xl, r1i, r23i);
	storel2(4, X, ml, xl);
}

INLINE void backward_4o(const sz_t mg, __global uint64_2 * restrict const x,
	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	bck4(xl, r1i, r23i);
	storeg2(4, x, mg, xl);
}

INLINE void forward_4i_0(const sz_t ml, __local uint64_2 * restrict const X,
	const sz_t mg, __global const uint64_2 * restrict const x)
{
	uint64_2 xl[4]; loadg2(4, xl, x, mg);
	fwd4_0(xl);
	storel2(4, X, ml, xl);
}

INLINE void backward_4o_0(const sz_t mg, __global uint64_2 * restrict const x,
	const sz_t ml, __local const uint64_2 * restrict const X)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, ml);
	bck4_0(xl);
	storeg2(4, x, mg, xl);
}

INLINE void forward_5i_0(const sz_t ml, __local uint64_2 * restrict const X,
	const sz_t mg, __global const uint64_2 * restrict const x)
{
	uint64_2 xl[5]; loadg2(5, xl, x, mg);
	fwd5_0(xl);
	storel2(5, X, ml, xl);
}

INLINE void backward_5o_0(const sz_t mg, __global uint64_2 * restrict const x,
	const sz_t ml, __local const uint64_2 * restrict const X)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[5]; loadl2(5, xl, X, ml);
	bck5_0(xl);
	storeg2(5, x, mg, xl);
}

INLINE void forward_mul_4o(__global uint64_2 * restrict const x, __local const uint64_2 * restrict const X, const uint64_2 r)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);
	storeg2(4, x, 1, xl);
}

INLINE void forward_mul_8o(__global uint64_2 * restrict const x, __local const uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	fwd4(xl, r1, r23);
	storeg2(4, x, 1, xl);
}

INLINE void square_4(__local uint64_2 * restrict const X, const uint64_2 r, const uint64_2 ri)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	sqr_4(xl, r, ri);
	storel2(4, X, 1, xl);
}

INLINE void square_8(__local uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	sqr_8(xl, r1, r23, r1i, r23i);
	storel2(4, X, 1, xl);
}

INLINE void mult_4(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y, const uint64_2 r, const uint64_2 ri)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	uint64_2 yl[4]; loadg2(4, yl, y, 1);
	mul_4(xl, yl, r, ri);
	storel2(4, X, 1, xl);
}

INLINE void mult_8(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y,
	const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint64_2 xl[4]; loadl2(4, xl, X, 1);
	uint64_2 yl[4]; loadg2(4, yl, y, 1);
	mul_8(xl, yl, r1, r23, r1i, r23i);
	storel2(4, X, 1, xl);
}

#define DECLARE_VAR_REG() \
	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]); \
	__global const uint64 * restrict const r2 = &root[0]; \
	__global const uint64 * restrict const r2i = &root[N_SZ / 2]; \
	__global const uint64_2 * restrict const r2_2 = (__global const uint64_2 *)(&root[0]); \
	__global const uint64_2 * restrict const r2i_2 = (__global const uint64_2 *)(&root[N_SZ / 2]); \
	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]); \
	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]); \
	const sz_t id = (sz_t)get_global_id(0);

/////////////////////////////////////

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
	const sz_t sj = s + idx_m;

#if (MAX_WG_SZ >= 16 / 4 * CHUNK16)

#define ATTR_FB_16()	__attribute__((reqd_work_group_size(16 / 4 * CHUNK16, 1, 1)))

/*__kernel
ATTR_FB_16()
void forward16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(16 / 4, CHUNK16);

	forward_4i(4 * CHUNK16, &X[i], 4u << lm, &x[ki], r2[sj / 4], r4[sj / 4]);
	forward_4o(1u << lm, &x[ko], 1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], r2[sj], r4[sj]);
}

__kernel
ATTR_FB_16()
void backward16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(16 / 4, CHUNK16);

	backward_4i(1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]);
	backward_4o(4u << lm, &x[ki], 4 * CHUNK16, &X[i], r2i[sj / 4], r4i[sj / 4]);
}*/

#if (N_SZ % 5 != 0) && (N_SZ >= 64) && (N_SZ <= 2048)

__kernel
ATTR_FB_16()
void forward16_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 16 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;
	DECLARE_VAR(16 / 4, CHUNK16);

	forward_4i_0(4 * CHUNK16, &X[i], 4u << lm, &x[ki]);
	forward_4o(1u << lm, &x[ko], 1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], r2[sj], r4[sj]);
}

__kernel
ATTR_FB_16()
void backward16_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 16 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;
	DECLARE_VAR(16 / 4, CHUNK16);

	backward_4i(1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]);
	backward_4o_0(4u << lm, &x[ki], 4 * CHUNK16, &X[i]);
}

#endif
#endif

#if (MAX_WG_SZ >= 20 / 4 * CHUNK20)

#if (N_SZ % 5 == 0) && (N_SZ >= 80) && (N_SZ <= 2560)

#define ATTR_FB_20()	__attribute__((reqd_work_group_size(20 / 4 * CHUNK20, 1, 1)))

__kernel
ATTR_FB_20()
void forward20_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 20 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;
	DECLARE_VAR(20 / 4, CHUNK20);

	if (i < 4 * (20 / 4 * CHUNK20) / 5) forward_5i_0(4 * CHUNK20, &X[i], 4u << lm, &x[ki]);
	forward_4o(1u << lm, &x[ko], 1 * CHUNK20, &Xi[CHUNK20 * 4 * thread_idx], r2[sj], r4[sj]);
}

__kernel
ATTR_FB_20()
void backward20_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 20 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;
	DECLARE_VAR(20 / 4, CHUNK20);

	backward_4i(1 * CHUNK20, &Xi[CHUNK20 * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]);
	if (i < 4 * (20 / 4 * CHUNK20) / 5) backward_5o_0(4u << lm, &x[ki], 4 * CHUNK20, &X[i]);
}

#endif
#endif

#define FORWARD_64_80(CHUNK_N) \
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	forward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2[sj / 4], r4[sj / 4]); \
	forward_4o(1u << lm, &x[ko], 1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], r2[sj], r4[sj]);

#define BACKWARD_64_80(CHUNK_N) \
	backward_4i(1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	backward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2i[sj / 4], r4i[sj / 4]);

#if (MAX_WG_SZ >= 64 / 4 * CHUNK64)

#define ATTR_FB_64()	__attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))

#if (N_SZ >= 655360)

__kernel
ATTR_FB_64()
void forward64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(64 / 4, CHUNK64);

	forward_4i(16 * CHUNK64, &X[i], 16u << lm, &x[ki], r2[sj / 16], r4[sj / 16]);
	FORWARD_64_80(CHUNK64);
}

__kernel
ATTR_FB_64()
void backward64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(64 / 4, CHUNK64);

	BACKWARD_64_80(CHUNK64);
	backward_4o(16u << lm, &x[ki], 16 * CHUNK64, &X[i], r2i[sj / 16], r4i[sj / 16]);
}

#endif
#if (N_SZ % 5 != 0) && (N_SZ >= 4096)

__kernel
ATTR_FB_64()
void forward64_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 64 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;
	DECLARE_VAR(64 / 4, CHUNK64);

	forward_4i_0(16 * CHUNK64, &X[i], 16u << lm, &x[ki]);
	FORWARD_64_80(CHUNK64);
}

__kernel
ATTR_FB_64()
void backward64_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 64 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;
	DECLARE_VAR(64 / 4, CHUNK64);

	BACKWARD_64_80(CHUNK64);
	backward_4o_0(16u << lm, &x[ki], 16 * CHUNK64, &X[i]);
}

#endif
#endif
#if (MAX_WG_SZ >= 80 / 4 * CHUNK80)

#if (N_SZ % 5 == 0) && (N_SZ >= 5120)

#define ATTR_FB_80()	__attribute__((reqd_work_group_size(80 / 4 * CHUNK80, 1, 1)))

__kernel
ATTR_FB_80()
void forward80_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 80 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;
	DECLARE_VAR(80 / 4, CHUNK80);

	if (i < 4 * (80 / 4 * CHUNK80) / 5) forward_5i_0(16 * CHUNK80, &X[i], 16u << lm, &x[ki]);
	FORWARD_64_80(CHUNK80);
}

__kernel
ATTR_FB_80()
void backward80_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 80 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;
	DECLARE_VAR(80 / 4, CHUNK80);

	BACKWARD_64_80(CHUNK80);
	if (i < 4 * (80 / 4 * CHUNK80) / 5) backward_5o_0(16u << lm, &x[ki], 16 * CHUNK80, &X[i]);
}

#endif
#endif

#define FORWARD_256_320(CHUNK_N) \
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \
	forward_4(16 * CHUNK_N, &Xi[CHUNK_N * i16], r2[sj / 16], r4[sj / 16]); \
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	forward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2[sj / 4], r4[sj / 4]); \
	forward_4o(1u << lm, &x[ko], 1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], r2[sj], r4[sj]);

#define BACKWARD_256_320(CHUNK_N) \
	backward_4i(1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	backward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2i[sj / 4], r4i[sj / 4]); \
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \
	backward_4(16 * CHUNK_N, &Xi[CHUNK_N * i16], r2i[sj / 16], r4i[sj / 16]);

#if (MAX_WG_SZ >= 256 / 4 * CHUNK256)

#define ATTR_FB_256()	__attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))

#if (N_SZ >= 2621440)

__kernel
ATTR_FB_256()
void forward256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(256 / 4, CHUNK256);

	forward_4i(64 * CHUNK256, &X[i], 64u << lm, &x[ki], r2[sj / 64], r4[sj / 64]);
	FORWARD_256_320(CHUNK256);
}

__kernel
ATTR_FB_256()
void backward256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(256 / 4, CHUNK256);

	BACKWARD_256_320(CHUNK256);
	backward_4o(64u << lm, &x[ki], 64 * CHUNK256, &X[i], r2i[sj / 64], r4i[sj / 64]);
}

#endif
#if (N_SZ % 5 != 0) && (N_SZ >= 131072)

__kernel
ATTR_FB_256()
void forward256_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 256 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;
	DECLARE_VAR(256 / 4, CHUNK256);

	forward_4i_0(64 * CHUNK256, &X[i], 64u << lm, &x[ki]);
	FORWARD_256_320(CHUNK256);
}

__kernel
ATTR_FB_256()
void backward256_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 256 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;
	DECLARE_VAR(256 / 4, CHUNK256);

	BACKWARD_256_320(CHUNK256);
	backward_4o_0(64u << lm, &x[ki], 64 * CHUNK256, &X[i]);
}

#endif
#endif
#if (MAX_WG_SZ >= 320 / 4 * CHUNK320)

#if (N_SZ % 5 == 0) && (N_SZ >= 81920)

#define ATTR_FB_320()	__attribute__((reqd_work_group_size(320 / 4 * CHUNK320, 1, 1)))

__kernel
ATTR_FB_320()
void forward320_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 320 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;
	DECLARE_VAR(320 / 4, CHUNK320);

	if (i < 4 * (320 / 4 * CHUNK320) / 5) forward_5i_0(64 * CHUNK320, &X[i], 64u << lm, &x[ki]);
	FORWARD_256_320(CHUNK320);
}

__kernel
ATTR_FB_320()
void backward320_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 320 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;
	DECLARE_VAR(320 / 4, CHUNK320);

	BACKWARD_256_320(CHUNK320);
	if (i < 4 * (320 / 4 * CHUNK320) / 5) backward_5o_0(64u << lm, &x[ki], 64 * CHUNK320, &X[i]);
}

#endif
#endif

#define FORWARD_1024_1280() \
	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64); \
	forward_4(64, &Xi[i64], r2[sj / 64], r4[sj / 64]); \
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \
	forward_4(16, &Xi[i16], r2[sj / 16], r4[sj / 16]); \
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	forward_4(4, &Xi[i4], r2[sj / 4], r4[sj / 4]); \
	forward_4o(1u << lm, &x[ko], 1, &Xi[4 * thread_idx], r2[sj], r4[sj]);

#define BACKWARD_1024_1280() \
	backward_4i(1, &Xi[4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \
	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	backward_4(4, &Xi[i4], r2i[sj / 4], r4i[sj / 4]); \
	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \
	backward_4(16, &Xi[i16], r2i[sj / 16], r4i[sj / 16]); \
	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64); \
	backward_4(64, &Xi[i64], r2i[sj / 64], r4i[sj / 64]);

#if (MAX_WG_SZ >= 1024 / 4)

#define ATTR_FB_1024()	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))

/*__kernel
ATTR_FB_1024()
void forward1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(1024 / 4, 1);

	forward_4i(256, &X[i], 256u << lm, &x[ki], r2[sj / 256], r4[sj / 256]);
	FORWARD_1024_1280();
}

__kernel
ATTR_FB_1024()
void backward1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,
	const sz_t s, const uint32 lm)
{
	DECLARE_VAR(1024 / 4, 1);

	BACKWARD_1024_1280();
	backward_4o(256u << lm, &x[ki], 256, &X[i], r2i[sj / 256], r4i[sj / 256]);
}*/

#if (N_SZ % 5 != 0) && (N_SZ >= 524288) && (N_SZ <= 1048576)

__kernel
ATTR_FB_1024()
void forward1024_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 1024 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;
	DECLARE_VAR(1024 / 4, 1);

	forward_4i_0(256, &X[i], 256u << lm, &x[ki]);
	FORWARD_1024_1280();
}

__kernel
ATTR_FB_1024()
void backward1024_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 1024 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;
	DECLARE_VAR(1024 / 4, 1);

	BACKWARD_1024_1280();
	backward_4o_0(256u << lm, &x[ki], 256, &X[i]);
}

#endif
#endif
#if (MAX_WG_SZ >= 1280 / 4)

#if (N_SZ % 5 == 0)

#define ATTR_FB_1280()	__attribute__((reqd_work_group_size(1280 / 4, 1, 1)))

/*__kernel
ATTR_FB_1280()
void forward1280_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 1280 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;
	DECLARE_VAR(1280 / 4, 1);

	if (i < 4 * (1280 / 4) / 5) forward_5i_0(256, &X[i], 256u << lm, &x[ki]);
	FORWARD_1024_1280();
}

__kernel
ATTR_FB_1280()
void backward1280_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	const sz_t s = 1280 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;
	DECLARE_VAR(1280 / 4, 1);

	BACKWARD_1024_1280();
	if (i < 4 * (1280 / 4) / 5) backward_5o_0(256u << lm, &x[ki], 256, &X[i]);
}*/

#endif
#endif

/////////////////////////////////////

#if (MAX_WG_SZ >= 16 / 4 * BLK16) && (N_SZ >= 256) && (N_SZ <= 320)

#define DECLARE_VAR_16() \
	__local uint64_2 X[16 * BLK16]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (16 * BLK16); \
	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % (16 * BLK16);

#define ATTR_16()	__attribute__((reqd_work_group_size(16 / 4 * BLK16, 1, 1)))

__kernel
ATTR_16()
void forward_mul16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_16();

	forward_4i(2, &X[i2], 2, &x[k2], r2[sj2], r4[sj2]);
	forward_mul_4o(&x[k], &X[i], r2_2[sj]);
}

__kernel
ATTR_16()
void sqr16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_16();

	forward_4i(2, &X[i2], 2, &x[k2], r2[sj2], r4[sj2]);
	square_4(&X[i], r2_2[sj], r2i_2[sj]);
	backward_4o(2, &x[k2], 2, &X[i2], r2i[sj2], r4i[sj2]);
}

__kernel
ATTR_16()
void mul16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_16();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(2, &X[i2], 2, &x[k2], r2[sj2], r4[sj2]);
	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);
	backward_4o(2, &x[k2], 2, &X[i2], r2i[sj2], r4i[sj2]);
}

#endif
#if (MAX_WG_SZ >= 32 / 4 * BLK32) && (N_SZ >= 512) && (N_SZ <= 640)

#define DECLARE_VAR_32() \
	__local uint64_2 X[32 * BLK32]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (32 * BLK32); \
	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % (32 * BLK32);

#define ATTR_32()	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))

__kernel
ATTR_32()
void forward_mul32(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_32();

	forward_4i(4, &X[i4], 4, &x[k4], r2[sj4], r4[sj4]);
	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);
}

__kernel
ATTR_32()
void sqr32(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_32();

	forward_4i(4, &X[i4], 4, &x[k4], r2[sj4], r4[sj4]);
	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4o(4, &x[k4], 4, &X[i4], r2i[sj4], r4i[sj4]);
}

__kernel
ATTR_32()
void mul32(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_32();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(4, &X[i4], 4, &x[k4], r2[sj4], r4[sj4]);
	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4o(4, &x[k4], 4, &X[i4], r2i[sj4], r4i[sj4]);
}

#endif
#if (MAX_WG_SZ >= 64 / 4 * BLK64) && (N_SZ >= 1024) && (N_SZ <= 5120)

#define DECLARE_VAR_64() \
	__local uint64_2 X[64 * BLK64]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (64 * BLK64); \
	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % (64 * BLK64); \
	const sz_t sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % (64 * BLK64);

#define ATTR_64()	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))

__kernel
ATTR_64()
void forward_mul64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_64();
	forward_4i(8, &X[i8], 8, &x[k8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	forward_mul_4o(&x[k], &X[i], r2_2[sj]);
}

__kernel
ATTR_64()
void sqr64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_64();

	forward_4i(8, &X[i8], 8, &x[k8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	square_4(&X[i], r2_2[sj], r2i_2[sj]);
	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	backward_4o(8, &x[k8], 8, &X[i8], r2i[sj8], r4i[sj8]);
}

__kernel
ATTR_64()
void mul64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_64();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(8, &X[i8], 8, &x[k8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);
	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	backward_4o(8, &x[k8], 8, &X[i8], r2i[sj8], r4i[sj8]);
}

#endif
#if (MAX_WG_SZ >= 128 / 4 * BLK128) && (N_SZ >= 2048)

#define DECLARE_VAR_128() \
	__local uint64_2 X[128 * BLK128]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (128 * BLK128); \
	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % (128 * BLK128); \
	const sz_t sj16 = sj / 16, k16 = 4 * (id & ~(16 - 1)) + (id % 16), i16 = k16 % (128 * BLK128);

#define ATTR_128()	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))

__kernel
ATTR_128()
void forward_mul128(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_128();
	forward_4i(16, &X[i16], 16, &x[k16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);
}

__kernel
ATTR_128()
void sqr128(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_128();

	forward_4i(16, &X[i16], 16, &x[k16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);
	backward_4o(16, &x[k16], 16, &X[i16], r2i[sj16], r4i[sj16]);
}

__kernel
ATTR_128()
void mul128(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_128();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(16, &X[i16], 16, &x[k16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);
	backward_4o(16, &x[k16], 16, &X[i16], r2i[sj16], r4i[sj16]);
}

#endif
#if (MAX_WG_SZ >= 256 / 4 * BLK256) && (N_SZ >= 16384)

#define DECLARE_VAR_256() \
	__local uint64_2 X[256 * BLK256]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (256 * BLK256); \
	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % (256 * BLK256); \
	const sz_t sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % (256 * BLK256); \
	const sz_t sj32 = sj / 32, k32 = 4 * (id & ~(32 - 1)) + (id % 32), i32 = k32 % (256 * BLK256);

#define ATTR_256()	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))

__kernel
ATTR_256()
void forward_mul256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_256();
	forward_4i(32, &X[i32], 32, &x[k32], r2[sj32], r4[sj32]);
	forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	forward_mul_4o(&x[k], &X[i], r2_2[sj]);
}

__kernel
ATTR_256()
void sqr256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_256();

	forward_4i(32, &X[i32], 32, &x[k32], r2[sj32], r4[sj32]);
	forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	square_4(&X[i], r2_2[sj], r2i_2[sj]);
	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);
	backward_4o(32, &x[k32], 32, &X[i32], r2i[sj32], r4i[sj32]);
}

__kernel
ATTR_256()
void mul256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_256();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(32, &X[i32], 32, &x[k32], r2[sj32], r4[sj32]);
	forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);
	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);
	backward_4o(32, &x[k32], 32, &X[i32], r2i[sj32], r4i[sj32]);
}

#endif
#if (MAX_WG_SZ >= 512 / 4 * BLK512) && (N_SZ >= 32768)

#define DECLARE_VAR_512() \
	__local uint64_2 X[512 * BLK512]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (512 * BLK512); \
	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % (512 * BLK512); \
	const sz_t sj16 = sj / 16, k16 = 4 * (id & ~(16 - 1)) + (id % 16), i16 = k16 % (512 * BLK512); \
	const sz_t sj64 = sj / 64, k64 = 4 * (id & ~(64 - 1)) + (id % 64), i64 = k64 % (512 * BLK512);

#define ATTR_512()	__attribute__((reqd_work_group_size(512 / 4 * BLK512, 1, 1)))

__kernel
ATTR_512()
void forward_mul512(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_512();
	forward_4i(64, &X[i64], 64, &x[k64], r2[sj64], r4[sj64]);
	forward_4(16, &X[i16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);
}

__kernel
ATTR_512()
void sqr512(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_512();

	forward_4i(64, &X[i64], 64, &x[k64], r2[sj64], r4[sj64]);
	forward_4(16, &X[i16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);
	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);
	backward_4o(64, &x[k64], 64, &X[i64], r2i[sj64], r4i[sj64]);
}

__kernel
ATTR_512()
void mul512(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_512();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(64, &X[i64], 64, &x[k64], r2[sj64], r4[sj64]);
	forward_4(16, &X[i16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);
	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);
	backward_4o(64, &x[k64], 64, &X[i64], r2i[sj64], r4i[sj64]);
}

#endif
#if (MAX_WG_SZ >= 1024 / 4) && (N_SZ >= 65536)

#define DECLARE_VAR_1024() \
	__local uint64_2 X[1024]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % 1024; \
	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % 1024; \
	const sz_t sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % 1024; \
	const sz_t sj32 = sj / 32, k32 = 4 * (id & ~(32 - 1)) + (id % 32), i32 = k32 % 1024; \
	const sz_t sj128 = sj / 128, k128 = 4 * (id & ~(128 - 1)) + (id % 128), i128 = k128 % 1024;

#define ATTR_1024()	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))

__kernel
ATTR_1024()
void forward_mul1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_1024();
	forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);
	forward_4(32, &X[i32], r2[sj32], r4[sj32]);
	forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	forward_mul_4o(&x[k], &X[i], r2_2[sj]);
}

__kernel
ATTR_1024()
void sqr1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_1024();

	forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);
	forward_4(32, &X[i32], r2[sj32], r4[sj32]);
	forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	square_4(&X[i], r2_2[sj], r2i_2[sj]);
	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);
	backward_4(32, &X[i32], r2i[sj32], r4i[sj32]);
	backward_4o(128, &x[k128], 128, &X[i128], r2i[sj128], r4i[sj128]);
}

__kernel
ATTR_1024()
void mul1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_1024();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);
	forward_4(32, &X[i32], r2[sj32], r4[sj32]);
	forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);
	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);
	backward_4(32, &X[i32], r2i[sj32], r4i[sj32]);
	backward_4o(128, &x[k128], 128, &X[i128], r2i[sj128], r4i[sj128]);
}

#endif
/* #if (MAX_WG_SZ >= 2048 / 4)

#define DECLARE_VAR_2048() \
	__local uint64_2 X[2048]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % 2048; \
	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % 2048; \
	const sz_t sj16 = sj / 16, k16 = 4 * (id & ~(16 - 1)) + (id % 16), i16 = k16 % 2048; \
	const sz_t sj64 = sj / 64, k64 = 4 * (id & ~(64 - 1)) + (id % 64), i64 = k64 % 2048; \
	const sz_t sj256 = sj / 256, k256 = 4 * (id & ~(256 - 1)) + (id % 256), i256 = k256 % 2048;

#define ATTR_2048()	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))

__kernel
ATTR_2048()
void forward_mul2048(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_2048();
	forward_4i(256, &X[i256], 256, &x[k256], r2[sj256], r4[sj256]);
	forward_4(64, &X[i64], r2[sj64], r4[sj64]);
	forward_4(16, &X[i16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);
}

__kernel
ATTR_2048()
void sqr2048(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)
{
	DECLARE_VAR_2048();

	forward_4i(256, &X[i256], 256, &x[k256], r2[sj256], r4[sj256]);
	forward_4(64, &X[i64], r2[sj64], r4[sj64]);
	forward_4(16, &X[i16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);
	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);
	backward_4(64, &X[i64], r2i[sj64], r4i[sj64]);
	backward_4o(256, &x[k256], 256, &X[i256], r2i[sj256], r4i[sj256]);
}

__kernel
ATTR_2048()
void mul2048(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)
{
	DECLARE_VAR_2048();
	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);

	forward_4i(256, &X[i256], 256, &x[k256], r2[sj256], r4[sj256]);
	forward_4(64, &X[i64], r2[sj64], r4[sj64]);
	forward_4(16, &X[i16], r2[sj16], r4[sj16]);
	forward_4(4, &X[i4], r2[sj4], r4[sj4]);
	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);
	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);
	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);
	backward_4(64, &X[i64], r2i[sj64], r4i[sj64]);
	backward_4o(256, &x[k256], 256, &X[i256], r2i[sj256], r4i[sj256]);
}

#endif */

// --- carry ---

// Unweight, carry, mul by a, weight (pass 1)
__kernel
__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))
void carry_weight_mul_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,
	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const uint32 a, const sz_t offset)
{
	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);
	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);
	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);
	__local uint64 cl[CWM_WG_SZ];

	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ;

	uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);

	const uint64_4 w = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);
	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);

	const uint_8_4 wd = width4[gid];

	uint64 c = 0;
	uint64_4 u = mod_mul4(mod_mul4(x[gid], INV_N_2), wi);
	u = adc_mul4(u, a, wd, &c);

	cl[lid] = c;

	barrier(CLK_LOCAL_MEM_FENCE);

	u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);
	x[gid] = mod_mul4(u, w);

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
	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);
	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);

	const sz_t gid = (sz_t)get_global_id(0), id = CWM_WG_SZ * gid;

	uint64_2 w2[4]; loadg2(4, w2, &weight2[id], N_SZ / 4);
	const uint64_4 w = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);
	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);

	const uint_8_4 wd = width4[id];

	uint64_4 u = mod_mul4(x[id], wi);
	u = adc4(u, wd, carry[gid]);
	x[id] = mod_mul4(u, w);
}

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
	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);

	uint32 c = a;
	while (c != 0)
	{
		// Unweight, sub with carry, weight
		for (size_t k = 0; k < N_SZ; ++k)
		{
			const uint64_2 w = weight2[k / 4 + (k % 4) * (N_SZ / 4)];
			x[k] = mod_mul(sbc(mod_mul(x[k], w.s1), width[k], &c), w.s0);
			if (c == 0) return;
		}
	}
}