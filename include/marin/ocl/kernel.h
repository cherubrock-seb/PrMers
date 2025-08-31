/*
Copyright 2025, Yves Gallot

marin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2025, Yves Gallot\n" \
"\n" \
"marin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#if __OPENCL_VERSION__ >= 120\n" \
"	#define INLINE	static inline\n" \
"#else\n" \
"	#define INLINE\n" \
"#endif\n" \
"\n" \
"#if defined(__NV_CL_C_VERSION)\n" \
"	#define PTX_ASM	1\n" \
"#endif\n" \
"\n" \
"#if !defined(N_SZ)\n" \
"#define N_SZ		65536u\n" \
"#define W_F1		4611686017353646079ul\n" \
"#define W_F2		5818851782451133869ul\n" \
"#define W_F3		10808002860802937880ul\n" \
"#define W_F4		1418753320236437486ul\n" \
"#define W_F5		7970496220330062908ul\n" \
"#define BLK16		16u\n" \
"#define BLK64		4u\n" \
"#define BLK40		32u\n" \
"#define BLK160		8u\n" \
"#define BLK640		2u\n" \
"#define CHUNK16		16u\n" \
"#define CHUNK64		4u\n" \
"#define CHUNK256	4u\n" \
"#define CHUNK16_5	8u\n" \
"#define CHUNK64_5	2u\n" \
"#define CWM_WG_SZ	32u\n" \
"#define CWM_WG_SZ2	16u\n" \
"#define MAX_WG_SZ	256\n" \
"#endif\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uchar	uint_8;\n" \
"typedef uint	uint32;\n" \
"typedef int		int32;\n" \
"typedef ulong	uint64;\n" \
"typedef uchar4	uint_8_4;\n" \
"typedef ulong2	uint64_2;\n" \
"typedef ulong4	uint64_4;\n" \
"\n" \
"INLINE sz_t div5(const sz_t n) { return mul_hi(n, 858993460u); }	// = n / 5 if n < 2^30\n" \
"\n" \
"// --- modular arithmetic ---\n" \
"\n" \
"#define	MOD_P		0xffffffff00000001ul		// 2^64 - 2^32 + 1\n" \
"#define	MOD_MP64	0xffffffffu					// -p mod (2^64) = 2^32 - 1\n" \
"\n" \
"// t = 2^64 * hi + lo modulo p. We must have t < p^2.\n" \
"INLINE uint64 reduce(const uint64 lo, const uint64 hi)\n" \
"{\n" \
"	const uint32 hi_hi = (uint32)(hi >> 32), hi_lo = (uint32)(hi);\n" \
"\n" \
"	// Let X = hi_lo * (2^32 - 1) - hi_hi + lo = hi_hi * 2^96 + hi_lo * 2^64 + lo (mod p)\n" \
"	// The trick is to add 2^32 - 1 to X (Nick Craig-Wood, ARM-32 assembly code)\n" \
"	const uint32 d = MOD_MP64 - hi_hi;\n" \
"	const uint64 s = upsample(hi_lo, d) - hi_lo;	// No carry: 0 <= s <= (2^32 - 1)^2 + 2^32 - 1 = 2^64 - 2^32 < p\n" \
"#if defined(PTX_ASM)\n" \
"	uint64 r; uint32 nc, c;\n" \
"	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=l\" (r) : \"l\" (s), \"l\" (lo));		// r = s + lo\n" \
"	asm volatile (\"addc.u32 %0, 0xffffffff, 0;\" : \"=r\" (nc));							// If no carry then nc = MOD_MP64 else nc = 0\n" \
"	const uint64 nc64 = upsample(0, nc);\n" \
"	asm volatile (\"sub.cc.u64 %0, %1, %2;\" : \"=l\" (r) : \"l\" (r), \"l\" (nc64));	// r -= nc\n" \
"	asm volatile (\"subc.u32 %0, 0, 0;\" : \"=r\" (c));								// If borrow then c = MOD_MP64 else c = 0\n" \
"	const uint64 c64 = upsample(0, c);\n" \
"	asm volatile (\"sub.cc.u64 %0, %1, %2;\" : \"=l\" (r) : \"l\" (r), \"l\" (c64));	// r -= c\n" \
"#else\n" \
"	uint64 r = s + lo;		// If carry then r + 2^64 = X + 2^32 - 1. We have r = X (mod p) and r < s < p\n" \
"	if (r >= s)				// No carry\n" \
"	{\n" \
"		// Subtract 2^32 - 1. If the difference is negative then add p (+p = -(-p))\n" \
"		const uint32 c = (r < MOD_MP64) ? MOD_MP64 : 0;	// borrow\n" \
"		r -= MOD_MP64; r -= c;\n" \
"	}\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint64 mod_add(const uint64 lhs, const uint64 rhs) { return lhs + rhs + ((lhs >= MOD_P - rhs) ? MOD_MP64 : 0); }\n" \
"INLINE uint64 mod_sub(const uint64 lhs, const uint64 rhs) { return lhs - rhs - ((lhs < rhs) ? MOD_MP64 : 0); }\n" \
"INLINE uint64 mod_mul(const uint64 lhs, const uint64 rhs) { return reduce(lhs * rhs, mul_hi(lhs, rhs)); }\n" \
"INLINE uint64 mod_sqr(const uint64 lhs) { return mod_mul(lhs, lhs); }\n" \
"INLINE uint64 mod_muli(const uint64 lhs) { return reduce(lhs << 48, lhs >> (64 - 48)); }	// sqrt(-1) = 2^48 (mod p)\n" \
"INLINE uint64 mod_half(const uint64 lhs) { return ((lhs % 2 == 0) ? lhs / 2 : ((lhs - 1) / 2 + (MOD_P + 1) / 2)); }\n" \
"\n" \
"INLINE uint64_2 mod_add2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_add(lhs.s0, rhs.s0), mod_add(lhs.s1, rhs.s1)); }\n" \
"INLINE uint64_2 mod_sub2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_sub(lhs.s0, rhs.s0), mod_sub(lhs.s1, rhs.s1)); }\n" \
"INLINE uint64_2 mod_mul2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_mul(lhs.s0, rhs.s0), mod_mul(lhs.s1, rhs.s1)); }\n" \
"INLINE uint64_2 mod_sqr2(const uint64_2 lhs) { return (uint64_2)(mod_sqr(lhs.s0), mod_sqr(lhs.s1)); }\n" \
"INLINE uint64_2 mod_muli2(const uint64_2 lhs) { return (uint64_2)(mod_muli(lhs.s0), mod_muli(lhs.s1)); }\n" \
"INLINE uint64_2 mod_half2(const uint64_2 lhs) { return (uint64_2)(mod_half(lhs.s0), mod_half(lhs.s1)); }\n" \
"\n" \
"INLINE uint64_4 mod_add4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_add2(lhs.s01, rhs.s01), mod_add2(lhs.s23, rhs.s23)); }\n" \
"INLINE uint64_4 mod_sub4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_sub2(lhs.s01, rhs.s01), mod_sub2(lhs.s23, rhs.s23)); }\n" \
"INLINE uint64_4 mod_mul4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_mul2(lhs.s01, rhs.s01), mod_mul2(lhs.s23, rhs.s23)); }\n" \
"INLINE uint64_4 mod_half4(const uint64_4 lhs) { return (uint64_4)(mod_half2(lhs.s01), mod_half2(lhs.s23)); }\n" \
"\n" \
"// Add a carry onto the number and return the carry of the first width bits\n" \
"INLINE uint32 adc(const uint64 lhs, const uint_8 width, uint64 * const carry)\n" \
"{\n" \
"	const uint64 s = lhs + *carry;\n" \
"	const uint64 c = (s < lhs) ? 1 : 0;\n" \
"	*carry = (s >> width) + (c << (64 - width));\n" \
"	return (uint32)(s) & ((1u << width) - 1);\n" \
"}\n" \
"\n" \
"// Add carry and mul\n" \
"INLINE uint32 adc_mul(const uint64 lhs, const uint32 a, const uint_8 width, uint64 * const carry)\n" \
"{\n" \
"	uint64 c = 0;\n" \
"	const uint32 d = adc(lhs, width, &c);\n" \
"	const uint32 r = adc((uint64)(d) * a, width, carry);\n" \
"	*carry += a * c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint64_4 adc4(const uint64_4 lhs, const uint_8_4 width, const uint64 carry)\n" \
"{\n" \
"	uint64_4 r;\n" \
"	uint64 c = carry;\n" \
"	r.s0 = adc(lhs.s0, width.s0, &c);\n" \
"	r.s1 = adc(lhs.s1, width.s1, &c);\n" \
"	r.s2 = adc(lhs.s2, width.s2, &c);\n" \
"	r.s3 = lhs.s3 + c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint64_4 adc_mul4(const uint64_4 lhs, const uint32 a, const uint_8_4 width, uint64 * const carry)\n" \
"{\n" \
"	uint64_4 r;\n" \
"	r.s0 = adc_mul(lhs.s0, a, width.s0, carry);\n" \
"	r.s1 = adc_mul(lhs.s1, a, width.s1, carry);\n" \
"	r.s2 = adc_mul(lhs.s2, a, width.s2, carry);\n" \
"	r.s3 = adc_mul(lhs.s3, a, width.s3, carry);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// Subtract a carry and return the carry if borrowing\n" \
"INLINE uint64 sbc(const uint64 lhs, const uint_8 width, uint32 * const carry)\n" \
"{\n" \
"	const bool borrow = (lhs < *carry);\n" \
"	const uint64 r = lhs - *carry + (borrow ? (1u << width) : 0);\n" \
"	*carry = borrow ? 1 : 0;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// --- transform - inline ---\n" \
"\n" \
"// Radix-2\n" \
"#define fwd2(x, r) \\\n" \
"{ \\\n" \
"	const uint64 u0 = x.s0, u1 = mod_mul(x.s1, r); \\\n" \
"	x.s0 = mod_add(u0, u1); x.s1 = mod_sub(u0, u1); \\\n" \
"}\n" \
"\n" \
"#define fwd2_2(x, r) \\\n" \
"{ \\\n" \
"	const uint64_2 u0 = x[0], u1 = mod_mul2(x[1], r); \\\n" \
"	x[0] = mod_add2(u0, u1); x[1] = mod_sub2(u0, u1); \\\n" \
"}\n" \
"\n" \
"#define fwd2_4(x, r) \\\n" \
"{ \\\n" \
"	const uint64_2 u0 = x[0], u1 = mod_mul2(x[1], r.s0), u2 = x[2], u3 = mod_mul2(x[3], r.s1); \\\n" \
"	x[0] = mod_add2(u0, u1); x[1] = mod_sub2(u0, u1); x[2] = mod_add2(u2, u3); x[3] = mod_sub2(u2, u3); \\\n" \
"}\n" \
"\n" \
"// Inverse radix-2\n" \
"#define bck2(x, ri) \\\n" \
"{ \\\n" \
"	const uint64 u0 = x.s0, u1 = x.s1; \\\n" \
"	x.s0 = mod_add(u0, u1); x.s1 = mod_mul(mod_sub(u0, u1), ri); \\\n" \
"}\n" \
"\n" \
"#define bck2_2(x, ri) \\\n" \
"{ \\\n" \
"	const uint64_2 u0 = x[0], u1 = x[1]; \\\n" \
"	x[0] = mod_add2(u0, u1); x[1] = mod_mul2(mod_sub2(u0, u1), ri); \\\n" \
"}\n" \
"\n" \
"#define bck2_4(x, ri) \\\n" \
"{ \\\n" \
"	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3]; \\\n" \
"	x[0] = mod_add2(u0, u1); x[1] = mod_mul2(mod_sub2(u0, u1), ri.s0); \\\n" \
"	x[2] = mod_add2(u2, u3); x[3] = mod_mul2(mod_sub2(u2, u3), ri.s1); \\\n" \
"}\n" \
"\n" \
"// Radix-4\n" \
"#define fwd4(x, r1, r2) \\\n" \
"{ \\\n" \
"	const uint64 u0 = x[0], u2 = mod_mul(x[2], r1), u1 = mod_mul(x[1], r2.s0), u3 = mod_mul(x[3], r2.s1); \\\n" \
"	const uint64 v0 = mod_add(u0, u2), v2 = mod_sub(u0, u2), v1 = mod_add(u1, u3), v3 = mod_muli(mod_sub(u1, u3)); \\\n" \
"	x[0] = mod_add(v0, v1); x[1] = mod_sub(v0, v1); x[2] = mod_add(v2, v3); x[3] = mod_sub(v2, v3); \\\n" \
"}\n" \
"\n" \
"#define fwd4_2(x, r1, r2) \\\n" \
"{ \\\n" \
"	const uint64_2 u0 = x[0], u2 = mod_mul2(x[2], r1), u1 = mod_mul2(x[1], r2.s0), u3 = mod_mul2(x[3], r2.s1); \\\n" \
"	const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3)); \\\n" \
"	x[0] = mod_add2(v0, v1); x[1] = mod_sub2(v0, v1); x[2] = mod_add2(v2, v3); x[3] = mod_sub2(v2, v3); \\\n" \
"}\n" \
"\n" \
"// Inverse radix-4\n" \
"#define bck4(x, ri1, ri2) \\\n" \
"{ \\\n" \
"	const uint64 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3]; \\\n" \
"	const uint64 v0 = mod_add(u0, u1), v1 = mod_sub(u0, u1), v2 = mod_add(u3, u2), v3 = mod_muli(mod_sub(u3, u2)); \\\n" \
"	x[0] = mod_add(v0, v2); x[2] = mod_mul(mod_sub(v0, v2), ri1); x[1] = mod_mul(mod_add(v1, v3), ri2.s0); x[3] = mod_mul(mod_sub(v1, v3), ri2.s1); \\\n" \
"}\n" \
"\n" \
"#define bck4_2(x, ri1, ri2) \\\n" \
"{ \\\n" \
"	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3]; \\\n" \
"	const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2)); \\\n" \
"	x[0] = mod_add2(v0, v2); x[2] = mod_mul2(mod_sub2(v0, v2), ri1); x[1] = mod_mul2(mod_add2(v1, v3), ri2.s0); x[3] = mod_mul2(mod_sub2(v1, v3), ri2.s1); \\\n" \
"}\n" \
"\n" \
"// squarex2 even\n" \
"#define sqr2_2(x, r) \\\n" \
"{ \\\n" \
"	const uint64 t = mod_mul(mod_sqr(x.s1), r); \\\n" \
"	x.s1 = mod_mul(mod_add(x.s0, x.s0), x.s1); \\\n" \
"	x.s0 = mod_add(mod_sqr(x.s0), t); \\\n" \
"}\n" \
"\n" \
"// squarex2 odd\n" \
"#define sqr2n_2(x, r) \\\n" \
"{ \\\n" \
"	const uint64 t = mod_mul(mod_sqr(x.s1), r); \\\n" \
"	x.s1 = mod_mul(mod_add(x.s0, x.s0), x.s1); \\\n" \
"	x.s0 = mod_sub(mod_sqr(x.s0), t); \\\n" \
"}\n" \
"\n" \
"// mulx2 even\n" \
"#define mul2_2(x, y, r) \\\n" \
"{ \\\n" \
"	const uint64 t = mod_mul(mod_mul(x.s1, y.s1), r); \\\n" \
"	x.s1 = mod_add(mod_mul(x.s0, y.s1), mod_mul(y.s0, x.s1)); \\\n" \
"	x.s0 = mod_add(mod_mul(x.s0, y.s0), t); \\\n" \
"}\n" \
"\n" \
"// mulx2 odd\n" \
"#define mul2n_2(x, y, r) \\\n" \
"{ \\\n" \
"	const uint64 t = mod_mul(mod_mul(x.s1, y.s1), r); \\\n" \
"	x.s1 = mod_add(mod_mul(x.s0, y.s1), mod_mul(y.s0, x.s1)); \\\n" \
"	x.s0 = mod_sub(mod_mul(x.s0, y.s0), t); \\\n" \
"}\n" \
"\n" \
"// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.\n" \
"#define butterfly5_2(a0, a1, a2, a3, a4) \\\n" \
"{ \\\n" \
"	const uint64_2 s1 = mod_add2(a1, a4), s2 = mod_sub2(a1, a4), s3 = mod_add2(a3, a2), s4 = mod_sub2(a3, a2); \\\n" \
"	const uint64_2 s5 = mod_add2(s1, s3), s6 = mod_sub2(s1, s3), s7 = mod_add2(s2, s4), s8 = mod_add2(s5, a0); \\\n" \
"	const uint64_2 m0 = s8; \\\n" \
"	const uint64_2 m1 = mod_mul2(s5, W_F1), m2 = mod_mul2(s6, W_F2), m3 = mod_mul2(s2, W_F3), m4 = mod_mul2(s7, W_F4), m5 = mod_mul2(s4, W_F5); \\\n" \
"	const uint64_2 s9 = mod_add2(m0, m1), s10 = mod_add2(s9, m2), s11 = mod_sub2(s9, m2), s12 = mod_sub2(m3, m4); \\\n" \
"	const uint64_2 s13 = mod_add2(m4, m5), s14 = mod_add2(s10, s12), s15 = mod_sub2(s10, s12), s16 = mod_add2(s11, s13); \\\n" \
"	const uint64_2 s17 = mod_sub2(s11, s13); \\\n" \
"	a0 = m0; a1 = s14; a2 = s16; a3 = s17; a4 = s15; \\\n" \
"}\n" \
"\n" \
"// Radix-5\n" \
"#define fwd5_2(x, r) \\\n" \
"{ \\\n" \
"	const uint64_2 r2 = mod_sqr2(r), r3 = mod_mul2(r, r2), r4 = mod_sqr2(r2); \\\n" \
"	uint64_2 a0 = x[0], a1 = mod_mul2(x[1], r), a2 = mod_mul2(x[2], r2), a3 = mod_mul2(x[3], r3), a4 = mod_mul2(x[4], r4); \\\n" \
"	butterfly5_2(a0, a1, a2, a3, a4); \\\n" \
"	x[0] = a0; x[1] = a1; x[2] = a2; x[3] = a3; x[4] = a4; \\\n" \
"}\n" \
"\n" \
"// Inverse radix-5\n" \
"#define bck5_2(x, ri) \\\n" \
"{ \\\n" \
"	uint64_2 a0 = x[0], a4 = x[1], a3 = x[2], a2 = x[3], a1 = x[4]; \\\n" \
"	butterfly5_2(a0, a1, a2, a3, a4); \\\n" \
"	const uint64_2 ri2 = mod_sqr2(ri), ri3 = mod_mul2(ri, ri2), ri4 = mod_sqr2(ri2); \\\n" \
"	x[0] = a0; x[1] = mod_mul2(a1, ri); x[2] = mod_mul2(a2, ri2); x[3] = mod_mul2(a3, ri3); x[4] = mod_mul2(a4, ri4); \\\n" \
"}\n" \
"\n" \
"// Transpose of matrices\n" \
"#define transpose_52(y, x) \\\n" \
"{ \\\n" \
"	y[0] = (uint64_2)(x[0].s0, x[2].s1); \\\n" \
"	y[1] = (uint64_2)(x[0].s1, x[3].s0); \\\n" \
"	y[2] = (uint64_2)(x[1].s0, x[3].s1); \\\n" \
"	y[3] = (uint64_2)(x[1].s1, x[4].s0); \\\n" \
"	y[4] = (uint64_2)(x[2].s0, x[4].s1); \\\n" \
"}\n" \
"\n" \
"#define transpose_25(y, x) \\\n" \
"{ \\\n" \
"	y[0] = (uint64_2)(x[0].s0, x[1].s0); \\\n" \
"	y[1] = (uint64_2)(x[2].s0, x[3].s0); \\\n" \
"	y[2] = (uint64_2)(x[4].s0, x[0].s1); \\\n" \
"	y[3] = (uint64_2)(x[1].s1, x[2].s1); \\\n" \
"	y[4] = (uint64_2)(x[3].s1, x[4].s1); \\\n" \
"}\n" \
"\n" \
"INLINE void loadg1(const sz_t n, uint64 * const xl, __global const uint64 * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }\n" \
"INLINE void storeg1(const sz_t n, __global uint64 * restrict const x, const sz_t s, const uint64 * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }\n" \
"\n" \
"INLINE void loadg2(const sz_t n, uint64_2 * const xl, __global const uint64_2 * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }\n" \
"INLINE void loadl2(const sz_t n, uint64_2 * const xl, __local const uint64_2 * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }\n" \
"INLINE void storeg2(const sz_t n, __global uint64_2 * restrict const x, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }\n" \
"INLINE void storel2(const sz_t n, __local uint64_2 * restrict const X, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }\n" \
"\n" \
"// --- transform - global mem ---\n" \
"\n" \
"#if N_SZ % 5 != 0\n" \
"\n" \
"#if N_SZ <= 1024\n" \
"\n" \
"// 2 x Radix-4\n" \
"__kernel\n" \
"void forward4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 2]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);\n" \
"	const uint64 r1 = r2[j]; const uint64_2 r23 = r4[j];\n" \
"	fwd4_2(xl, r1, r23);\n" \
"	storeg2(4, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"// 2 x Inverse radix-4\n" \
"__kernel\n" \
"void backward4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ];\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 2]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);\n" \
"	const uint64 r1i = r2i[j]; const uint64_2 r23i = r4i[j];\n" \
"	bck4_2(xl, r1i, r23i);\n" \
"	storeg2(4, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if N_SZ == 4\n" \
"\n" \
"// Radix-2\n" \
"__kernel\n" \
"void forward_mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r0 = &root[0];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;\n" \
"\n" \
"	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);\n" \
"	const uint64 r = r0[j]; fwd2_2(xl, r);\n" \
"	storeg2(2, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// Radix-2, square2x2, inverse radix-2\n" \
"__kernel\n" \
"void sqr4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r0 = &root[0];\n" \
"	__global const uint64 * restrict const r0i = &root[N_SZ];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;\n" \
"\n" \
"	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);\n" \
"	const uint64 r = r0[j]; fwd2_2(xl, r);\n" \
"	sqr2_2(xl[0], r); sqr2n_2(xl[1], r);\n" \
"	const uint64 ri = r0i[j]; bck2_2(xl, ri);\n" \
"	storeg2(2, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// Radix-2, mul2x2, inverse radix-2\n" \
"__kernel\n" \
"void mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"	__global const uint64 * restrict const r0 = &root[0];\n" \
"	__global const uint64 * restrict const r0i = &root[N_SZ];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;\n" \
"\n" \
"	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);\n" \
"	const uint64 r = r0[j]; fwd2_2(xl, r);\n" \
"	uint64_2 yl[2]; loadg2(2, yl, &y[k], 1);\n" \
"	mul2_2(xl[0], yl[0], r); mul2n_2(xl[1], yl[1], r);\n" \
"	const uint64 ri = r0i[j]; bck2_2(xl, ri);\n" \
"	storeg2(2, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"#elif (N_SZ == 8) || (N_SZ == 16)\n" \
"\n" \
"// 2 x Radix-2\n" \
"__kernel\n" \
"void forward_mul4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	const uint64_2 r = r0[j]; fwd2_4(xl, r);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// 2 x Radix-2, square2x2, inverse radix-2\n" \
"__kernel\n" \
"void sqr4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0];\n" \
"	__global const uint64_2 * restrict const r0i = (__global const uint64_2 *)&root[N_SZ];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	const uint64_2 r = r0[j]; fwd2_4(xl, r);\n" \
"	sqr2_2(xl[0], r.s0); sqr2n_2(xl[1], r.s0); sqr2_2(xl[2], r.s1); sqr2n_2(xl[3], r.s1);\n" \
"	const uint64_2 ri = r0i[j]; bck2_4(xl, ri);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// 2 x Radix-2, mul2x2, inverse radix-2\n" \
"__kernel\n" \
"void mul4x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0];\n" \
"	__global const uint64_2 * restrict const r0i = (__global const uint64_2 *)&root[N_SZ];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	const uint64_2 r = r0[j]; fwd2_4(xl, r);\n" \
"	uint64_2 yl[4]; loadg2(4, yl, &y[k], 1);\n" \
"	mul2_2(xl[0], yl[0], r.s0); mul2n_2(xl[1], yl[1], r.s0); mul2_2(xl[2], yl[2], r.s1); mul2n_2(xl[3], yl[3], r.s1);\n" \
"	const uint64_2 ri = r0i[j]; bck2_4(xl, ri);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"#else	// N_SZ % 5 != 0\n" \
"\n" \
"#if (N_SZ == 2560) || (N_SZ == 5120)\n" \
"\n" \
"// 2 x Radix-4, 5 | N_SZ\n" \
"__kernel\n" \
"void forward4x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 5 / 2]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), m = 1u << lm, m5 = 5u << lm;\n" \
"	const sz_t j = id_5 >> lm, k = 3 * 5 * (id_5 & ~(m - 1)) + id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], m5);\n" \
"	const uint64 r1 = r2[j]; const uint64_2 r23 = r4[j];\n" \
"	fwd4_2(xl, r1, r23);\n" \
"	storeg2(4, &x[k], m5, xl);\n" \
"}\n" \
"\n" \
"// 2 x Inverse radix-4, 5 | N_SZ\n" \
"__kernel\n" \
"void backward4x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ];\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5 / 2]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), m = 1u << lm, m5 = 5u << lm;\n" \
"	const sz_t j = id_5 >> lm, k = 3 * 5 * (id_5 & ~(m - 1)) + id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], m5);\n" \
"	const uint64 r1i = r2i[j]; const uint64_2 r23i = r4i[j];\n" \
"	bck4_2(xl, r1i, r23i);\n" \
"	storeg2(4, &x[k], m5, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"/*\n" \
"\n" \
"// Radix-2, radix-5\n" \
"__kernel\n" \
"void forward_mul10(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;\n" \
"	if (id_mod5 != 4)\n" \
"	{\n" \
"		const sz_t id4 = 4 * id_5 + id_mod5, j = id4, k = 5 * id4;\n" \
"\n" \
"		uint64_2 xl[5], xt[5]; loadg2(5, xl, &x[k], 1); transpose_52(xt, xl);\n" \
"		const uint64 r_2 = r2[j]; for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r_2);\n" \
"		const uint64_2 r_5 = r5[j]; fwd5_2(xt, r_5);\n" \
"		transpose_25(xl, xt); storeg2(5, &x[k], 1, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"// Radix-2, radix-5, square, inverse radix-5, inverse radix-2\n" \
"__kernel\n" \
"void sqr10(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ];\n" \
"	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]);\n" \
"	__global const uint64_2 * restrict const r5i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;\n" \
"	if (id_mod5 != 4)\n" \
"	{\n" \
"		const sz_t id4 = 4 * id_5 + id_mod5, j = id4, k = 5 * id4;\n" \
"\n" \
"		uint64_2 xl[5], xt[5]; loadg2(5, xl, &x[k], 1); transpose_52(xt, xl);\n" \
"		const uint64 r = r2[j]; for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r);\n" \
"		const uint64_2 r_5 = r5[j]; fwd5_2(xt, r_5);\n" \
"		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_sqr2(xt[i]);\n" \
"		const uint64_2 r_5i = r5i[j]; bck5_2(xt, r_5i);\n" \
"		const uint64 ri = r2i[j]; for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], ri);\n" \
"		transpose_25(xl, xt); storeg2(5, &x[k], 1, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"// Radix-2, radix-5, mul, inverse radix-5, inverse radix-2\n" \
"__kernel\n" \
"void mul10(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ];\n" \
"	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]);\n" \
"	__global const uint64_2 * restrict const r5i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;\n" \
"	if (id_mod5 != 4)\n" \
"	{\n" \
"		const sz_t id4 = 4 * id_5 + id_mod5, j = id4, k = 5 * id4;\n" \
"\n" \
"		uint64_2 xl[5], xt[5]; loadg2(5, xl, &x[k], 1); transpose_52(xt, xl);\n" \
"		const uint64 r = r2[j]; for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r);\n" \
"		const uint64_2 r_5 = r5[j]; fwd5_2(xt, r_5);\n" \
"		uint64_2 yl[5], yt[5]; loadg2(5, yl, &y[k], 1); transpose_52(yt, yl);\n" \
"		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_mul2(xt[i], yt[i]);\n" \
"		const uint64_2 r_5i = r5i[j]; bck5_2(xt, r_5i);\n" \
"		const uint64 ri = r2i[j]; for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], ri);\n" \
"		transpose_25(xl, xt); storeg2(5, &x[k], 1, xl);\n" \
"	}\n" \
"}\n" \
"*/\n" \
"\n" \
"#endif	// N_SZ % 5 != 0\n" \
"\n" \
"// --- transform - local mem ---\n" \
"\n" \
"INLINE void forward_4i(const sz_t ml, __local uint64_2 * restrict const X,\n" \
"	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	uint64_2 xl[4]; loadg2(4, xl, x, mg);\n" \
"	fwd4_2(xl, r1, r23);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4(const sz_t ml, __local uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	fwd4_2(xl, r1, r23);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4o(const sz_t mg, __global uint64_2 * restrict const x,\n" \
"	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	fwd4_2(xl, r1, r23);\n" \
"	storeg2(4, x, mg, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const sz_t ml, __local uint64_2 * restrict const X,\n" \
"	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	uint64_2 xl[4]; loadg2(4, xl, x, mg);\n" \
"	bck4_2(xl, r1i, r23i);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4(const sz_t ml, __local uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	bck4_2(xl, r1i, r23i);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const sz_t mg, __global uint64_2 * restrict const x,\n" \
"	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	bck4_2(xl, r1i, r23i);\n" \
"	storeg2(4, x, mg, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4x2o(__global uint64_2 * restrict const x, __local uint64_2 * restrict const X, const uint64_2 r)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	fwd2_4(xl, r);\n" \
"	storeg2(4, x, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_10o(__global uint64_2 * restrict const x, __local uint64_2 * restrict const X, const uint64 r2, const uint64_2 r5, const bool cond)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (cond)\n" \
"	{\n" \
"		uint64_2 xl[5], xt[5]; loadl2(5, xl, X, 1); transpose_52(xt, xl);\n" \
"		for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r2);\n" \
"		fwd5_2(xt, r5);\n" \
"		transpose_25(xl, xt); storeg2(5, x, 1, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"INLINE void square_4x2(__local uint64_2 * restrict const X, const uint64_2 r, const uint64_2 ri)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	fwd2_4(xl, r);\n" \
"	sqr2_2(xl[0], r.s0); sqr2n_2(xl[1], r.s0); sqr2_2(xl[2], r.s1); sqr2n_2(xl[3], r.s1);\n" \
"	bck2_4(xl, ri);\n" \
"	storel2(4, X, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void mul_4x2(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y, const uint64_2 r, const uint64_2 ri)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	fwd2_4(xl, r);\n" \
"	uint64_2 yl[4]; loadg2(4, yl, y, 1);\n" \
"	mul2_2(xl[0], yl[0], r.s0); mul2n_2(xl[1], yl[1], r.s0); mul2_2(xl[2], yl[2], r.s1); mul2n_2(xl[3], yl[3], r.s1);\n" \
"	bck2_4(xl, ri);\n" \
"	storel2(4, X, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void square_10(__local uint64_2 * restrict const X, const uint64 r2, const uint64 r2i, const uint64_2 r5, const uint64_2 r5i, const bool cond)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (cond)\n" \
"	{\n" \
"		uint64_2 xl[5], xt[5]; loadl2(5, xl, X, 1); transpose_52(xt, xl);\n" \
"		for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r2);\n" \
"		fwd5_2(xt, r5);\n" \
"		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_sqr2(xt[i]);\n" \
"		bck5_2(xt, r5i);\n" \
"		for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], r2i);\n" \
"		transpose_25(xl, xt); storel2(5, X, 1, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"INLINE void mul_10(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y,\n" \
"	const uint64 r2, const uint64 r2i, const uint64_2 r5, const uint64_2 r5i, const bool cond)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (cond)\n" \
"	{\n" \
"		uint64_2 xl[5], xt[5]; loadl2(5, xl, X, 1); transpose_52(xt, xl);\n" \
"		for (sz_t i = 0; i <= 4; ++i) fwd2(xt[i], r2);\n" \
"		fwd5_2(xt, r5);\n" \
"		uint64_2 yl[5], yt[5]; loadg2(5, yl, y, 1); transpose_52(yt, yl);\n" \
"		for (sz_t i = 0; i <= 4; ++i) xt[i] = mod_mul2(xt[i], yt[i]);\n" \
"		bck5_2(xt, r5i);\n" \
"		for (sz_t i = 0; i <= 4; ++i) bck2(xt[i], r2i);\n" \
"		transpose_25(xl, xt); storel2(5, X, 1, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"#if N_SZ % 5 != 0\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]); \\\n" \
"	__global const uint64_2 * restrict const r0 = (__global const uint64_2 *)&root[0]; \\\n" \
"	__global const uint64_2 * restrict const r0i = (__global const uint64_2 *)&root[N_SZ]; \\\n" \
"	__global const uint64 * restrict const r2 = &root[0]; \\\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ]; \\\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 2]); \\\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 2]); \\\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	\\\n" \
"	__local uint64_2 X[4 * B_N * CHUNK_N]; \\\n" \
"	\\\n" \
"	/* thread_idx < B_N */ \\\n" \
"	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \\\n" \
"	const sz_t i = local_id, chunk_idx = i % CHUNK_N, thread_idx = i / CHUNK_N, block_idx = group_id * CHUNK_N + chunk_idx; \\\n" \
"	__local uint64_2 * const Xi = &X[chunk_idx]; \\\n" \
"	\\\n" \
"	const sz_t block_idx_m = block_idx >> lm, idx_m = block_idx_m * B_N + thread_idx; \\\n" \
"	const sz_t block_idx_mm = block_idx_m << lm, idx_mm = idx_m << lm; \\\n" \
"	\\\n" \
"	const sz_t ki = block_idx + block_idx_mm * (B_N * 3 - 1) + idx_mm, ko = block_idx - block_idx_mm + idx_mm * 4; \\\n" \
"	const sz_t j = idx_m;\n" \
"\n" \
"#if ((N_SZ == 4096) || (N_SZ == 8192)) && (MAX_WG_SZ >= 16 / 4 * CHUNK16)\n" \
"\n" \
"#define ATTR_FB_16x2()	__attribute__((reqd_work_group_size(16 / 4 * CHUNK16, 1, 1)))\n" \
"\n" \
"// 2 x Radix-4, radix-4\n" \
"__kernel\n" \
"ATTR_FB_16x2()\n" \
"void forward16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(16 / 4, CHUNK16);\n" \
"\n" \
"	forward_4i(4 * CHUNK16, &X[i], 4u << lm, &x[ki], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"// 2 x Inverse radix-4, radix-4\n" \
"__kernel\n" \
"ATTR_FB_16x2()\n" \
"void backward16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(16 / 4, CHUNK16);\n" \
"\n" \
"	backward_4i(1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	backward_4o(4u << lm, &x[ki], 4 * CHUNK16, &X[i], r2i[j / 4], r4i[j / 4]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 16384) && (MAX_WG_SZ >= 64 / 4 * CHUNK64)\n" \
"\n" \
"#define ATTR_FB_64x2()	__attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64x2()\n" \
"void forward64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(64 / 4, CHUNK64);\n" \
"\n" \
"	forward_4i(16 * CHUNK64, &X[i], 16u << lm, &x[ki], r2[j / 16], r4[j / 16]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	forward_4(4 * CHUNK64, &Xi[CHUNK64 * i4], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK64, &Xi[CHUNK64 * 4 * thread_idx], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64x2()\n" \
"void backward64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(64 / 4, CHUNK64);\n" \
"\n" \
"	backward_4i(1 * CHUNK64, &Xi[CHUNK64 * 4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	backward_4(4 * CHUNK64, &Xi[CHUNK64 * i4], r2i[j / 4], r4i[j / 4]);\n" \
"	backward_4o(16u << lm, &x[ki], 16 * CHUNK64, &X[i], r2i[j / 16], r4i[j / 16]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 65536) && (MAX_WG_SZ >= 256 / 4 * CHUNK256)\n" \
"\n" \
"#define ATTR_FB_256x2()	__attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256x2()\n" \
"void forward256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(256 / 4, CHUNK256);\n" \
"\n" \
"	forward_4i(64 * CHUNK256, &X[i], 64u << lm, &x[ki], r2[j / 64], r4[j / 64]);\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);\n" \
"	forward_4(16 * CHUNK256, &Xi[CHUNK256 * i16], r2[j / 16], r4[j / 16]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	forward_4(4 * CHUNK256, &Xi[CHUNK256 * i4], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK256, &Xi[CHUNK256 * 4 * thread_idx], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256x2()\n" \
"void backward256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(256 / 4, CHUNK256);\n" \
"\n" \
"	backward_4i(1 * CHUNK256, &Xi[CHUNK256 * 4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	backward_4(4 * CHUNK256, &Xi[CHUNK256 * i4], r2i[j / 4], r4i[j / 4]);\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);\n" \
"	backward_4(16 * CHUNK256, &Xi[CHUNK256 * i16], r2i[j / 16], r4i[j / 16]);\n" \
"	backward_4o(64u << lm, &x[ki], 64 * CHUNK256, &X[i], r2i[j / 64], r4i[j / 64]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 262144) && (MAX_WG_SZ >= 1024 / 4)\n" \
"\n" \
"#define ATTR_FB_1024x2()	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_1024x2()\n" \
"void forward1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(1024 / 4, 1);\n" \
"\n" \
"	forward_4i(256, &X[i], 256u << lm, &x[ki], r2[j / 256], r4[j / 256]);\n" \
"	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64);\n" \
"	forward_4(64, &Xi[i64], r2[j / 64], r4[j / 64]);\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);\n" \
"	forward_4(16, &Xi[i16], r2[j / 16], r4[j / 16]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	forward_4(4, &Xi[i4], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(1u << lm, &x[ko], 1, &Xi[4 * thread_idx], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_1024x2()\n" \
"void backward1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(1024 / 4, 1);\n" \
"\n" \
"	backward_4i(1, &Xi[4 * thread_idx], 1u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	backward_4(4, &Xi[i4], r2i[j / 4], r4i[j / 4]);\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);\n" \
"	backward_4(16, &Xi[i16], r2i[j / 16], r4i[j / 16]);\n" \
"	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64);\n" \
"	backward_4(64, &Xi[i64], r2i[j / 64], r4i[j / 64]);\n" \
"	backward_4o(256u << lm, &x[ki], 256, &X[i], r2i[j / 256], r4i[j / 256]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"////////////////////////////////////\n" \
"\n" \
"#if ((N_SZ == 32) || (N_SZ == 64)) && (MAX_WG_SZ >= 16 / 4 * BLK16)\n" \
"\n" \
"#define DECLARE_VAR_16x2() \\\n" \
"	__local uint64_2 X[16 * BLK16]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, k = 4 * id, i = k % (16 * BLK16); \\\n" \
"	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % (16 * BLK16);\n" \
"\n" \
"#define ATTR_16x2()	__attribute__((reqd_work_group_size(16 / 4 * BLK16, 1, 1)))\n" \
"\n" \
"// 2 x Radix-4, 2 x radix-2\n" \
"__kernel\n" \
"ATTR_16x2()\n" \
"void forward_mul16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_16x2();\n" \
"\n" \
"	forward_4i(2, &X[i4], 2, &x[k4], r2[j4], r4[j4]);\n" \
"	forward_4x2o(&x[k], &X[i], r0[j]);\n" \
"}\n" \
"\n" \
"// 2 x Radix-4, square4, inverse radix-4\n" \
"__kernel\n" \
"ATTR_16x2()\n" \
"void sqr16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_16x2();\n" \
"\n" \
"	forward_4i(2, &X[i4], 2, &x[k4], r2[j4], r4[j4]);\n" \
"	square_4x2(&X[i], r0[j], r0i[j]);\n" \
"	backward_4o(2, &x[k4], 2, &X[i4], r2i[j4], r4i[j4]);\n" \
"}\n" \
"\n" \
"// 2 x Radix-4, mul4, inverse radix-4\n" \
"__kernel\n" \
"ATTR_16x2()\n" \
"void mul16x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_16x2();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(2, &X[i4], 2, &x[k4], r2[j4], r4[j4]);\n" \
"	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);\n" \
"	backward_4o(2, &x[k4], 2, &X[i4], r2i[j4], r4i[j4]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if ((N_SZ == 128) || (N_SZ == 256)) && (MAX_WG_SZ >= 64 / 4 * BLK64)\n" \
"\n" \
"#define DECLARE_VAR_64x2() \\\n" \
"	__local uint64_2 X[64 * BLK64]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, k = 4 * id, i = k % (64 * BLK64); \\\n" \
"	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % (64 * BLK64); \\\n" \
"	const sz_t j16 = id / 8, k16 = 4 * (id & ~(8 - 1)) + (id % 8), i16 = k16 % (64 * BLK64);\n" \
"\n" \
"#define ATTR_64x2()	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_64x2()\n" \
"void forward_mul64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_64x2();\n" \
"\n" \
"	forward_4i(8, &X[i16], 8, &x[k16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4x2o(&x[k], &X[i], r0[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64x2()\n" \
"void sqr64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_64x2();\n" \
"\n" \
"	forward_4i(8, &X[i16], 8, &x[k16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	square_4x2(&X[i], r0[j], r0i[j]);\n" \
"	backward_4(2, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4o(8, &x[k16], 8, &X[i16], r2i[j16], r4i[j16]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64x2()\n" \
"void mul64x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_64x2();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(8, &X[i16], 8, &x[k16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);\n" \
"	backward_4(2, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4o(8, &x[k16], 8, &X[i16], r2i[j16], r4i[j16]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ > 256) && (MAX_WG_SZ >= 256 / 4)\n" \
"\n" \
"#define ATTR_256x2()	__attribute__((reqd_work_group_size(256 / 4, 1, 1)))\n" \
"\n" \
"#define DECLARE_VAR_256x2() \\\n" \
"	__local uint64_2 X[256]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, k = 4 * id, i = k % 256; \\\n" \
"	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % 256; \\\n" \
"	const sz_t j16 = id / 8, k16 = 4 * (id & ~(8 - 1)) + (id % 8), i16 = k16 % 256; \\\n" \
"	const sz_t j64 = id / 32, k64 = 4 * (id & ~(32 - 1)) + (id % 32), i64 = k64 % 256;\n" \
"\n" \
"__kernel\n" \
"ATTR_256x2()\n" \
"void forward_mul256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_256x2();\n" \
"\n" \
"	forward_4i(32, &X[i64], 32, &x[k64], r2[j64], r4[j64]);\n" \
"	forward_4(8, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4x2o(&x[k], &X[i], r0[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256x2()\n" \
"void sqr256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_256x2();\n" \
"\n" \
"	forward_4i(32, &X[i64], 32, &x[k64], r2[j64], r4[j64]);\n" \
"	forward_4(8, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	square_4x2(&X[i], r0[j], r0i[j]);\n" \
"	backward_4(2, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4(8, &X[i16], r2i[j16], r4i[j16]);\n" \
"	backward_4o(32, &x[k64], 32, &X[i64], r2i[j64], r4i[j64]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256x2()\n" \
"void mul256x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_256x2();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(32, &X[i64], 32, &x[k64], r2[j64], r4[j64]);\n" \
"	forward_4(8, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);\n" \
"	backward_4(2, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4(8, &X[i16], r2i[j16], r4i[j16]);\n" \
"	backward_4o(32, &x[k64], 32, &X[i64], r2i[j64], r4i[j64]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ > 256) && (MAX_WG_SZ >= 1024 / 4)\n" \
"\n" \
"#define ATTR_1024x2()	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"\n" \
"#define DECLARE_VAR_1024x2() \\\n" \
"	__local uint64_2 X[1024]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, k = 4 * id, i = k % 1024; \\\n" \
"	const sz_t j4 = id / 2, k4 = 4 * (id & ~(2 - 1)) + (id % 2), i4 = k4 % 1024; \\\n" \
"	const sz_t j16 = id / 8, k16 = 4 * (id & ~(8 - 1)) + (id % 8), i16 = k16 % 1024; \\\n" \
"	const sz_t j64 = id / 32, k64 = 4 * (id & ~(32 - 1)) + (id % 32), i64 = k64 % 1024; \\\n" \
"	const sz_t j256 = id / 128, k256 = 4 * (id & ~(128 - 1)) + (id % 128), i256 = k256 % 1024;\n" \
"\n" \
"__kernel\n" \
"ATTR_1024x2()\n" \
"void forward_mul1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_1024x2();\n" \
"\n" \
"	forward_4i(128, &X[i256], 128, &x[k256], r2[j256], r4[j256]);\n" \
"	forward_4(32, &X[i64], r2[j64], r4[j64]);\n" \
"	forward_4(8, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4x2o(&x[k], &X[i], r0[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024x2()\n" \
"void sqr1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_1024x2();\n" \
"\n" \
"	forward_4i(128, &X[i256], 128, &x[k256], r2[j256], r4[j256]);\n" \
"	forward_4(32, &X[i64], r2[j64], r4[j64]);\n" \
"	forward_4(8, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	square_4x2(&X[i], r0[j], r0i[j]);\n" \
"	backward_4(2, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4(8, &X[i16], r2i[j16], r4i[j16]);\n" \
"	backward_4(32, &X[i64], r2i[j64], r4i[j64]);\n" \
"	backward_4o(128, &x[k256], 128, &X[i256], r2i[j256], r4i[j256]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024x2()\n" \
"void mul1024x2(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_1024x2();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(128, &X[i256], 128, &x[k256], r2[j256], r4[j256]);\n" \
"	forward_4(32, &X[i64], r2[j64], r4[j64]);\n" \
"	forward_4(8, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(2, &X[i4], r2[j4], r4[j4]);\n" \
"	mul_4x2(&X[i], &y[k], r0[j], r0i[j]);\n" \
"	backward_4(2, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4(8, &X[i16], r2i[j16], r4i[j16]);\n" \
"	backward_4(32, &X[i64], r2i[j64], r4i[j64]);\n" \
"	backward_4o(128, &x[k256], 128, &X[i256], r2i[j256], r4i[j256]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"#else	// N_SZ % 5 != 0\n" \
"\n" \
"#define DECLARE_VAR_REG_5() \\\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]); \\\n" \
"	__global const uint64 * restrict const r2 = &root[0]; \\\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ]; \\\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ / 5 / 2]); \\\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5 / 2]); \\\n" \
"	const sz_t id = (sz_t)get_global_id(0), id_5 = div5(id), id_mod5 = id - 5 * id_5;\n" \
"\n" \
"#define DECLARE_VAR5(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR_REG_5(); \\\n" \
"	\\\n" \
"	__local uint64_2 X[5 * 4 * B_N * CHUNK_N]; \\\n" \
"	\\\n" \
"	/* thread_idx < B_N */ \\\n" \
"	const sz_t local_id = id_5 % (B_N * CHUNK_N), group_id = id_5 / (B_N * CHUNK_N); \\\n" \
"	const sz_t i = local_id, chunk_idx = i % CHUNK_N, thread_idx = i / CHUNK_N, block_idx = group_id * CHUNK_N + chunk_idx; \\\n" \
"	__local uint64_2 * const Xi = &X[5 * chunk_idx]; \\\n" \
"	\\\n" \
"	const sz_t block_idx_m = block_idx >> lm, idx_m = block_idx_m * B_N + thread_idx; \\\n" \
"	const sz_t block_idx_mm = block_idx_m << lm, idx_mm = idx_m << lm; \\\n" \
"	\\\n" \
"	const sz_t ki = 5 * (block_idx + block_idx_mm * (B_N * 3 - 1) + idx_mm) + id_mod5; \\\n" \
"	const sz_t ko = 5 * (block_idx - block_idx_mm + idx_mm * 4) + id_mod5; \\\n" \
"	const sz_t j = idx_m;\n" \
"\n" \
"#if (N_SZ >= 10240) && (MAX_WG_SZ >= 5 * 16 / 4 * CHUNK16_5)\n" \
"\n" \
"#define ATTR_FB_16x2_5()	__attribute__((reqd_work_group_size(5 * 16 / 4 * CHUNK16_5, 1, 1)))\n" \
"\n" \
"// 2 x Radix-4, radix-4, 5 | N_SZ\n" \
"__kernel\n" \
"ATTR_FB_16x2_5()\n" \
"void forward16x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR5(16 / 4, CHUNK16_5);\n" \
"\n" \
"	forward_4i(20 * CHUNK16_5, &X[5 * i + id_mod5], 20u << lm, &x[ki], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(5u << lm, &x[ko], 5 * CHUNK16_5, &Xi[5 * CHUNK16_5 * 4 * thread_idx + id_mod5], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"// 2 x Inverse radix-4, radix-4, 5 | N_SZ\n" \
"__kernel\n" \
"ATTR_FB_16x2_5()\n" \
"void backward16x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR5(16 / 4, CHUNK16_5);\n" \
"\n" \
"	backward_4i(5 * CHUNK16_5, &Xi[5 * CHUNK16_5 * 4 * thread_idx + id_mod5], 5u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	backward_4o(20u << lm, &x[ki], 20 * CHUNK16_5, &X[5 * i + id_mod5], r2i[j / 4], r4i[j / 4]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 10240) && (MAX_WG_SZ >= 5 * 64 / 4 * CHUNK64_5)\n" \
"\n" \
"#define ATTR_FB_64x2_5()	__attribute__((reqd_work_group_size(5 * 64 / 4 * CHUNK64_5, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64x2_5()\n" \
"void forward64x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR5(64 / 4, CHUNK64_5);\n" \
"\n" \
"	forward_4i(80 * CHUNK64_5, &X[5 * i + id_mod5], 80u << lm, &x[ki], r2[j / 16], r4[j / 16]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	forward_4(20 * CHUNK64_5, &Xi[5 * CHUNK64_5 * i4 + id_mod5], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(5u << lm, &x[ko], 5 * CHUNK64_5, &Xi[5 * CHUNK64_5 * 4 * thread_idx + id_mod5], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64x2_5()\n" \
"void backward64x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR5(64 / 4, CHUNK64_5);\n" \
"\n" \
"	backward_4i(5 * CHUNK64_5, &Xi[5 * CHUNK64_5 * 4 * thread_idx + id_mod5], 5u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	backward_4(20 * CHUNK64_5, &Xi[5 * CHUNK64_5 * i4 + id_mod5], r2i[j / 4], r4i[j / 4]);\n" \
"	backward_4o(80u << lm, &x[ki], 80 * CHUNK64_5, &X[5 * i + id_mod5], r2i[j / 16], r4i[j / 16]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 10240) && (MAX_WG_SZ >= 5 * 256 / 4)\n" \
"\n" \
"#define ATTR_FB_256x2_5()	__attribute__((reqd_work_group_size(5 * 256 / 4, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256x2_5()\n" \
"void forward256x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR5(256 / 4, 1);\n" \
"\n" \
"	forward_4i(320, &X[5 * i + id_mod5], 320u << lm, &x[ki], r2[j / 64], r4[j / 64]);\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);\n" \
"	forward_4(80, &Xi[5 * i16 + id_mod5], r2[j / 16], r4[j / 16]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	forward_4(20, &Xi[5 * i4 + id_mod5], r2[j / 4], r4[j / 4]);\n" \
"	forward_4o(5u << lm, &x[ko], 5, &Xi[5 * (4 * thread_idx) + id_mod5], r2[j], r4[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256x2_5()\n" \
"void backward256x2_5(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR5(256 / 4, 1);\n" \
"\n" \
"	backward_4i(5, &Xi[5 * 4 * thread_idx + id_mod5], 5u << lm, &x[ko], r2i[j], r4i[j]);\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4);\n" \
"	backward_4(20, &Xi[5 * i4 + id_mod5], r2i[j / 4], r4i[j / 4]);\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16);\n" \
"	backward_4(80, &Xi[5 * i16 + id_mod5], r2i[j / 16], r4i[j / 16]);\n" \
"	backward_4o(320u << lm, &x[ki], 320, &X[5 * i + id_mod5], r2i[j / 64], r4i[j / 64]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"////////////////////////////////////\n" \
"\n" \
"#define DECLARE_VAR_REG5() \\\n" \
"	DECLARE_VAR_REG_5(); \\\n" \
"	__global const uint64 * restrict const r0 = &root[0]; \\\n" \
"	__global const uint64 * restrict const r0i = &root[N_SZ]; \\\n" \
"	__global const uint64_2 * restrict const r5 = (__global const uint64_2 *)(&root[N_SZ / 5]); \\\n" \
"	__global const uint64_2 * restrict const r5i = (__global const uint64_2 *)(&root[N_SZ + N_SZ / 5]); \\\n" \
"	const sz_t local_id = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"\n" \
"#define WGSIZE40	(40 / 8 * BLK40)\n" \
"\n" \
"#if ((N_SZ == 40) || (N_SZ == 80)) && (MAX_WG_SZ >= WGSIZE40)\n" \
"\n" \
"#define ATTR_40()	__attribute__((reqd_work_group_size(WGSIZE40, 1, 1)))\n" \
"\n" \
"#define DECLARE_VAR_40() \\\n" \
"	__local uint64_2 X[20 * BLK40]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG5(); \\\n" \
"	const sz_t lid4 = local_id, id4 = 4 * WGSIZE40 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \\\n" \
"	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, k1 = 5 * t1 + id_mod5, i1 = 5 * (t1 % (20 * BLK40 / 5)) + id_mod5;\n" \
"\n" \
"// Radix-4, radix-2, radix-5\n" \
"__kernel\n" \
"ATTR_40()\n" \
"void forward_mul40(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_40();\n" \
"\n" \
"	forward_4i(1 * 5, &X[i1], 1 * 5, &x[k1], r2[j1], r4[j1]);\n" \
"	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE40 / 5);\n" \
"}\n" \
"\n" \
"// Radix-4, radix-2, radix-5, square, inverse radix-5, inverse radix-2, inverse radix-4\n" \
"__kernel\n" \
"ATTR_40()\n" \
"void sqr40(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_40();\n" \
"\n" \
"	forward_4i(1 * 5, &X[i1], 1 * 5, &x[k1], r2[j1], r4[j1]);\n" \
"	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE40 / 5);\n" \
"	backward_4o(1 * 5, &x[k1], 1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"}\n" \
"\n" \
"// Radix-4, radix-2, radix-5, mul, inverse radix-5, inverse radix-2, inverse radix-4\n" \
"__kernel\n" \
"ATTR_40()\n" \
"void mul40(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_40();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(1 * 5, &X[i1], 1 * 5, &x[k1], r2[j1], r4[j1]);\n" \
"	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE40 / 5);\n" \
"	backward_4o(1 * 5, &x[k1], 1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"#define WGSIZE160	(160 / 8 * BLK160)\n" \
"\n" \
"#if ((N_SZ == 160) || (N_SZ == 320)) && (MAX_WG_SZ >= WGSIZE160)\n" \
"\n" \
"#define ATTR_160()	__attribute__((reqd_work_group_size(WGSIZE160, 1, 1)))\n" \
"\n" \
"#define DECLARE_VAR_160() \\\n" \
"	__local uint64_2 X[80 * BLK160]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG5(); \\\n" \
"	const sz_t lid4 = local_id, id4 = 4 * WGSIZE160 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \\\n" \
"	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, /*k1 = 5 * t1 + id_mod5,*/ i1 = 5 * (t1 % (80 * BLK160 / 5)) + id_mod5; \\\n" \
"	const sz_t j4 = id_5 / 4, t4 = 4 * (id_5 & ~(4 - 1)) + (id_5 % 4), k4 = 5 * t4 + id_mod5, i4 = 5 * (t4 % (80 * BLK160 / 5)) + id_mod5;\n" \
"\n" \
"__kernel\n" \
"ATTR_160()\n" \
"void forward_mul160(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_160();\n" \
"\n" \
"	forward_4i(4 * 5, &X[i4], 4 * 5, &x[k4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE160 / 5);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_160()\n" \
"void sqr160(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_160();\n" \
"\n" \
"	forward_4i(4 * 5, &X[i4], 4 * 5, &x[k4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE160 / 5);\n" \
"	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"	backward_4o(4 * 5, &x[k4], 4 * 5, &X[i4], r2i[j4], r4i[j4]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_160()\n" \
"void mul160(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_160();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(4 * 5, &X[i4], 4 * 5, &x[k4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE160 / 5);\n" \
"	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"	backward_4o(4 * 5, &x[k4], 4 * 5, &X[i4], r2i[j4], r4i[j4]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"#define WGSIZE640	(640 / 8 * BLK640)\n" \
"\n" \
"#if (N_SZ > 320) && (MAX_WG_SZ >= WGSIZE640)\n" \
"\n" \
"#define ATTR_640() __attribute__((reqd_work_group_size(WGSIZE640, 1, 1)))\n" \
"\n" \
"#define DECLARE_VAR_640() \\\n" \
"	__local uint64_2 X[320 * BLK640]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG5(); \\\n" \
"	const sz_t lid4 = local_id, id4 = 4 * WGSIZE640 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \\\n" \
"	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, /*k1 = 5 * t1 + id_mod5,*/ i1 = 5 * (t1 % (320 * BLK640 / 5)) + id_mod5; \\\n" \
"	const sz_t j4 = id_5 / 4, t4 = 4 * (id_5 & ~(4 - 1)) + (id_5 % 4), /*k4 = 5 * t4 + id_mod5,*/ i4 = 5 * (t4 % (320 * BLK640 / 5)) + id_mod5; \\\n" \
"	const sz_t j16 = id_5 / 16, t16 = 4 * (id_5 & ~(16 - 1)) + (id_5 % 16), k16 = 5 * t16 + id_mod5, i16 = 5 * (t16 % (320 * BLK640 / 5)) + id_mod5;\n" \
"\n" \
"__kernel\n" \
"ATTR_640()\n" \
"void forward_mul640(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_640();\n" \
"\n" \
"	forward_4i(16 * 5, &X[i16], 16 * 5, &x[k16], r2[j16], r4[j16]);\n" \
"	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE640 / 5);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_640()\n" \
"void sqr640(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_640();\n" \
"\n" \
"	forward_4i(16 * 5, &X[i16], 16 * 5, &x[k16], r2[j16], r4[j16]);\n" \
"	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE640 / 5);\n" \
"	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4o(16 * 5, &x[k16], 16 * 5, &X[i16], r2i[j16], r4i[j16]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_640()\n" \
"void mul640(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_640();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(16 * 5, &X[i16], 16 * 5, &x[k16], r2[j16], r4[j16]);\n" \
"	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE640 / 5);\n" \
"	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4o(16 * 5, &x[k16], 16 * 5, &X[i16], r2i[j16], r4i[j16]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"#define WGSIZE2560	(2560 / 8)\n" \
"\n" \
"#if (N_SZ > 320) && (MAX_WG_SZ >= WGSIZE2560)\n" \
"\n" \
"#define ATTR_2560() __attribute__((reqd_work_group_size(WGSIZE2560, 1, 1)))\n" \
"\n" \
"#define DECLARE_VAR_2560() \\\n" \
"	__local uint64_2 X[1280]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG5(); \\\n" \
"	const sz_t lid4 = local_id, id4 = 4 * WGSIZE2560 / 5 * group_id + lid4, j = id4, k = 5 * id4, i = 5 * lid4; \\\n" \
"	const sz_t j1 = id_5 / 1, t1 = 4 * id_5, /*k1 = 5 * t1 + id_mod5,*/ i1 = 5 * (t1 % (1280 / 5)) + id_mod5; \\\n" \
"	const sz_t j4 = id_5 / 4, t4 = 4 * (id_5 & ~(4 - 1)) + (id_5 % 4), /*k4 = 5 * t4 + id_mod5,*/ i4 = 5 * (t4 % (1280 / 5)) + id_mod5; \\\n" \
"	const sz_t j16 = id_5 / 16, t16 = 4 * (id_5 & ~(16 - 1)) + (id_5 % 16), /*k16 = 5 * t16 + id_mod5,*/ i16 = 5 * (t16 % (1280 / 5)) + id_mod5; \\\n" \
"	const sz_t j64 = id_5 / 64, t64 = 4 * (id_5 & ~(64 - 1)) + (id_5 % 64), k64 = 5 * t64 + id_mod5, i64 = local_id;	// 5 * (t64 % (1280 / 5)) + id_mod5;\n" \
"\n" \
"__kernel\n" \
"ATTR_2560()\n" \
"void forward_mul2560(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_2560();\n" \
"\n" \
"	forward_4i(64 * 5, &X[i64], 64 * 5, &x[k64], r2[j64], r4[j64]);\n" \
"	forward_4(16 * 5, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	forward_10o(&x[k], &X[i], r0[j], r5[j], lid4 < 4 * WGSIZE2560 / 5);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_2560()\n" \
"void sqr2560(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_2560();\n" \
"\n" \
"	forward_4i(64 * 5, &X[i64], 64 * 5, &x[k64], r2[j64], r4[j64]);\n" \
"	forward_4(16 * 5, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	square_10(&X[i], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE2560 / 5);\n" \
"	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4(16 * 5, &X[i16], r2i[j16], r4i[j16]);\n" \
"	backward_4o(64 * 5, &x[k64], 64 * 5, &X[i64], r2i[j64], r4i[j64]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_2560()\n" \
"void mul2560(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_2560();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(64 * 5, &X[i64], 64 * 5, &x[k64], r2[j64], r4[j64]);\n" \
"	forward_4(16 * 5, &X[i16], r2[j16], r4[j16]);\n" \
"	forward_4(4 * 5, &X[i4], r2[j4], r4[j4]);\n" \
"	forward_4(1 * 5, &X[i1], r2[j1], r4[j1]);\n" \
"	mul_10(&X[i], &y[k], r0[j], r0i[j], r5[j], r5i[j], lid4 < 4 * WGSIZE2560 / 5);\n" \
"	backward_4(1 * 5, &X[i1], r2i[j1], r4i[j1]);\n" \
"	backward_4(4 * 5, &X[i4], r2i[j4], r4i[j4]);\n" \
"	backward_4(16 * 5, &X[i16], r2i[j16], r4i[j16]);\n" \
"	backward_4o(64 * 5, &x[k64], 64 * 5, &X[i64], r2i[j64], r4i[j64]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"#endif	// N_SZ % 5 != 0\n" \
"\n" \
"// --- carry ---\n" \
"\n" \
"#if defined(CWM_WG_SZ)\n" \
"\n" \
"// Unweight, carry, mul by a, weight (pass 1)\n" \
"__kernel\n" \
"void carry_weight_mul_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const uint32 a, const sz_t offset)\n" \
"{\n" \
"	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);\n" \
"	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);\n" \
"	__global const uint64_4 * restrict const weight4i_n = (__global const uint64_4 *)(&weight[N_SZ]);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;\n" \
"	__local uint64 cl[CWM_WG_SZ];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ;\n" \
"\n" \
"	const uint_8_4 wd = width4[gid];\n" \
"\n" \
"	uint64 c = 0;\n" \
"	uint64_4 u = mod_mul4(x[gid], weight4i_n[gid]);\n" \
"	u = adc_mul4(u, a, wd, &c);\n" \
"\n" \
"	cl[lid] = c;\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);\n" \
"	x[gid] = mod_mul4(u, weight4[gid]);\n" \
"\n" \
"	if (lid == CWM_WG_SZ - 1)\n" \
"	{\n" \
"		carry[(gid != N_SZ / 4 - 1) ? gid / CWM_WG_SZ + 1 : 0] = c;\n" \
"	}\n" \
"}\n" \
"\n" \
"// Unweight, carry, mul by a, weight (pass 2)\n" \
"__kernel\n" \
"void carry_weight_mul_p2(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const sz_t offset)\n" \
"{\n" \
"	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);\n" \
"	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);\n" \
"	__global const uint64_4 * restrict const weight4i = (__global const uint64_4 *)(&weight[2 * N_SZ]);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), id = CWM_WG_SZ * gid;\n" \
"\n" \
"	const uint_8_4 wd = width4[id];\n" \
"\n" \
"	uint64_4 u = mod_mul4(x[id], weight4i[id]);\n" \
"	u = adc4(u, wd, carry[gid]);\n" \
"	x[id] = mod_mul4(u, weight4[id]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if defined(CWM_WG_SZ2)\n" \
"\n" \
"#define N_SZ_8	(N_SZ / 8)\n" \
"\n" \
"// Inverse radix-2, unweight, carry, mul by a, weight, radix-2 (pass 1)\n" \
"__kernel\n" \
"void carry_weight_mul2_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const uint32 a, const sz_t offset)\n" \
"{\n" \
"	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);\n" \
"	__global uint64_2 * restrict const carry2 = (__global uint64_2 *)carry;\n" \
"	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);\n" \
"	__global const uint64_4 * restrict const weight4i_n = (__global const uint64_4 *)(&weight[N_SZ]);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;\n" \
"	__local uint64_2 cl[CWM_WG_SZ2];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ2;\n" \
"\n" \
"	const uint_8_4 wd0 = width4[gid + 0 * N_SZ_8], wd1 = width4[gid + 1 * N_SZ_8];\n" \
"\n" \
"	uint64 c1_0 = 0, c1_1 = 0;\n" \
"	const uint64_4 x0 = x[gid + 0 * N_SZ_8], x1 = x[gid + 1 * N_SZ_8];\n" \
"	uint64_4 u0 = mod_add4(x0, x1), u1 = mod_sub4(x0, x1);\n" \
"	u0 = mod_mul4(u0, weight4i_n[gid + 0 * N_SZ_8]);\n" \
"	u1 = mod_mul4(u1, weight4i_n[gid + 1 * N_SZ_8]);\n" \
"	u0 = adc_mul4(u0, a, wd0, &c1_0);\n" \
"	u1 = adc_mul4(u1, a, wd1, &c1_1);\n" \
"\n" \
"	cl[lid] = (uint64_2)(c1_0, c1_1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint64_2 c2 = (lid == 0) ? (uint64_2)(0, 0) : cl[lid - 1];\n" \
"	u0 = adc4(u0, wd0, c2.s0);\n" \
"	u1 = adc4(u1, wd1, c2.s1);\n" \
"	u0 = mod_mul4(u0, weight4[gid + 0 * N_SZ_8]);\n" \
"	u1 = mod_mul4(u1, weight4[gid + 1 * N_SZ_8]);\n" \
"	x[gid + 0 * N_SZ_8] = mod_add4(u0, u1);\n" \
"	x[gid + 1 * N_SZ_8] = mod_sub4(u0, u1);\n" \
"\n" \
"	if (lid == CWM_WG_SZ2 - 1)\n" \
"	{\n" \
"		const uint64_2 c1 = (gid != N_SZ_8 - 1) ? (uint64_2)(c1_0, c1_1) : (uint64_2)(c1_1, c1_0);\n" \
"		carry2[(gid != N_SZ_8 - 1) ? gid / CWM_WG_SZ2 + 1 : 0] = c1;\n" \
"	}\n" \
"}\n" \
"\n" \
"// Inverse radix-2, unweight, carry, mul by a, weight, radix-2 (pass 2)\n" \
"__kernel\n" \
"void carry_weight_mul2_p2(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const sz_t offset)\n" \
"{\n" \
"	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const carry2 = (__global const uint64_2 *)carry;\n" \
"	__global const uint64_4 * restrict const weight4 = (__global const uint64_4 *)(&weight[0]);\n" \
"	__global const uint64_4 * restrict const weight4i = (__global const uint64_4 *)(&weight[2 * N_SZ]);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)width;\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), id = CWM_WG_SZ2 * gid;\n" \
"\n" \
"	const uint_8_4 wd0 = width4[id + 0 * N_SZ_8], wd1 = width4[id + 1 * N_SZ_8];\n" \
"\n" \
"	const uint64_4 x0 = x[id + 0 * N_SZ_8], x1 = x[id + 1 * N_SZ_8];\n" \
"	uint64_4 u0 = mod_half4(mod_add4(x0, x1)), u1 = mod_half4(mod_sub4(x0, x1));\n" \
"	u0 = mod_mul4(u0, weight4i[id + 0 * N_SZ_8]);\n" \
"	u1 = mod_mul4(u1, weight4i[id + 1 * N_SZ_8]);\n" \
"	const uint64_2 c = carry2[gid];\n" \
"	u0 = adc4(u0, wd0, c.s0);\n" \
"	u1 = adc4(u1, wd1, c.s1);\n" \
"	u0 = mod_mul4(u0, weight4[id + 0 * N_SZ_8]);\n" \
"	u1 = mod_mul4(u1, weight4[id + 1 * N_SZ_8]);\n" \
"	x[id + 0 * N_SZ_8] = mod_add4(u0, u1);\n" \
"	x[id + 1 * N_SZ_8] = mod_sub4(u0, u1);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"// --- misc ---\n" \
"\n" \
"__kernel\n" \
"void copy(__global uint64 * restrict const reg, const sz_t offset_y, const sz_t offset_x)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	reg[offset_y + gid] = reg[offset_x + gid];\n" \
"}\n" \
"\n" \
"#if defined(CWM_WG_SZ)\n" \
"\n" \
"__kernel\n" \
"void subtract(__global uint64 * restrict const reg, __global const uint64 * restrict const weight,\n" \
"	__global const uint_8 * restrict const width, const sz_t offset, const uint32 a)\n" \
"{\n" \
"	__global uint64 * restrict const x = &reg[offset];\n" \
"	__global const uint64 * restrict const weighti = &weight[2 * N_SZ];\n" \
"\n" \
"	uint32 c = a;\n" \
"	while (c != 0)\n" \
"	{\n" \
"		// Unweight, sub with carry, weight\n" \
"		for (size_t k = 0; k < N_SZ; ++k)\n" \
"		{\n" \
"			x[k] = mod_mul(sbc(mod_mul(x[k], weighti[k]), width[k], &c), weight[k]);\n" \
"			if (c == 0) return;\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if defined(CWM_WG_SZ2)\n" \
"\n" \
"#define N_SZ_2	(N_SZ / 2)\n" \
"\n" \
"__kernel\n" \
"void subtract2(__global uint64 * restrict const reg, __global const uint64 * restrict const weight,\n" \
"	__global const uint_8 * restrict const width, const sz_t offset, const uint32 a)\n" \
"{\n" \
"	__global uint64 * restrict const x = &reg[offset];\n" \
"	__global const uint64 * restrict const weighti = &weight[2 * N_SZ];\n" \
"\n" \
"	uint32 c = a;\n" \
"	while (c != 0)\n" \
"	{\n" \
"		// Inverse radix-2, unweight, sub with carry, weight, radix-2\n" \
"		for (size_t k = 0; k < N_SZ_2; ++k)\n" \
"		{\n" \
"			const uint64 u0 = x[k + 0 * N_SZ_2], u1 = x[k + 1 * N_SZ_2];\n" \
"			const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));\n" \
"			const uint64 v0n = mod_mul(sbc(mod_mul(v0, weighti[k + 0 * N_SZ_2]), width[k + 0 * N_SZ_2], &c), weight[k + 0 * N_SZ_2]);\n" \
"			x[k + 0 * N_SZ_2] = mod_add(v0n, v1); x[k + 1 * N_SZ_2] = mod_sub(v0n, v1);\n" \
"			if (c == 0) return;\n" \
"		}\n" \
"		for (size_t k = 0; k < N_SZ_2; ++k)\n" \
"		{\n" \
"			const uint64 u0 = x[k + 0 * N_SZ_2], u1 = x[k + 1 * N_SZ_2];\n" \
"			const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));\n" \
"			const uint64 v1n = mod_mul(sbc(mod_mul(v1, weighti[k + 1 * N_SZ_2]), width[k + 1 * N_SZ_2], &c), weight[k + 1 * N_SZ_2]);\n" \
"			x[k + 0 * N_SZ_2] = mod_add(v0, v1n); x[k + 1 * N_SZ_2] = mod_sub(v0, v1n);\n" \
"			if (c == 0) return;\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"#endif\n" \
"";