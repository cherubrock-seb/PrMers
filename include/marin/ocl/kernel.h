/*
Copyright 2025, Yves Gallot

marin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

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
"#define LN_SZ_S5	14\n" \
"#define INV_N_2		18446181119461294081ul\n" \
"#define W_F1		4611686017353646079ul\n" \
"#define W_F2		5818851782451133869ul\n" \
"#define W_F3		10808002860802937880ul\n" \
"#define W_F4		1418753320236437486ul\n" \
"#define W_F5		7970496220330062908ul\n" \
"#define BLK16		16u\n" \
"#define BLK32		8u\n" \
"#define BLK64		4u\n" \
"#define BLK128		2u\n" \
"#define BLK256		1u\n" \
"#define BLK512		1u\n" \
"#define CHUNK16		16u\n" \
"#define CHUNK20		16u\n" \
"#define CHUNK64		8u\n" \
"#define CHUNK80		8u\n" \
"#define CHUNK256	4u\n" \
"#define CHUNK320	2u\n" \
"#define CWM_WG_SZ	256u\n" \
"#define MAX_WG_SZ	256u\n" \
"#endif\n" \
"\n" \
"#define sz_t		uint\n" \
"#define uint_8		uchar\n" \
"#ifndef uint32\n" \
"#define uint32		uint\n" \
"#endif\n" \
"#ifndef int32\n" \
"#define int32		int\n" \
"#endif\n" \
"#ifndef uint64\n" \
"#define uint64		ulong\n" \
"#endif\n" \
"#define uint_8_4	uchar4\n" \
"#define uint64_2	ulong2\n" \
"#define uint64_4	ulong4\n" \
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
"	asm volatile (\"addc.u32 %0, 0xffffffff, 0;\" : \"=r\" (nc));					// If no carry then nc = MOD_MP64 else nc = 0\n" \
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
"\n" \
"INLINE uint64_2 mod_add2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_add(lhs.s0, rhs.s0), mod_add(lhs.s1, rhs.s1)); }\n" \
"INLINE uint64_2 mod_sub2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_sub(lhs.s0, rhs.s0), mod_sub(lhs.s1, rhs.s1)); }\n" \
"INLINE uint64_2 mod_mul2(const uint64_2 lhs, const uint64_2 rhs) { return (uint64_2)(mod_mul(lhs.s0, rhs.s0), mod_mul(lhs.s1, rhs.s1)); }\n" \
"INLINE uint64_2 mod_sqr2(const uint64_2 lhs) { return (uint64_2)(mod_sqr(lhs.s0), mod_sqr(lhs.s1)); }\n" \
"INLINE uint64_2 mod_muli2(const uint64_2 lhs) { return (uint64_2)(mod_muli(lhs.s0), mod_muli(lhs.s1)); }\n" \
"\n" \
"INLINE uint64_4 mod_mul4(const uint64_4 lhs, const uint64_4 rhs) { return (uint64_4)(mod_mul2(lhs.s01, rhs.s01), mod_mul2(lhs.s23, rhs.s23)); }\n" \
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
"INLINE uint64 mask_w(const uint w) {\n" \
"    return (w >= 64) ? (uint64)(~(uint64)0) : (((uint64)1 << w) - 1ul);\n" \
"}\n" \
"\n" \
"INLINE uint64_4 neg_mp4(const uint64_4 v, const uint_8_4 wd) {\n" \
"    const uint64 m0 = mask_w(wd.s0);\n" \
"    const uint64 m1 = mask_w(wd.s1);\n" \
"    const uint64 m2 = mask_w(wd.s2);\n" \
"    const uint64 m3 = mask_w(wd.s3);\n" \
"    uint64_4 r;\n" \
"    r.s0 = (m0 - (v.s0 & m0));\n" \
"    r.s1 = (m1 - (v.s1 & m1));\n" \
"    r.s2 = (m2 - (v.s2 & m2));\n" \
"    r.s3 = (m3 - (v.s3 & m3));\n" \
"    return r;\n" \
"}\n" \
"\n" \
"INLINE uint64_4 adc_mul4(const uint64_4 lhs, const uint32 a, const uint_8_4 width, uint64 * const carry)\n" \
"{\n" \
"	uint64_4 r;\n" \
"	uint64 c = *carry;\n" \
"	r.s0 = adc_mul(lhs.s0, a, width.s0, &c);\n" \
"	r.s1 = adc_mul(lhs.s1, a, width.s1, &c);\n" \
"	r.s2 = adc_mul(lhs.s2, a, width.s2, &c);\n" \
"	r.s3 = adc_mul(lhs.s3, a, width.s3, &c);\n" \
"	*carry = c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint64_4 addc4(const uint64_4 lhs, const uint64_4 rhs, const uint_8_4 width, uint64 * const carry)\n" \
"{\n" \
"	uint64_4 r;\n" \
"	uint64 c = *carry;\n" \
"	c += rhs.s0; r.s0 = adc(lhs.s0, width.s0, &c);\n" \
"	c += rhs.s1; r.s1 = adc(lhs.s1, width.s1, &c);\n" \
"	c += rhs.s2; r.s2 = adc(lhs.s2, width.s2, &c);\n" \
"	c += rhs.s3; r.s3 = adc(lhs.s3, width.s3, &c);\n" \
"	*carry = c;\n" \
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
"// Radix-4\n" \
"INLINE void fwd4(uint64_2 * const x, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	const uint64_2 u0 = x[0], u2 = mod_mul2(x[2], r1), u1 = mod_mul2(x[1], r23.s0), u3 = mod_mul2(x[3], r23.s1);\n" \
"	const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3));\n" \
"	x[0] = mod_add2(v0, v1); x[1] = mod_sub2(v0, v1); x[2] = mod_add2(v2, v3); x[3] = mod_sub2(v2, v3);\n" \
"}\n" \
"\n" \
"// Inverse radix-4\n" \
"INLINE void bck4(uint64_2 * const x, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3];\n" \
"	const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2));\n" \
"	x[0] = mod_add2(v0, v2); x[2] = mod_mul2(mod_sub2(v0, v2), r1i); x[1] = mod_mul2(mod_add2(v1, v3), r23i.s0); x[3] = mod_mul2(mod_sub2(v1, v3), r23i.s1);\n" \
"}\n" \
"\n" \
"// Radix-4, first stage\n" \
"INLINE void fwd4_0(uint64_2 * const x)\n" \
"{\n" \
"	const uint64_2 u0 = x[0], u2 = x[2], u1 = x[1], u3 = x[3];\n" \
"	const uint64_2 v0 = mod_add2(u0, u2), v2 = mod_sub2(u0, u2), v1 = mod_add2(u1, u3), v3 = mod_muli2(mod_sub2(u1, u3));\n" \
"	x[0] = mod_add2(v0, v1); x[1] = mod_sub2(v0, v1); x[2] = mod_add2(v2, v3); x[3] = mod_sub2(v2, v3);\n" \
"}\n" \
"\n" \
"// Inverse radix-4, first stage\n" \
"INLINE void bck4_0(uint64_2 * const x)\n" \
"{\n" \
"	const uint64_2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3];\n" \
"	const uint64_2 v0 = mod_add2(u0, u1), v1 = mod_sub2(u0, u1), v2 = mod_add2(u3, u2), v3 = mod_muli2(mod_sub2(u3, u2));\n" \
"	x[0] = mod_add2(v0, v2); x[2] = mod_sub2(v0, v2); x[1] = mod_add2(v1, v3); x[3] = mod_sub2(v1, v3);\n" \
"}\n" \
"\n" \
"// 2 x radix-2\n" \
"INLINE void fwd22(uint64_2 * const x, const uint64_2 r)\n" \
"{\n" \
"	const uint64_2 u0 = x[0], u1 = mod_mul2(x[1], r);\n" \
"	x[0] = mod_add2(u0, u1); x[1] = mod_sub2(u0, u1);\n" \
"}\n" \
"\n" \
"// 2 x inverse radix-2\n" \
"INLINE void bck22(uint64_2 * const x, const uint64_2 ri)\n" \
"{\n" \
"	const uint64_2 u0 = x[0], u1 = x[1];\n" \
"	x[0] = mod_add2(u0, u1); x[1] = mod_mul2(mod_sub2(u0, u1), ri);\n" \
"}\n" \
"\n" \
"// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.\n" \
"#define butterfly5(a0, a1, a2, a3, a4) \\\n" \
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
"// Radix-5, first stage\n" \
"INLINE void fwd5_0(uint64_2 * const x)\n" \
"{\n" \
"	uint64_2 a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], a4 = x[4];\n" \
"	butterfly5(a0, a1, a2, a3, a4);\n" \
"	x[0] = a0; x[1] = a1; x[2] = a2; x[3] = a3; x[4] = a4;\n" \
"}\n" \
"	\n" \
"// Inverse radix-5, first stage\n" \
"INLINE void bck5_0(uint64_2 * const x)\n" \
"{\n" \
"	uint64_2 a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], a4 = x[4];\n" \
"	butterfly5(a0, a1, a2, a3, a4);\n" \
"	x[0] = a0; x[4] = a1; x[3] = a2; x[2] = a3; x[1] = a4;\n" \
"}\n" \
"\n" \
"// 2 x Radix-2, sqr, inverse radix-2\n" \
"INLINE void sqr22(uint64_2 * const x, const uint64 r)\n" \
"{\n" \
"	const uint64_2 sx0 = mod_sqr2(x[0]), sx1 = mod_sqr2(x[1]);\n" \
"	x[0].s1 = mod_mul(x[0].s1, mod_add(x[0].s0, x[0].s0)); x[0].s0 = mod_add(sx0.s0, mod_mul(sx0.s1, r));\n" \
"	x[1].s1 = mod_mul(x[1].s1, mod_add(x[1].s0, x[1].s0)); x[1].s0 = mod_sub(sx1.s0, mod_mul(sx1.s1, r));\n" \
"}\n" \
"\n" \
"// 2 x Radix-2, mul, inverse radix-2\n" \
"INLINE void mul22(uint64_2 * const x, const uint64_2 * const y, const uint64 r)\n" \
"{\n" \
"	const uint64_2 xy0 = mod_mul2(x[0], y[0]), xy1 = mod_mul2(x[1], y[1]);\n" \
"	x[0].s1 = mod_add(mod_mul(x[0].s0, y[0].s1), mod_mul(x[0].s1, y[0].s0)); x[0].s0 = mod_add(xy0.s0, mod_mul(xy0.s1, r));\n" \
"	x[1].s1 = mod_add(mod_mul(x[1].s0, y[1].s1), mod_mul(x[1].s1, y[1].s0)); x[1].s0 = mod_sub(xy1.s0, mod_mul(xy1.s1, r));\n" \
"}\n" \
"\n" \
"INLINE void sqr_4x1(uint64_2 * const xl, const uint64 r, const uint64 ri)\n" \
"{\n" \
"	fwd22(xl, r);\n" \
"	sqr22(xl, r);\n" \
"	bck22(xl, ri);\n" \
"}\n" \
"\n" \
"INLINE void mul_4x1(uint64_2 * const xl, const uint64_2 * const yl, const uint64 r, const uint64 ri)\n" \
"{\n" \
"	fwd22(xl, r);\n" \
"	mul22(xl, yl, r);\n" \
"	bck22(xl, ri);\n" \
"}\n" \
"\n" \
"INLINE void sqr_4(uint64_2 * const xl, const uint64_2 r, const uint64_2 ri)\n" \
"{\n" \
"	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);\n" \
"	sqr22(&xl[0], r.s0); sqr22(&xl[2], r.s1);\n" \
"	bck22(&xl[0], ri.s0); bck22(&xl[2], ri.s1);\n" \
"}\n" \
"\n" \
"INLINE void mul_4(uint64_2 * const xl, const uint64_2 * const yl, const uint64_2 r, const uint64_2 ri)\n" \
"{\n" \
"	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);\n" \
"	mul22(&xl[0], &yl[0], r.s0); mul22(&xl[2], &yl[2], r.s1);\n" \
"	bck22(&xl[0], ri.s0); bck22(&xl[2], ri.s1);\n" \
"}\n" \
"\n" \
"INLINE void sqr_8(uint64_2 * const xl, const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	fwd4(xl, r1, r23);\n" \
"	sqr22(&xl[0], r23.s0);\n" \
"	sqr22(&xl[2], mod_muli(r23.s0));\n" \
"	bck4(xl, r1i, r23i);\n" \
"}\n" \
"\n" \
"INLINE void mul_8(uint64_2 * const xl, const uint64_2 * const yl, const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	fwd4(xl, r1, r23);\n" \
"	mul22(&xl[0], &yl[0], r23.s0);\n" \
"	mul22(&xl[2], &yl[2], mod_muli(r23.s0));\n" \
"	bck4(xl, r1i, r23i);\n" \
"}\n" \
"\n" \
"INLINE void loadg2(const sz_t n, uint64_2 * const xl, __global const uint64_2 * restrict const x, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = x[l * s]; }\n" \
"INLINE void loadl2(const sz_t n, uint64_2 * const xl, __local const uint64_2 * restrict const X, const sz_t s) { for (sz_t l = 0; l < n; ++l) xl[l] = X[l * s]; }\n" \
"INLINE void storeg2(const sz_t n, __global uint64_2 * restrict const x, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) x[l * s] = xl[l]; }\n" \
"INLINE void storel2(const sz_t n, __local uint64_2 * restrict const X, const sz_t s, const uint64_2 * const xl) { for (sz_t l = 0; l < n; ++l) X[l * s] = xl[l]; }\n" \
"\n" \
"// --- transform - global mem ---\n" \
"\n" \
"// Radix-4\n" \
"/*__kernel\n" \
"void forward4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t s, const uint32 lm)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);\n" \
"	fwd4(xl, r2[sj], r4[sj]);\n" \
"	storeg2(4, &x[k], m, xl);\n" \
"}\n" \
"\n" \
"// Inverse radix-4\n" \
"__kernel\n" \
"void backward4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t s, const uint32 lm)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ / 2];\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), m = 1u << lm, sj = s + (id >> lm), k = 3 * (id & ~(m - 1)) + id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], m);\n" \
"	bck4(xl, r2i[sj], r4i[sj]);\n" \
"	storeg2(4, &x[k], m, xl);\n" \
"}*/\n" \
"\n" \
"#if (N_SZ % 5 != 0) && (N_SZ <= 32)\n" \
"\n" \
"// Radix-4, first stage\n" \
"__kernel\n" \
"void forward4_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), k = id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], N_SZ / 8);\n" \
"	fwd4_0(xl);\n" \
"	storeg2(4, &x[k], N_SZ / 8, xl);\n" \
"}\n" \
"\n" \
"// Inverse radix-4, first stage\n" \
"__kernel\n" \
"void backward4_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), k = id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], N_SZ / 8);\n" \
"	bck4_0(xl);\n" \
"	storeg2(4, &x[k], N_SZ / 8, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ == 40)\n" \
"\n" \
"// Radix-5, first stage\n" \
"__kernel\n" \
"void forward5_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), k = id;\n" \
"\n" \
"	if (k < N_SZ / 10)\n" \
"	{\n" \
"		uint64_2 xl[5]; loadg2(5, xl, &x[k], N_SZ / 10);\n" \
"		fwd5_0(xl);\n" \
"		storeg2(5, &x[k], N_SZ / 10, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"// Inverse radix-5, first stage\n" \
"__kernel\n" \
"void backward5_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), k = id;\n" \
"\n" \
"	if (k < N_SZ / 10)\n" \
"	{\n" \
"		uint64_2 xl[5]; loadg2(5, xl, &x[k], N_SZ / 10);\n" \
"		bck5_0(xl);\n" \
"		storeg2(5, &x[k], N_SZ / 10, xl);\n" \
"	}\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ == 4)\n" \
"\n" \
"// Radix-4\n" \
"__kernel\n" \
"void forward_mul4x1(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;\n" \
"\n" \
"	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);\n" \
"	fwd22(xl, r2[N_SZ / 4 + j]);\n" \
"	storeg2(2, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// Radix-4, square, inverse radix-4\n" \
"__kernel\n" \
"void sqr4x1(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ / 2];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;\n" \
"\n" \
"	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);\n" \
"	sqr_4x1(xl, r2[N_SZ / 4 + j], r2i[N_SZ / 4 + j]);\n" \
"	storeg2(2, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// Radix-4, mul, inverse radix-4\n" \
"__kernel\n" \
"void mul4x1(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ / 2];\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 2 * id;\n" \
"\n" \
"	uint64_2 xl[2]; loadg2(2, xl, &x[k], 1);\n" \
"	uint64_2 yl[2]; loadg2(2, yl, &y[k], 1);\n" \
"	mul_4x1(xl, yl, r2[N_SZ / 4 + j], r2i[N_SZ / 4 + j]);\n" \
"	storeg2(2, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 16) && (N_SZ <= 80)\n" \
"\n" \
"// 2 x Radix-4\n" \
"__kernel\n" \
"void forward_mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const r2 = (__global const uint64_2 *)(&root[0]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	const uint64_2 r = r2[N_SZ / 8 + j];\n" \
"	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// 2 x Radix-4, square, inverse radix-4\n" \
"__kernel\n" \
"void sqr4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const r2 = (__global const uint64_2 *)(&root[0]);\n" \
"	__global const uint64_2 * restrict const r2i = (__global const uint64_2 *)(&root[N_SZ / 2]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	sqr_4(xl, r2[N_SZ / 8 + j], r2i[N_SZ / 8 + j]);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// 2 x Radix-4, mul, inverse radix-4\n" \
"__kernel\n" \
"void mul4(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"	__global const uint64_2 * restrict const r2 = (__global const uint64_2 *)(&root[0]);\n" \
"	__global const uint64_2 * restrict const r2i = (__global const uint64_2 *)(&root[N_SZ / 2]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	uint64_2 yl[4]; loadg2(4, yl, &y[k], 1);\n" \
"	mul_4(xl, yl, r2[N_SZ / 8 + j], r2i[N_SZ / 8 + j]);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ >= 8) && (N_SZ <= 160)\n" \
"\n" \
"// Radix-8\n" \
"__kernel\n" \
"void forward_mul8(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	fwd4(xl, r2[N_SZ / 8 + j], r4[N_SZ / 8 + j]);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// Radix-8, square, inverse radix-8\n" \
"__kernel\n" \
"void sqr8(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ / 2];\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	const uint64 r1 = r2[N_SZ / 8 + j]; const uint64_2 r23 = r4[N_SZ / 8 + j];\n" \
"	const uint64 r1i = r2i[N_SZ / 8 + j]; const uint64_2 r23i = r4i[N_SZ / 8 + j];\n" \
"	sqr_8(xl, r1, r23, r1i, r23i);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"// Radix-8, mul, inverse radix-8\n" \
"__kernel\n" \
"void mul8(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset_x, const sz_t offset_y)\n" \
"{\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"	__global const uint64 * restrict const r2 = &root[0];\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ / 2];\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]);\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]);\n" \
"\n" \
"	const sz_t id = (sz_t)get_global_id(0), j = id, k = 4 * id;\n" \
"\n" \
"	uint64_2 xl[4]; loadg2(4, xl, &x[k], 1);\n" \
"	uint64_2 yl[4]; loadg2(4, yl, &y[k], 1);\n" \
"	const uint64 r1 = r2[N_SZ / 8 + j]; const uint64_2 r23 = r4[N_SZ / 8 + j];\n" \
"	const uint64 r1i = r2i[N_SZ / 8 + j]; const uint64_2 r23i = r4i[N_SZ / 8 + j];\n" \
"	mul_8(xl, yl, r1, r23, r1i, r23i);\n" \
"	storeg2(4, &x[k], 1, xl);\n" \
"}\n" \
"\n" \
"#endif\n" \
"\n" \
"// --- transform - local mem ---\n" \
"\n" \
"INLINE void forward_4(const sz_t m, __local uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, m);\n" \
"	fwd4(xl, r1, r23);\n" \
"	storel2(4, X, m, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i(const sz_t ml, __local uint64_2 * restrict const X,\n" \
"	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	uint64_2 xl[4]; loadg2(4, xl, x, mg);\n" \
"	fwd4(xl, r1, r23);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4o(const sz_t mg, __global uint64_2 * restrict const x,\n" \
"	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	fwd4(xl, r1, r23);\n" \
"	storeg2(4, x, mg, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4(const sz_t m, __local uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, m);\n" \
"	bck4(xl, r1i, r23i);\n" \
"	storel2(4, X, m, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const sz_t ml, __local uint64_2 * restrict const X,\n" \
"	const sz_t mg, __global const uint64_2 * restrict const x, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	uint64_2 xl[4]; loadg2(4, xl, x, mg);\n" \
"	bck4(xl, r1i, r23i);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const sz_t mg, __global uint64_2 * restrict const x,\n" \
"	const sz_t ml, __local const uint64_2 * restrict const X, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	bck4(xl, r1i, r23i);\n" \
"	storeg2(4, x, mg, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i_0(const sz_t ml, __local uint64_2 * restrict const X,\n" \
"	const sz_t mg, __global const uint64_2 * restrict const x)\n" \
"{\n" \
"	uint64_2 xl[4]; loadg2(4, xl, x, mg);\n" \
"	fwd4_0(xl);\n" \
"	storel2(4, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o_0(const sz_t mg, __global uint64_2 * restrict const x,\n" \
"	const sz_t ml, __local const uint64_2 * restrict const X)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, ml);\n" \
"	bck4_0(xl);\n" \
"	storeg2(4, x, mg, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_5i_0(const sz_t ml, __local uint64_2 * restrict const X,\n" \
"	const sz_t mg, __global const uint64_2 * restrict const x)\n" \
"{\n" \
"	uint64_2 xl[5]; loadg2(5, xl, x, mg);\n" \
"	fwd5_0(xl);\n" \
"	storel2(5, X, ml, xl);\n" \
"}\n" \
"\n" \
"INLINE void backward_5o_0(const sz_t mg, __global uint64_2 * restrict const x,\n" \
"	const sz_t ml, __local const uint64_2 * restrict const X)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[5]; loadl2(5, xl, X, ml);\n" \
"	bck5_0(xl);\n" \
"	storeg2(5, x, mg, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_mul_4o(__global uint64_2 * restrict const x, __local const uint64_2 * restrict const X, const uint64_2 r)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	fwd22(&xl[0], r.s0); fwd22(&xl[2], r.s1);\n" \
"	storeg2(4, x, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void forward_mul_8o(__global uint64_2 * restrict const x, __local const uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	fwd4(xl, r1, r23);\n" \
"	storeg2(4, x, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void square_4(__local uint64_2 * restrict const X, const uint64_2 r, const uint64_2 ri)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	sqr_4(xl, r, ri);\n" \
"	storel2(4, X, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void square_8(__local uint64_2 * restrict const X, const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	sqr_8(xl, r1, r23, r1i, r23i);\n" \
"	storel2(4, X, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void mult_4(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y, const uint64_2 r, const uint64_2 ri)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	uint64_2 yl[4]; loadg2(4, yl, y, 1);\n" \
"	mul_4(xl, yl, r, ri);\n" \
"	storel2(4, X, 1, xl);\n" \
"}\n" \
"\n" \
"INLINE void mult_8(__local uint64_2 * restrict const X, __global const uint64_2 * restrict const y,\n" \
"	const uint64 r1, const uint64_2 r23, const uint64 r1i, const uint64_2 r23i)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint64_2 xl[4]; loadl2(4, xl, X, 1);\n" \
"	uint64_2 yl[4]; loadg2(4, yl, y, 1);\n" \
"	mul_8(xl, yl, r1, r23, r1i, r23i);\n" \
"	storel2(4, X, 1, xl);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	__global uint64_2 * restrict const x = (__global uint64_2 *)(&reg[offset]); \\\n" \
"	__global const uint64 * restrict const r2 = &root[0]; \\\n" \
"	__global const uint64 * restrict const r2i = &root[N_SZ / 2]; \\\n" \
"	__global const uint64_2 * restrict const r2_2 = (__global const uint64_2 *)(&root[0]); \\\n" \
"	__global const uint64_2 * restrict const r2i_2 = (__global const uint64_2 *)(&root[N_SZ / 2]); \\\n" \
"	__global const uint64_2 * restrict const r4 = (__global const uint64_2 *)(&root[N_SZ]); \\\n" \
"	__global const uint64_2 * restrict const r4i = (__global const uint64_2 *)(&root[N_SZ + N_SZ]); \\\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"\n" \
"/////////////////////////////////////\n" \
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
"	const sz_t sj = s + idx_m;\n" \
"\n" \
"#if (MAX_WG_SZ >= 16 / 4 * CHUNK16)\n" \
"\n" \
"#define ATTR_FB_16()	__attribute__((reqd_work_group_size(16 / 4 * CHUNK16, 1, 1)))\n" \
"\n" \
"/*__kernel\n" \
"ATTR_FB_16()\n" \
"void forward16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(16 / 4, CHUNK16);\n" \
"\n" \
"	forward_4i(4 * CHUNK16, &X[i], 4u << lm, &x[ki], r2[sj / 4], r4[sj / 4]);\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_16()\n" \
"void backward16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(16 / 4, CHUNK16);\n" \
"\n" \
"	backward_4i(1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]);\n" \
"	backward_4o(4u << lm, &x[ki], 4 * CHUNK16, &X[i], r2i[sj / 4], r4i[sj / 4]);\n" \
"}*/\n" \
"\n" \
"#if (N_SZ % 5 != 0) && (N_SZ >= 64) && (N_SZ <= 2048)\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_16()\n" \
"void forward16_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 16 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;\n" \
"	DECLARE_VAR(16 / 4, CHUNK16);\n" \
"\n" \
"	forward_4i_0(4 * CHUNK16, &X[i], 4u << lm, &x[ki]);\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_16()\n" \
"void backward16_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 16 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;\n" \
"	DECLARE_VAR(16 / 4, CHUNK16);\n" \
"\n" \
"	backward_4i(1 * CHUNK16, &Xi[CHUNK16 * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]);\n" \
"	backward_4o_0(4u << lm, &x[ki], 4 * CHUNK16, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"\n" \
"#if (MAX_WG_SZ >= 20 / 4 * CHUNK20)\n" \
"\n" \
"#if (N_SZ % 5 == 0) && (N_SZ >= 80) && (N_SZ <= 2560)\n" \
"\n" \
"#define ATTR_FB_20()	__attribute__((reqd_work_group_size(20 / 4 * CHUNK20, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_20()\n" \
"void forward20_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 20 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;\n" \
"	DECLARE_VAR(20 / 4, CHUNK20);\n" \
"\n" \
"	if (i < 4 * (20 / 4 * CHUNK20) / 5) forward_5i_0(4 * CHUNK20, &X[i], 4u << lm, &x[ki]);\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK20, &Xi[CHUNK20 * 4 * thread_idx], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_20()\n" \
"void backward20_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 20 / 4; const uint32 lm = LN_SZ_S5 - 1 - 2;\n" \
"	DECLARE_VAR(20 / 4, CHUNK20);\n" \
"\n" \
"	backward_4i(1 * CHUNK20, &Xi[CHUNK20 * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]);\n" \
"	if (i < 4 * (20 / 4 * CHUNK20) / 5) backward_5o_0(4u << lm, &x[ki], 4 * CHUNK20, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"\n" \
"#define FORWARD_64_80(CHUNK_N) \\\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \\\n" \
"	forward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2[sj / 4], r4[sj / 4]); \\\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], r2[sj], r4[sj]);\n" \
"\n" \
"#define BACKWARD_64_80(CHUNK_N) \\\n" \
"	backward_4i(1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \\\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \\\n" \
"	backward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2i[sj / 4], r4i[sj / 4]);\n" \
"\n" \
"#if (MAX_WG_SZ >= 64 / 4 * CHUNK64)\n" \
"\n" \
"#define ATTR_FB_64()	__attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))\n" \
"\n" \
"#if (N_SZ >= 655360)\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64()\n" \
"void forward64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(64 / 4, CHUNK64);\n" \
"\n" \
"	forward_4i(16 * CHUNK64, &X[i], 16u << lm, &x[ki], r2[sj / 16], r4[sj / 16]);\n" \
"	FORWARD_64_80(CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64()\n" \
"void backward64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(64 / 4, CHUNK64);\n" \
"\n" \
"	BACKWARD_64_80(CHUNK64);\n" \
"	backward_4o(16u << lm, &x[ki], 16 * CHUNK64, &X[i], r2i[sj / 16], r4i[sj / 16]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ % 5 != 0) && (N_SZ >= 4096)\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64()\n" \
"void forward64_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 64 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;\n" \
"	DECLARE_VAR(64 / 4, CHUNK64);\n" \
"\n" \
"	forward_4i_0(16 * CHUNK64, &X[i], 16u << lm, &x[ki]);\n" \
"	FORWARD_64_80(CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_64()\n" \
"void backward64_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 64 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;\n" \
"	DECLARE_VAR(64 / 4, CHUNK64);\n" \
"\n" \
"	BACKWARD_64_80(CHUNK64);\n" \
"	backward_4o_0(16u << lm, &x[ki], 16 * CHUNK64, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 80 / 4 * CHUNK80)\n" \
"\n" \
"#if (N_SZ % 5 == 0) && (N_SZ >= 5120)\n" \
"\n" \
"#define ATTR_FB_80()	__attribute__((reqd_work_group_size(80 / 4 * CHUNK80, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_80()\n" \
"void forward80_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 80 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;\n" \
"	DECLARE_VAR(80 / 4, CHUNK80);\n" \
"\n" \
"	if (i < 4 * (80 / 4 * CHUNK80) / 5) forward_5i_0(16 * CHUNK80, &X[i], 16u << lm, &x[ki]);\n" \
"	FORWARD_64_80(CHUNK80);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_80()\n" \
"void backward80_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 80 / 4; const uint32 lm = LN_SZ_S5 - 1 - 4;\n" \
"	DECLARE_VAR(80 / 4, CHUNK80);\n" \
"\n" \
"	BACKWARD_64_80(CHUNK80);\n" \
"	if (i < 4 * (80 / 4 * CHUNK80) / 5) backward_5o_0(16u << lm, &x[ki], 16 * CHUNK80, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"\n" \
"#define FORWARD_256_320(CHUNK_N) \\\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \\\n" \
"	forward_4(16 * CHUNK_N, &Xi[CHUNK_N * i16], r2[sj / 16], r4[sj / 16]); \\\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \\\n" \
"	forward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2[sj / 4], r4[sj / 4]); \\\n" \
"	forward_4o(1u << lm, &x[ko], 1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], r2[sj], r4[sj]);\n" \
"\n" \
"#define BACKWARD_256_320(CHUNK_N) \\\n" \
"	backward_4i(1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \\\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \\\n" \
"	backward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2i[sj / 4], r4i[sj / 4]); \\\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \\\n" \
"	backward_4(16 * CHUNK_N, &Xi[CHUNK_N * i16], r2i[sj / 16], r4i[sj / 16]);\n" \
"\n" \
"#if (MAX_WG_SZ >= 256 / 4 * CHUNK256)\n" \
"\n" \
"#define ATTR_FB_256()	__attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"\n" \
"#if (N_SZ >= 2621440)\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256()\n" \
"void forward256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(256 / 4, CHUNK256);\n" \
"\n" \
"	forward_4i(64 * CHUNK256, &X[i], 64u << lm, &x[ki], r2[sj / 64], r4[sj / 64]);\n" \
"	FORWARD_256_320(CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256()\n" \
"void backward256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(256 / 4, CHUNK256);\n" \
"\n" \
"	BACKWARD_256_320(CHUNK256);\n" \
"	backward_4o(64u << lm, &x[ki], 64 * CHUNK256, &X[i], r2i[sj / 64], r4i[sj / 64]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (N_SZ % 5 != 0) && (N_SZ >= 131072)\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256()\n" \
"void forward256_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 256 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;\n" \
"	DECLARE_VAR(256 / 4, CHUNK256);\n" \
"\n" \
"	forward_4i_0(64 * CHUNK256, &X[i], 64u << lm, &x[ki]);\n" \
"	FORWARD_256_320(CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_256()\n" \
"void backward256_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 256 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;\n" \
"	DECLARE_VAR(256 / 4, CHUNK256);\n" \
"\n" \
"	BACKWARD_256_320(CHUNK256);\n" \
"	backward_4o_0(64u << lm, &x[ki], 64 * CHUNK256, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 320 / 4 * CHUNK320)\n" \
"\n" \
"#if (N_SZ % 5 == 0) && (N_SZ >= 81920)\n" \
"\n" \
"#define ATTR_FB_320()	__attribute__((reqd_work_group_size(320 / 4 * CHUNK320, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_320()\n" \
"void forward320_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 320 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;\n" \
"	DECLARE_VAR(320 / 4, CHUNK320);\n" \
"\n" \
"	if (i < 4 * (320 / 4 * CHUNK320) / 5) forward_5i_0(64 * CHUNK320, &X[i], 64u << lm, &x[ki]);\n" \
"	FORWARD_256_320(CHUNK320);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_320()\n" \
"void backward320_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 320 / 4; const uint32 lm = LN_SZ_S5 - 1 - 6;\n" \
"	DECLARE_VAR(320 / 4, CHUNK320);\n" \
"\n" \
"	BACKWARD_256_320(CHUNK320);\n" \
"	if (i < 4 * (320 / 4 * CHUNK320) / 5) backward_5o_0(64u << lm, &x[ki], 64 * CHUNK320, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"\n" \
"#define FORWARD_1024_1280() \\\n" \
"	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64); \\\n" \
"	forward_4(64, &Xi[i64], r2[sj / 64], r4[sj / 64]); \\\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \\\n" \
"	forward_4(16, &Xi[i16], r2[sj / 16], r4[sj / 16]); \\\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \\\n" \
"	forward_4(4, &Xi[i4], r2[sj / 4], r4[sj / 4]); \\\n" \
"	forward_4o(1u << lm, &x[ko], 1, &Xi[4 * thread_idx], r2[sj], r4[sj]);\n" \
"\n" \
"#define BACKWARD_1024_1280() \\\n" \
"	backward_4i(1, &Xi[4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \\\n" \
"	const sz_t i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \\\n" \
"	backward_4(4, &Xi[i4], r2i[sj / 4], r4i[sj / 4]); \\\n" \
"	const sz_t i16 = 4 * (thread_idx & ~(16 - 1)) + (thread_idx % 16); \\\n" \
"	backward_4(16, &Xi[i16], r2i[sj / 16], r4i[sj / 16]); \\\n" \
"	const sz_t i64 = 4 * (thread_idx & ~(64 - 1)) + (thread_idx % 64); \\\n" \
"	backward_4(64, &Xi[i64], r2i[sj / 64], r4i[sj / 64]);\n" \
"\n" \
"#if (MAX_WG_SZ >= 1024 / 4)\n" \
"\n" \
"#define ATTR_FB_1024()	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"\n" \
"/*__kernel\n" \
"ATTR_FB_1024()\n" \
"void forward1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(1024 / 4, 1);\n" \
"\n" \
"	forward_4i(256, &X[i], 256u << lm, &x[ki], r2[sj / 256], r4[sj / 256]);\n" \
"	FORWARD_1024_1280();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_1024()\n" \
"void backward1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset,\n" \
"	const sz_t s, const uint32 lm)\n" \
"{\n" \
"	DECLARE_VAR(1024 / 4, 1);\n" \
"\n" \
"	BACKWARD_1024_1280();\n" \
"	backward_4o(256u << lm, &x[ki], 256, &X[i], r2i[sj / 256], r4i[sj / 256]);\n" \
"}*/\n" \
"\n" \
"#if (N_SZ % 5 != 0) && (N_SZ >= 524288) && (N_SZ <= 1048576)\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_1024()\n" \
"void forward1024_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 1024 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;\n" \
"	DECLARE_VAR(1024 / 4, 1);\n" \
"\n" \
"	forward_4i_0(256, &X[i], 256u << lm, &x[ki]);\n" \
"	FORWARD_1024_1280();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_1024()\n" \
"void backward1024_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 1024 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;\n" \
"	DECLARE_VAR(1024 / 4, 1);\n" \
"\n" \
"	BACKWARD_1024_1280();\n" \
"	backward_4o_0(256u << lm, &x[ki], 256, &X[i]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 1280 / 4)\n" \
"\n" \
"#if (N_SZ % 5 == 0)\n" \
"\n" \
"#define ATTR_FB_1280()	__attribute__((reqd_work_group_size(1280 / 4, 1, 1)))\n" \
"\n" \
"/*__kernel\n" \
"ATTR_FB_1280()\n" \
"void forward1280_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 1280 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;\n" \
"	DECLARE_VAR(1280 / 4, 1);\n" \
"\n" \
"	if (i < 4 * (1280 / 4) / 5) forward_5i_0(256, &X[i], 256u << lm, &x[ki]);\n" \
"	FORWARD_1024_1280();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_FB_1280()\n" \
"void backward1280_0(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	const sz_t s = 1280 / 4; const uint32 lm = LN_SZ_S5 - 1 - 8;\n" \
"	DECLARE_VAR(1280 / 4, 1);\n" \
"\n" \
"	BACKWARD_1024_1280();\n" \
"	if (i < 4 * (1280 / 4) / 5) backward_5o_0(256u << lm, &x[ki], 256, &X[i]);\n" \
"}*/\n" \
"\n" \
"#endif\n" \
"#endif\n" \
"\n" \
"/////////////////////////////////////\n" \
"\n" \
"#if (MAX_WG_SZ >= 16 / 4 * BLK16) && (N_SZ >= 256) && (N_SZ <= 320)\n" \
"\n" \
"#define DECLARE_VAR_16() \\\n" \
"	__local uint64_2 X[16 * BLK16]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (16 * BLK16); \\\n" \
"	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % (16 * BLK16);\n" \
"\n" \
"#define ATTR_16()	__attribute__((reqd_work_group_size(16 / 4 * BLK16, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_16()\n" \
"void forward_mul16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_16();\n" \
"\n" \
"	forward_4i(2, &X[i2], 2, &x[k2], r2[sj2], r4[sj2]);\n" \
"	forward_mul_4o(&x[k], &X[i], r2_2[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_16()\n" \
"void sqr16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_16();\n" \
"\n" \
"	forward_4i(2, &X[i2], 2, &x[k2], r2[sj2], r4[sj2]);\n" \
"	square_4(&X[i], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4o(2, &x[k2], 2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_16()\n" \
"void mul16(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_16();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(2, &X[i2], 2, &x[k2], r2[sj2], r4[sj2]);\n" \
"	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4o(2, &x[k2], 2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 32 / 4 * BLK32) && (N_SZ >= 512) && (N_SZ <= 640)\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local uint64_2 X[32 * BLK32]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (32 * BLK32); \\\n" \
"	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % (32 * BLK32);\n" \
"\n" \
"#define ATTR_32()	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_32()\n" \
"void forward_mul32(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(4, &X[i4], 4, &x[k4], r2[sj4], r4[sj4]);\n" \
"	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_32()\n" \
"void sqr32(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(4, &X[i4], 4, &x[k4], r2[sj4], r4[sj4]);\n" \
"	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4o(4, &x[k4], 4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_32()\n" \
"void mul32(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(4, &X[i4], 4, &x[k4], r2[sj4], r4[sj4]);\n" \
"	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4o(4, &x[k4], 4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 64 / 4 * BLK64) && (N_SZ >= 1024) && (N_SZ <= 5120)\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local uint64_2 X[64 * BLK64]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (64 * BLK64); \\\n" \
"	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % (64 * BLK64); \\\n" \
"	const sz_t sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % (64 * BLK64);\n" \
"\n" \
"#define ATTR_64()	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void forward_mul64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	forward_4i(8, &X[i8], 8, &x[k8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	forward_mul_4o(&x[k], &X[i], r2_2[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void sqr64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(8, &X[i8], 8, &x[k8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	square_4(&X[i], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"	backward_4o(8, &x[k8], 8, &X[i8], r2i[sj8], r4i[sj8]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void mul64(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(8, &X[i8], 8, &x[k8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"	backward_4o(8, &x[k8], 8, &X[i8], r2i[sj8], r4i[sj8]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 128 / 4 * BLK128) && (N_SZ >= 2048)\n" \
"\n" \
"#define DECLARE_VAR_128() \\\n" \
"	__local uint64_2 X[128 * BLK128]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (128 * BLK128); \\\n" \
"	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % (128 * BLK128); \\\n" \
"	const sz_t sj16 = sj / 16, k16 = 4 * (id & ~(16 - 1)) + (id % 16), i16 = k16 % (128 * BLK128);\n" \
"\n" \
"#define ATTR_128()	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_128()\n" \
"void forward_mul128(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	forward_4i(16, &X[i16], 16, &x[k16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_128()\n" \
"void sqr128(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(16, &X[i16], 16, &x[k16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"	backward_4o(16, &x[k16], 16, &X[i16], r2i[sj16], r4i[sj16]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_128()\n" \
"void mul128(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(16, &X[i16], 16, &x[k16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"	backward_4o(16, &x[k16], 16, &X[i16], r2i[sj16], r4i[sj16]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 256 / 4 * BLK256) && (N_SZ >= 16384)\n" \
"\n" \
"#define DECLARE_VAR_256() \\\n" \
"	__local uint64_2 X[256 * BLK256]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (256 * BLK256); \\\n" \
"	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % (256 * BLK256); \\\n" \
"	const sz_t sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % (256 * BLK256); \\\n" \
"	const sz_t sj32 = sj / 32, k32 = 4 * (id & ~(32 - 1)) + (id % 32), i32 = k32 % (256 * BLK256);\n" \
"\n" \
"#define ATTR_256()	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void forward_mul256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	forward_4i(32, &X[i32], 32, &x[k32], r2[sj32], r4[sj32]);\n" \
"	forward_4(8, &X[i8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	forward_mul_4o(&x[k], &X[i], r2_2[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void sqr256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(32, &X[i32], 32, &x[k32], r2[sj32], r4[sj32]);\n" \
"	forward_4(8, &X[i8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	square_4(&X[i], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);\n" \
"	backward_4o(32, &x[k32], 32, &X[i32], r2i[sj32], r4i[sj32]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void mul256(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(32, &X[i32], 32, &x[k32], r2[sj32], r4[sj32]);\n" \
"	forward_4(8, &X[i8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);\n" \
"	backward_4o(32, &x[k32], 32, &X[i32], r2i[sj32], r4i[sj32]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 512 / 4 * BLK512) && (N_SZ >= 32768)\n" \
"\n" \
"#define DECLARE_VAR_512() \\\n" \
"	__local uint64_2 X[512 * BLK512]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % (512 * BLK512); \\\n" \
"	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % (512 * BLK512); \\\n" \
"	const sz_t sj16 = sj / 16, k16 = 4 * (id & ~(16 - 1)) + (id % 16), i16 = k16 % (512 * BLK512); \\\n" \
"	const sz_t sj64 = sj / 64, k64 = 4 * (id & ~(64 - 1)) + (id % 64), i64 = k64 % (512 * BLK512);\n" \
"\n" \
"#define ATTR_512()	__attribute__((reqd_work_group_size(512 / 4 * BLK512, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_512()\n" \
"void forward_mul512(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	forward_4i(64, &X[i64], 64, &x[k64], r2[sj64], r4[sj64]);\n" \
"	forward_4(16, &X[i16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_512()\n" \
"void sqr512(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(64, &X[i64], 64, &x[k64], r2[sj64], r4[sj64]);\n" \
"	forward_4(16, &X[i16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);\n" \
"	backward_4o(64, &x[k64], 64, &X[i64], r2i[sj64], r4i[sj64]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_512()\n" \
"void mul512(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(64, &X[i64], 64, &x[k64], r2[sj64], r4[sj64]);\n" \
"	forward_4(16, &X[i16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);\n" \
"	backward_4o(64, &x[k64], 64, &X[i64], r2i[sj64], r4i[sj64]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"#if (MAX_WG_SZ >= 1024 / 4) && (N_SZ >= 65536)\n" \
"\n" \
"#define DECLARE_VAR_1024() \\\n" \
"	__local uint64_2 X[1024]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % 1024; \\\n" \
"	const sz_t sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % 1024; \\\n" \
"	const sz_t sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % 1024; \\\n" \
"	const sz_t sj32 = sj / 32, k32 = 4 * (id & ~(32 - 1)) + (id % 32), i32 = k32 % 1024; \\\n" \
"	const sz_t sj128 = sj / 128, k128 = 4 * (id & ~(128 - 1)) + (id % 128), i128 = k128 % 1024;\n" \
"\n" \
"#define ATTR_1024()	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void forward_mul1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);\n" \
"	forward_4(32, &X[i32], r2[sj32], r4[sj32]);\n" \
"	forward_4(8, &X[i8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	forward_mul_4o(&x[k], &X[i], r2_2[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void sqr1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);\n" \
"	forward_4(32, &X[i32], r2[sj32], r4[sj32]);\n" \
"	forward_4(8, &X[i8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	square_4(&X[i], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);\n" \
"	backward_4(32, &X[i32], r2i[sj32], r4i[sj32]);\n" \
"	backward_4o(128, &x[k128], 128, &X[i128], r2i[sj128], r4i[sj128]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void mul1024(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);\n" \
"	forward_4(32, &X[i32], r2[sj32], r4[sj32]);\n" \
"	forward_4(8, &X[i8], r2[sj8], r4[sj8]);\n" \
"	forward_4(2, &X[i2], r2[sj2], r4[sj2]);\n" \
"	mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);\n" \
"	backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);\n" \
"	backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);\n" \
"	backward_4(32, &X[i32], r2i[sj32], r4i[sj32]);\n" \
"	backward_4o(128, &x[k128], 128, &X[i128], r2i[sj128], r4i[sj128]);\n" \
"}\n" \
"\n" \
"#endif\n" \
"/* #if (MAX_WG_SZ >= 2048 / 4)\n" \
"\n" \
"#define DECLARE_VAR_2048() \\\n" \
"	__local uint64_2 X[2048]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t j = id, sj = N_SZ / 8 + j, k = 4 * id, i = k % 2048; \\\n" \
"	const sz_t sj4 = sj / 4, k4 = 4 * (id & ~(4 - 1)) + (id % 4), i4 = k4 % 2048; \\\n" \
"	const sz_t sj16 = sj / 16, k16 = 4 * (id & ~(16 - 1)) + (id % 16), i16 = k16 % 2048; \\\n" \
"	const sz_t sj64 = sj / 64, k64 = 4 * (id & ~(64 - 1)) + (id % 64), i64 = k64 % 2048; \\\n" \
"	const sz_t sj256 = sj / 256, k256 = 4 * (id & ~(256 - 1)) + (id % 256), i256 = k256 % 2048;\n" \
"\n" \
"#define ATTR_2048()	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"\n" \
"__kernel\n" \
"ATTR_2048()\n" \
"void forward_mul2048(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	forward_4i(256, &X[i256], 256, &x[k256], r2[sj256], r4[sj256]);\n" \
"	forward_4(64, &X[i64], r2[sj64], r4[sj64]);\n" \
"	forward_4(16, &X[i16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	forward_mul_8o(&x[k], &X[i], r2[sj], r4[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_2048()\n" \
"void sqr2048(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(256, &X[i256], 256, &x[k256], r2[sj256], r4[sj256]);\n" \
"	forward_4(64, &X[i64], r2[sj64], r4[sj64]);\n" \
"	forward_4(16, &X[i16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	square_8(&X[i], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);\n" \
"	backward_4(64, &X[i64], r2i[sj64], r4i[sj64]);\n" \
"	backward_4o(256, &x[k256], 256, &X[i256], r2i[sj256], r4i[sj256]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_2048()\n" \
"void mul2048(__global uint64 * restrict const reg, __global const uint64 * restrict const root, const sz_t offset, const sz_t offset_y)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	__global uint64_2 * restrict const y = (__global uint64_2 *)(&reg[offset_y]);\n" \
"\n" \
"	forward_4i(256, &X[i256], 256, &x[k256], r2[sj256], r4[sj256]);\n" \
"	forward_4(64, &X[i64], r2[sj64], r4[sj64]);\n" \
"	forward_4(16, &X[i16], r2[sj16], r4[sj16]);\n" \
"	forward_4(4, &X[i4], r2[sj4], r4[sj4]);\n" \
"	mult_8(&X[i], &y[k], r2[sj], r4[sj], r2i[sj], r4i[sj]);\n" \
"	backward_4(4, &X[i4], r2i[sj4], r4i[sj4]);\n" \
"	backward_4(16, &X[i16], r2i[sj16], r4i[sj16]);\n" \
"	backward_4(64, &X[i64], r2i[sj64], r4i[sj64]);\n" \
"	backward_4o(256, &x[k256], 256, &X[i256], r2i[sj256], r4i[sj256]);\n" \
"}\n" \
"\n" \
"#endif */\n" \
"\n" \
"// --- carry ---\n" \
"\n" \
"// Unweight, mul by a, carry (pass 1)\n" \
"__kernel\n" \
"__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))\n" \
"void carry_weight_mul_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const uint32 a, const sz_t offset)\n" \
"{\n" \
"	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"	__local uint64 cl[CWM_WG_SZ];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ;\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);\n" \
"\n" \
"	const uint64_4 w = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"\n" \
"	const uint_8_4 wd = width4[gid];\n" \
"\n" \
"	uint64 c = 0;\n" \
"	uint64_4 u = mod_mul4(mod_mul4(x[gid], INV_N_2), wi);\n" \
"	u = adc_mul4(u, a, wd, &c);\n" \
"\n" \
"	cl[lid] = c;\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);\n" \
"	x[gid] = mod_mul4(u, w);\n" \
"\n" \
"	if (lid == CWM_WG_SZ - 1)\n" \
"	{\n" \
"		carry[(gid != N_SZ / 4 - 1) ? gid / CWM_WG_SZ + 1 : 0] = c;\n" \
"	}\n" \
"}\n" \
"\n" \
"\n" \
"__kernel\n" \
"__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))\n" \
"void carry_weight_mul_p1_copy(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const uint32 a, const sz_t off_src, const sz_t off_dst)\n" \
"{\n" \
"	__global uint64_4 * restrict const xs = (__global uint64_4 *)(&reg[off_src]);\n" \
"	__global uint64_4 * restrict const xc = (__global uint64_4 *)(&reg[off_dst]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"	__local uint64 cl[CWM_WG_SZ];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	const sz_t lid = (sz_t)get_local_id(0);\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);\n" \
"	const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"	const uint_8_4 wd = width4[gid];\n" \
"\n" \
"	uint64 c = 0;\n" \
"	uint64_4 u = mod_mul4(mod_mul4(xs[gid], INV_N_2), wi);\n" \
"	u = adc_mul4(u, a, wd, &c);\n" \
"\n" \
"	cl[lid] = c;\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);\n" \
"\n" \
"	const uint64_4 y = mod_mul4(u, w);\n" \
"	xs[gid] = y;\n" \
"	xc[gid] = y;\n" \
"\n" \
"	if (lid == CWM_WG_SZ - 1)\n" \
"	{\n" \
"		const sz_t j = (gid != N_SZ / 4 - 1) ? gid / CWM_WG_SZ + 1 : 0;\n" \
"		carry[j] = c;\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_weight_p2_copy(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const sz_t off_src, const sz_t off_dst)\n" \
"{\n" \
"	__global uint64_4 * restrict const xs = (__global uint64_4 *)(&reg[off_src]);\n" \
"	__global uint64_4 * restrict const xc = (__global uint64_4 *)(&reg[off_dst]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	const sz_t id = (sz_t)(CWM_WG_SZ * gid);\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[id], N_SZ / 4);\n" \
"	const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"	const uint_8_4 wd = width4[id];\n" \
"\n" \
"	const uint64 c = carry[gid];\n" \
"\n" \
"	uint64_4 us = mod_mul4(xs[id], wi);\n" \
"	us = adc4(us, wd, c);\n" \
"	xs[id] = mod_mul4(us, w);\n" \
"\n" \
"	uint64_4 uc = mod_mul4(xc[id], wi);\n" \
"	uc = adc4(uc, wd, c);\n" \
"	xc[id] = mod_mul4(uc, w);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel\n" \
"__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))\n" \
"void carry_weight_add_neg_p1(__global uint64 * restrict reg, __global uint64 * restrict carry,\n" \
"    __global const uint64 * restrict weight, __global const uint_8 * restrict width,\n" \
"    const sz_t offset_y, const sz_t offset_x)\n" \
"{\n" \
"    __global uint64_4 * restrict y = (__global uint64_4 *)(&reg[offset_y]);\n" \
"    __global const uint64_4 * restrict x = (__global const uint64_4 *)(&reg[offset_x]);\n" \
"    __global const uint64_2 * restrict weight2 = (__global const uint64_2 *)(weight);\n" \
"    __global const uint_8_4 * restrict width4  = (__global const uint_8_4 *)(width);\n" \
"    __local  uint64 cl[CWM_WG_SZ];\n" \
"\n" \
"    const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ;\n" \
"\n" \
"    uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);\n" \
"    const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"    const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"    const uint_8_4  wd = width4[gid];\n" \
"\n" \
"    uint64 c = 0;\n" \
"    uint64_4 u  = mod_mul4(y[gid], wi);\n" \
"    uint64_4 vx = mod_mul4(x[gid], wi);\n" \
"    uint64_4 vn = neg_mp4(vx, wd);               // vn = Mp - X\n" \
"\n" \
"    u = addc4(u, vn, wd, &c);                    \n" \
"    cl[lid] = c;\n" \
"    barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"    u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);\n" \
"    y[gid] = mod_mul4(u, w);\n" \
"    if (lid == CWM_WG_SZ - 1) {\n" \
"        carry[(gid != N_SZ/4 - 1) ? gid / CWM_WG_SZ + 1 : 0] = c;\n" \
"    }\n" \
"}\n" \
"\n" \
"__kernel\n" \
"__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))\n" \
"void carry_weight_addsub_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const sz_t off_sum, const sz_t off_diff, const sz_t off_a, const sz_t off_b)\n" \
"{\n" \
"	__global uint64_4 * restrict const yS = (__global uint64_4 *)(&reg[off_sum]);\n" \
"	__global uint64_4 * restrict const yD = (__global uint64_4 *)(&reg[off_diff]);\n" \
"	__global const uint64_4 * restrict const a = (__global const uint64_4 *)(&reg[off_a]);\n" \
"	__global const uint64_4 * restrict const b = (__global const uint64_4 *)(&reg[off_b]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	const sz_t lid = (sz_t)get_local_id(0);\n" \
"	const sz_t grp = (sz_t)get_group_id(0);\n" \
"	const sz_t ngr = (sz_t)get_num_groups(0);\n" \
"\n" \
"	__local uint2 lcc[CWM_WG_SZ];\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);\n" \
"	const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"	const uint_8_4 wd = width4[gid];\n" \
"\n" \
"	uint64 cS = 0, cD = 0;\n" \
"	const uint64_4 av = mod_mul4(a[gid], wi);\n" \
"	const uint64_4 bv = mod_mul4(b[gid], wi);\n" \
"	const uint64_4 nb = neg_mp4(bv, wd);\n" \
"\n" \
"	uint64_4 uS = addc4(av, bv, wd, &cS);\n" \
"	uint64_4 uD = addc4(av, nb, wd, &cD);\n" \
"\n" \
"	lcc[lid] = (uint2)((uint)cS, (uint)cD);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 prev = (lid == 0) ? (uint2)(0u,0u) : lcc[lid - 1];\n" \
"	uS = adc4(uS, wd, (uint64)prev.x);\n" \
"	uD = adc4(uD, wd, (uint64)prev.y);\n" \
"\n" \
"	yS[gid] = mod_mul4(uS, w);\n" \
"	yD[gid] = mod_mul4(uD, w);\n" \
"\n" \
"	if (lid == (CWM_WG_SZ - 1))\n" \
"	{\n" \
"		uint j = (uint)(grp + 1u);\n" \
"		if (j == (uint)ngr) j = 0u;\n" \
"		carry[j] = (uint64)((uint64)lcc[lid].x | ((uint64)lcc[lid].y << 32));\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_weight_p2x2(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const sz_t off_sum, const sz_t off_diff)\n" \
"{\n" \
"	__global uint64_4 * restrict const xs = (__global uint64_4 *)(&reg[off_sum]);\n" \
"	__global uint64_4 * restrict const xd = (__global uint64_4 *)(&reg[off_diff]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	const sz_t id = (sz_t)(CWM_WG_SZ * gid);\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[id], N_SZ / 4);\n" \
"	const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"	const uint_8_4 wd = width4[id];\n" \
"\n" \
"	const uint64 packed = carry[gid];\n" \
"	const uint cs = (uint)(packed & 0xffffffffUL);\n" \
"	const uint cd = (uint)(packed >> 32);\n" \
"\n" \
"	uint64_4 us = mod_mul4(xs[id], wi);\n" \
"	us = adc4(us, wd, (uint64)cs);\n" \
"	xs[id] = mod_mul4(us, w);\n" \
"\n" \
"	uint64_4 ud = mod_mul4(xd[id], wi);\n" \
"	ud = adc4(ud, wd, (uint64)cd);\n" \
"	xd[id] = mod_mul4(ud, w);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))\n" \
"void carry_weight_addsub_p1_copy(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const sz_t off_sum, const sz_t off_diff, const sz_t off_sum_copy, const sz_t off_diff_copy,\n" \
"	const sz_t off_a, const sz_t off_b)\n" \
"{\n" \
"	__global uint64_4 * restrict const yS  = (__global uint64_4 *)(&reg[off_sum]);\n" \
"	__global uint64_4 * restrict const yD  = (__global uint64_4 *)(&reg[off_diff]);\n" \
"	__global uint64_4 * restrict const ySc = (__global uint64_4 *)(&reg[off_sum_copy]);\n" \
"	__global uint64_4 * restrict const yDc = (__global uint64_4 *)(&reg[off_diff_copy]);\n" \
"	__global const uint64_4 * restrict const a = (__global const uint64_4 *)(&reg[off_a]);\n" \
"	__global const uint64_4 * restrict const b = (__global const uint64_4 *)(&reg[off_b]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	const sz_t lid = (sz_t)get_local_id(0);\n" \
"\n" \
"	__local uint2 lcc[CWM_WG_SZ];\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);\n" \
"	const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"	const uint_8_4 wd = width4[gid];\n" \
"\n" \
"	const uint64_4 av = mod_mul4(a[gid], wi);\n" \
"	const uint64_4 bv = mod_mul4(b[gid], wi);\n" \
"	const uint64_4 nb = neg_mp4(bv, wd);\n" \
"\n" \
"	uint64 cS = 0, cD = 0;\n" \
"	uint64_4 uS = addc4(av, bv, wd, &cS);\n" \
"	uint64_4 uD = addc4(av, nb, wd, &cD);\n" \
"\n" \
"	lcc[lid] = (uint2)((uint)cS,(uint)cD);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 prev = (lid == 0) ? (uint2)(0u,0u) : lcc[lid - 1];\n" \
"	uS = adc4(uS, wd, (uint64)prev.x);\n" \
"	uD = adc4(uD, wd, (uint64)prev.y);\n" \
"\n" \
"	const uint64_4 outS = mod_mul4(uS, w);\n" \
"	const uint64_4 outD = mod_mul4(uD, w);\n" \
"\n" \
"	yS [gid] = outS;\n" \
"	yD [gid] = outD;\n" \
"	ySc[gid] = outS;\n" \
"	yDc[gid] = outD;\n" \
"\n" \
"	if (lid == (CWM_WG_SZ - 1))\n" \
"	{\n" \
"		const sz_t j = ((gid != N_SZ / 4 - 1) ? (gid / CWM_WG_SZ + 1) : 0);\n" \
"		carry[j] = (uint64)((uint64)lcc[lid].x | ((uint64)lcc[lid].y << 32));\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry_weight_p2x2_copy(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const sz_t off_sum, const sz_t off_diff, const sz_t off_sum_copy, const sz_t off_diff_copy)\n" \
"{\n" \
"	__global uint64_4 * restrict const xs  = (__global uint64_4 *)(&reg[off_sum]);\n" \
"	__global uint64_4 * restrict const xd  = (__global uint64_4 *)(&reg[off_diff]);\n" \
"	__global uint64_4 * restrict const xsc = (__global uint64_4 *)(&reg[off_sum_copy]);\n" \
"	__global uint64_4 * restrict const xdc = (__global uint64_4 *)(&reg[off_diff_copy]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	const sz_t id  = (sz_t)(CWM_WG_SZ * gid);\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[id], N_SZ / 4);\n" \
"	const uint64_4 w  = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"	const uint_8_4 wd = width4[id];\n" \
"\n" \
"	const uint64 packed = carry[gid];\n" \
"	const uint cs = (uint)(packed & 0xffffffffUL);\n" \
"	const uint cd = (uint)(packed >> 32);\n" \
"\n" \
"	uint64_4 us = mod_mul4(xs[id], wi);\n" \
"	us = adc4(us, wd, (uint64)cs);\n" \
"	xs[id] = mod_mul4(us, w);\n" \
"\n" \
"	uint64_4 ud = mod_mul4(xd[id], wi);\n" \
"	ud = adc4(ud, wd, (uint64)cd);\n" \
"	xd[id] = mod_mul4(ud, w);\n" \
"\n" \
"	uint64_4 usc = mod_mul4(xsc[id], wi);\n" \
"	usc = adc4(usc, wd, (uint64)cs);\n" \
"	xsc[id] = mod_mul4(usc, w);\n" \
"\n" \
"	uint64_4 udc = mod_mul4(xdc[id], wi);\n" \
"	udc = adc4(udc, wd, (uint64)cd);\n" \
"	xdc[id] = mod_mul4(udc, w);\n" \
"}\n" \
"\n" \
"\n" \
"\n" \
"// Unweight, add, carry (pass 1)\n" \
"__kernel\n" \
"__attribute__((reqd_work_group_size(CWM_WG_SZ, 1, 1)))\n" \
"void carry_weight_add_p1(__global uint64 * restrict const reg, __global uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width,\n" \
"	const sz_t offset_y, const sz_t offset_x)\n" \
"{\n" \
"	__global uint64_4 * restrict const y = (__global uint64_4 *)(&reg[offset_y]);\n" \
"	__global const uint64_4 * restrict const x = (__global const uint64_4 *)(&reg[offset_x]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"	__local uint64 cl[CWM_WG_SZ];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % CWM_WG_SZ;\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[gid], N_SZ / 4);\n" \
"\n" \
"	const uint64_4 w = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"\n" \
"	const uint_8_4 wd = width4[gid];\n" \
"\n" \
"	uint64 c = 0;\n" \
"	uint64_4 u = mod_mul4(y[gid], wi); const uint64_4 v = mod_mul4(x[gid], wi);\n" \
"	u = addc4(u, v, wd, &c);\n" \
"\n" \
"	cl[lid] = c;\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	u = adc4(u, wd, (lid == 0) ? 0 : cl[lid - 1]);\n" \
"	y[gid] = mod_mul4(u, w);\n" \
"\n" \
"	if (lid == CWM_WG_SZ - 1)\n" \
"	{\n" \
"		carry[(gid != N_SZ / 4 - 1) ? gid / CWM_WG_SZ + 1 : 0] = c;\n" \
"	}\n" \
"}\n" \
"\n" \
"// Carry, weight (pass 2)\n" \
"__kernel\n" \
"void carry_weight_p2(__global uint64 * restrict const reg, __global const uint64 * restrict const carry,\n" \
"	__global const uint64 * restrict const weight, __global const uint_8 * restrict const width, const sz_t offset)\n" \
"{\n" \
"	__global uint64_4 * restrict const x = (__global uint64_4 *)(&reg[offset]);\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"	__global const uint_8_4 * restrict const width4 = (__global const uint_8_4 *)(width);\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), id = CWM_WG_SZ * gid;\n" \
"\n" \
"	uint64_2 w2[4]; loadg2(4, w2, &weight2[id], N_SZ / 4);\n" \
"	const uint64_4 w = (uint64_4)(w2[0].s0, w2[1].s0, w2[2].s0, w2[3].s0);\n" \
"	const uint64_4 wi = (uint64_4)(w2[0].s1, w2[1].s1, w2[2].s1, w2[3].s1);\n" \
"\n" \
"	const uint_8_4 wd = width4[id];\n" \
"\n" \
"	uint64_4 u = mod_mul4(x[id], wi);\n" \
"	u = adc4(u, wd, carry[gid]);\n" \
"	x[id] = mod_mul4(u, w);\n" \
"}\n" \
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
"__kernel\n" \
"void subtract(__global uint64 * restrict const reg, __global const uint64 * restrict const weight,\n" \
"	__global const uint_8 * restrict const width, const sz_t offset, const uint32 a)\n" \
"{\n" \
"	__global uint64 * restrict const x = &reg[offset];\n" \
"	__global const uint64_2 * restrict const weight2 = (__global const uint64_2 *)(weight);\n" \
"\n" \
"	uint32 c = a;\n" \
"	while (c != 0)\n" \
"	{\n" \
"		// Unweight, sub with carry, weight\n" \
"		for (size_t k = 0; k < N_SZ; ++k)\n" \
"		{\n" \
"			const uint64_2 w = weight2[k / 4 + (k % 4) * (N_SZ / 4)];\n" \
"			x[k] = mod_mul(sbc(mod_mul(x[k], w.s1), width[k], &c), w.s0);\n" \
"			if (c == 0) return;\n" \
"		}\n" \
"	}\n" \
"}\n" \
"";
