/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"

#include <cstring>

class engine_cpu : public engine
{
private:
	uint64 * _reg;	// the weighted representation of R0, R1, ...
	uint64 * _root;
	uint64 * _weight;
	uint8 * _digit_width;
	uint64 * _carry;

private:
	// Radix-2
	static void fwd2(uint64 & x0, uint64 & x1, const uint64 r)
	{
		const uint64 u0 = x0, u1 = mod_mul(r, x1);
		x0 = mod_add(u0, u1); x1 = mod_sub(u0, u1);
	}

	// Inverse radix-2
	static void bck2(uint64 & x0, uint64 & x1, const uint64 ri)
	{
		const uint64 u0 = x0, u1 = x1;
		x0 = mod_add(u0, u1); x1 = mod_mul(ri, mod_sub(u0, u1));
	}

	// Radix-4
	static void fwd4(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 r1, const uint64 r20, const uint64 r21)
	{
		const uint64 u0 = x0, u2 = mod_mul(r1, x2), u1 = mod_mul(r20, x1), u3 = mod_mul(r21, x3);
		const uint64 v0 = mod_add(u0, u2), v2 = mod_sub(u0, u2), v1 = mod_add(u1, u3), v3 = mod_muli(mod_sub(u1, u3));
		x0 = mod_add(v0, v1); x1 = mod_sub(v0, v1); x2 = mod_add(v2, v3); x3 = mod_sub(v2, v3);
	}

	// Inverse radix-4
	static void bck4(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 ri1, const uint64 ri20, const uint64 ri21)
	{
		const uint64 u0 = x0, u1 = x1, u2 = x2, u3 = x3;
		const uint64 v0 = mod_add(u0, u1), v1 = mod_sub(u0, u1), v2 = mod_add(u3, u2), v3 = mod_muli(mod_sub(u3, u2));
		x0 = mod_add(v0, v2); x2 = mod_mul(ri1, mod_sub(v0, v2)); x1 = mod_mul(ri20, mod_add(v1, v3)); x3 = mod_mul(ri21, mod_sub(v1, v3));
	}

	// 2 x radix-2
	static void fwd22(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 r)
	{
		const uint64 u0 = x0, u2 = mod_mul(r, x2), u1 = x1, u3 = mod_mul(r, x3);
		x0 = mod_add(u0, u2); x2 = mod_sub(u0, u2); x1 = mod_add(u1, u3); x3 = mod_sub(u1, u3);
	}

	// 2 x inverse radix-2
	static void bck22(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, const uint64 ri)
	{
		const uint64 u0 = x0, u2 = x2, u1 = x1, u3 = x3;
		x0 = mod_add(u0, u2); x2 = mod_mul(ri, mod_sub(u0, u2)); x1 = mod_add(u1, u3); x3 = mod_mul(ri, mod_sub(u1, u3));
	}

	// Winograd, S. On computing the discrete Fourier transform, Math. Comp. 32 (1978), no. 141, 175–199.
	static void butterfly5(uint64 & a0, uint64 & a1, uint64 & a2, uint64 & a3, uint64 & a4)
	{
		static const uint64 K = mod_root_nth(5), K2 = mod_sqr(K), K3 = mod_mul(K, K2), K4 = mod_sqr(K2);
		static const uint64 cosu = mod_half(mod_add(K, K4)), isinu = mod_half(mod_sub(K, K4));
		static const uint64 cos2u = mod_half(mod_add(K2, K3)), isin2u = mod_half(mod_sub(K2, K3));
		static const uint64 F1 = mod_sub(mod_half(mod_add(cosu, cos2u)), 1), F2 = mod_half(mod_sub(cosu, cos2u));
		static const uint64 F3 = mod_add(isinu, isin2u), F4 = isin2u, F5 = mod_sub(isinu, isin2u);

		const uint64 s1 = mod_add(a1, a4), s2 = mod_sub(a1, a4), s3 = mod_add(a3, a2), s4 = mod_sub(a3, a2);
		const uint64 s5 = mod_add(s1, s3), s6 = mod_sub(s1, s3), s7 = mod_add(s2, s4), s8 = mod_add(s5, a0);
		const uint64 m0 = s8;
		const uint64 m1 = mod_mul(F1, s5), m2 = mod_mul(F2, s6), m3 = mod_mul(F3, s2), m4 = mod_mul(F4, s7), m5 = mod_mul(F5, s4);
		const uint64 s9 = mod_add(m0, m1), s10 = mod_add(s9, m2), s11 = mod_sub(s9, m2), s12 = mod_sub(m3, m4);
		const uint64 s13 = mod_add(m4, m5), s14 = mod_add(s10, s12), s15 = mod_sub(s10, s12), s16 = mod_add(s11, s13);
		const uint64 s17 = mod_sub(s11, s13);
		a0 = m0; a1 = s14; a2 = s16; a3 = s17; a4 = s15;
	}

	// Radix-5
	static void fwd5(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, uint64 & x4, const uint64 r)
	{
		const uint64 r2 = mod_mul(r, r), r3 = mod_mul(r, r2), r4 = mod_sqr(r2);
		uint64 a0 = x0, a1 = mod_mul(r, x1), a2 = mod_mul(r2, x2), a3 = mod_mul(r3, x3), a4 = mod_mul(r4, x4);
		butterfly5(a0, a1, a2, a3, a4);
		x0 = a0; x1 = a1; x2 = a2; x3 = a3; x4 = a4;
	}

	// Inverse radix-5
	static void bck5(uint64 & x0, uint64 & x1, uint64 & x2, uint64 & x3, uint64 & x4, const uint64 ri)
	{
		uint64 a0 = x0, a4 = x1, a3 = x2, a2 = x3, a1 = x4;
		butterfly5(a0, a1, a2, a3, a4);
		const uint64 ri2 = mod_mul(ri, ri), ri3 = mod_mul(ri, ri2), ri4 = mod_sqr(ri2);
		x0 = a0; x1 = mod_mul(ri, a1); x2 = mod_mul(ri2, a2); x3 = mod_mul(ri3, a3); x4 = mod_mul(ri4, a4);
	}

	// Transform, n = 4^e
	void forward4(uint64 * const x) const
	{
		const size_t n = get_size(), n_4 = n / 4;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r4 = &_root[n / 2];

		const int lm_max = get_ln_max();
		for (int lm = lm_max; lm >= 2; lm -= 2)
		{
			const size_t m = size_t(1) << lm;
			for (size_t id = 0; id < n_4; ++id)
			{
				const size_t j = id >> lm, k = 3 * (id & ~(m - 1)) + id;
				fwd4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r2[j], r4[2 * j + 0], r4[2 * j + 1]);
			}
		}
	}

	// Inverse transform, n = 4^e
	void backward4(uint64 * const x) const
	{
		const size_t n = get_size(), n_4 = n / 4;
		const uint64 * const r2i = &_root[n];
		const uint64 * const r4i = &_root[n + n / 2];

		const int lm_max = get_ln_max();
		for (int lm = 2; lm <= lm_max; lm += 2)
		{
			const size_t m = size_t(1) << lm;
			for (size_t id = 0; id < n_4; ++id)
			{
				const size_t j = id >> lm, k = 3 * (id & ~(m - 1)) + id;
				bck4(x[k + 0 * m], x[k + 1 * m], x[k + 2 * m], x[k + 3 * m], r2i[j], r4i[2 * j + 0], r4i[2 * j + 1]);
			}
		}
	}

	// Radix-2
	void forward_mul4(uint64 * const x) const
	{
		const size_t n_4 = get_size() / 4;
		const uint64 * const r2 = &_root[0];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 4 * id;
			fwd22(x[k + 0], x[k + 1], x[k + 2], x[k + 3], r2[j]);
		}
	}

	// Radix-2, square2x2, inverse radix-2
	void sqr4(uint64 * const x) const
	{
		const size_t n = get_size(), n_4 = n / 4;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r2i = &_root[n];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 4 * id;
			const uint64 r = r2[j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r);
			const uint64 t0 = mod_add(mod_sqr(x0), mod_mul(mod_sqr(x1), r)); x1 = mod_mul(x1, mod_add(x0, x0)); x0 = t0;
			const uint64 t2 = mod_sub(mod_sqr(x2), mod_mul(mod_sqr(x3), r)); x3 = mod_mul(x3, mod_add(x2, x2)); x2 = t2;
			bck22(x0, x1, x2, x3, r2i[j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
		}
	}

	// Radix-2, mul2x2, inverse radix-2
	void mul4(uint64 * const x, const uint64 * const y) const
	{
		const size_t n = get_size(), n_4 = n / 4;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r2i = &_root[n];

		for (size_t id = 0; id < n_4; ++id)
		{
			const size_t j = id, k = 4 * id;
			const uint64 r = r2[j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
			fwd22(x0, x1, x2, x3, r);
			const uint64 y0 = y[k + 0], y1 = y[k + 1], y2 = y[k + 2], y3 = y[k + 3];
			const uint64 t0 = mod_add(mod_mul(x0, y0), mod_mul(mod_mul(x1, y1), r)); x1 = mod_add(mod_mul(x0, y1), mod_mul(x1, y0)); x0 = t0;
			const uint64 t2 = mod_sub(mod_mul(x2, y2), mod_mul(mod_mul(x3, y3), r)); x3 = mod_add(mod_mul(x2, y3), mod_mul(x3, y2)); x2 = t2;
			bck22(x0, x1, x2, x3, r2i[j]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3;
		}
	}

	// Transform, n = 10 * 4^e
	void forward5(uint64 * const x) const
	{
		const size_t n = get_size(), n_4 = n / 4, n_5 = n / 5;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r4 = &_root[n_5 / 2];

		const int lm_max = get_ln_max();
		for (int lm = lm_max; lm >= 1; lm -= 2)
		{
			const size_t m = size_t(1) << lm, m5 = size_t(5) << lm;
			for (size_t id = 0; id < n_4; ++id)
			{
				const uint32 id5 = uint32((uint64(id) * 858993460u) >> 32);	// id5 = id / 5 if id < 2^30
				const size_t j = id5 >> lm, k = 3 * 5 * (id5 & ~(m - 1)) + id;
				fwd4(x[k + 0 * m5], x[k + 1 * m5], x[k + 2 * m5], x[k + 3 * m5], r2[j], r4[2 * j + 0], r4[2 * j + 1]);
			}
		}
	}

	// Inverse transform, n = 10 * 4^e
	void backward5(uint64 * const x) const
	{
		const size_t n = get_size(), n_4 = n / 4, n_5 = n / 5;
		const uint64 * const r2i = &_root[n];
		const uint64 * const r4i = &_root[n + n_5 / 2];

		const int lm_max = get_ln_max();
		for (int lm = 1; lm <= lm_max; lm += 2)
		{
			const size_t m = size_t(1) << lm, m5 = size_t(5) << lm;
			for (size_t id = 0; id < n_4; ++id)
			{
				const uint32 id5 = uint32((uint64(id) * 858993460u) >> 32);	// id5 = id / 5 if id < 2^30
				const size_t j = id5 >> lm, k = 3 * 5 * (id5 & ~(m - 1)) + id;
				bck4(x[k + 0 * m5], x[k + 1 * m5], x[k + 2 * m5], x[k + 3 * m5], r2i[j], r4i[2 * j + 0], r4i[2 * j + 1]);
			}
		}
	}

	// Radix-2, Radix-5
	void forward_mul10(uint64 * const x) const
	{
		const size_t n = get_size(), n_5 = n / 5, n_10 = n_5 / 2;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r5 = &_root[n_5];

		for (size_t id = 0; id < n_10; ++id)
		{
			const size_t j = id, k = 10 * id;
			const uint64 r = r2[j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3], x4 = x[k + 4];
			uint64 x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7], x8 = x[k + 8], x9 = x[k + 9];
			fwd2(x0, x5, r); fwd2(x1, x6, r); fwd2(x2, x7, r); fwd2(x3, x8, r); fwd2(x4, x9, r);
			fwd5(x0, x1, x2, x3, x4, r5[2 * j + 0]);
			fwd5(x5, x6, x7, x8, x9, r5[2 * j + 1]);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3; x[k + 4] = x4;
			x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7; x[k + 8] = x8; x[k + 9] = x9;
		}
	}

	// Radix-2, Radix-5, square, inverse radix-5, inverse radix-2
	void sqr10(uint64 * const x) const
	{
		const size_t n = get_size(), n_5 = n / 5, n_10 = n_5 / 2;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r2i = &_root[n];
		const uint64 * const r5 = &_root[n_5];
		const uint64 * const r5i = &_root[n + n_5];

		for (size_t id = 0; id < n_10; ++id)
		{
			const size_t j = id, k = 10 * id;
			const uint64 r = r2[j], ri = r2i[j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3], x4 = x[k + 4];
			uint64 x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7], x8 = x[k + 8], x9 = x[k + 9];
			fwd2(x0, x5, r); fwd2(x1, x6, r); fwd2(x2, x7, r); fwd2(x3, x8, r); fwd2(x4, x9, r);
			fwd5(x0, x1, x2, x3, x4, r5[2 * j + 0]);
			fwd5(x5, x6, x7, x8, x9, r5[2 * j + 1]);
			x0 = mod_sqr(x0); x1 = mod_sqr(x1); x2 = mod_sqr(x2); x3 = mod_sqr(x3); x4 = mod_sqr(x4);
			x5 = mod_sqr(x5); x6 = mod_sqr(x6); x7 = mod_sqr(x7); x8 = mod_sqr(x8); x9 = mod_sqr(x9);
			bck5(x0, x1, x2, x3, x4, r5i[2 * j + 0]);
			bck5(x5, x6, x7, x8, x9, r5i[2 * j + 1]);
			bck2(x0, x5, ri); bck2(x1, x6, ri); bck2(x2, x7, ri); bck2(x3, x8, ri); bck2(x4, x9, ri);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3; x[k + 4] = x4;
			x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7; x[k + 8] = x8; x[k + 9] = x9;
		}
	}

	// Radix-2, Radix-5, mul, inverse radix-5, inverse radix-2
	void mul10(uint64 * const x, const uint64 * const y) const
	{
		const size_t n = get_size(), n_5 = n / 5, n_10 = n_5 / 2;
		const uint64 * const r2 = &_root[0];
		const uint64 * const r2i = &_root[n];
		const uint64 * const r5 = &_root[n_5];
		const uint64 * const r5i = &_root[n + n_5];

		for (size_t id = 0; id < n_10; ++id)
		{
			const size_t j = id, k = 10 * id;
			const uint64 r = r2[j], ri = r2i[j];
			uint64 x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3], x4 = x[k + 4];
			uint64 x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7], x8 = x[k + 8], x9 = x[k + 9];
			fwd2(x0, x5, r); fwd2(x1, x6, r); fwd2(x2, x7, r); fwd2(x3, x8, r); fwd2(x4, x9, r);
			fwd5(x0, x1, x2, x3, x4, r5[2 * j + 0]);
			fwd5(x5, x6, x7, x8, x9, r5[2 * j + 1]);
			x0 = mod_mul(x0, y[k + 0]); x1 = mod_mul(x1, y[k + 1]); x2 = mod_mul(x2, y[k + 2]); x3 = mod_mul(x3, y[k + 3]); x4 = mod_mul(x4, y[k + 4]);
			x5 = mod_mul(x5, y[k + 5]); x6 = mod_mul(x6, y[k + 6]); x7 = mod_mul(x7, y[k + 7]); x8 = mod_mul(x8, y[k + 8]); x9 = mod_mul(x9, y[k + 9]);
			bck5(x0, x1, x2, x3, x4, r5i[2 * j + 0]);
			bck5(x5, x6, x7, x8, x9, r5i[2 * j + 1]);
			bck2(x0, x5, ri); bck2(x1, x6, ri); bck2(x2, x7, ri); bck2(x3, x8, ri); bck2(x4, x9, ri);
			x[k + 0] = x0; x[k + 1] = x1; x[k + 2] = x2; x[k + 3] = x3; x[k + 4] = x4;
			x[k + 5] = x5; x[k + 6] = x6; x[k + 7] = x7; x[k + 8] = x8; x[k + 9] = x9;
		}
	}

	// Unweight, carry, mul by a, weight
	void carry_weight_mul(uint64 * const x, const uint32 a = 1) const
	{
		const size_t n = get_size(), n_4 = n / 4;
		const uint64 * const w = &_weight[0];
		const uint64 * const wi_n = &_weight[n];
		const uint8 * const width = _digit_width;
		uint64 * const carry = _carry;

		for (size_t id = 0; id < n_4; ++id)
		{
			uint64 c = 0;
			for (size_t i = 0; i < 4; ++i)
			{
				const size_t k = 4 * id + i;
				const uint64 u = mod_mul(x[k], wi_n[k]);
				x[k] = adc_mul(u, a, width[k], c);
			}
			carry[(id != n_4 - 1) ? id + 1 : 0] = c;
		}

		for (size_t id = 0; id < n_4; ++id)
		{
			uint64 c = carry[id];
			for (size_t i = 0; i < 3; ++i)
			{
				const size_t k = 4 * id + i;
				x[k] = mod_mul(adc(x[k], width[k], c), w[k]);
			}
			const size_t k = 4 * id + 3;
			x[k] = mod_mul(x[k] + c, w[k]);
		}
	}

	// Inverse radix-2, unweight, carry, mul by a, weight, radix-2
	void carry_weight_mul2(uint64 * const x, const uint32 a = 1) const
	{
		const size_t n = get_size(), n_8 = n / 8, n_2 = n / 2;
		const uint64 * const w = &_weight[0];
		const uint64 * const wi_n = &_weight[n];
		const uint8 * const width = _digit_width;
		uint64 * const carry = _carry;

		for (size_t id = 0; id < n_8; ++id)
		{
			uint64 c0 = 0, c1 = 0;
			for (size_t i = 0; i < 4; ++i)
			{
				const size_t k = 4 * id + i;
				const uint64 u0 = x[k + 0 * n_2], u1 = x[k + 1 * n_2];
				const uint64 v0 = mod_mul(mod_add(u0, u1), wi_n[k + 0 * n_2]);
				const uint64 v1 = mod_mul(mod_sub(u0, u1), wi_n[k + 1 * n_2]);
				x[k + 0 * n_2] = adc_mul(v0, a, width[k + 0 * n_2], c0);
				x[k + 1 * n_2] = adc_mul(v1, a, width[k + 1 * n_2], c1);
			}
			carry[id + 1 + 0 * n_8] = c0; carry[(id != n_8 - 1) ? id + 1 + 1 * n_8 : 0] = c1;
		}

		for (size_t id = 0; id < n_8; ++id)
		{
			uint64 c0 = carry[id + 0 * n_8], c1 = carry[id + 1 * n_8];
			for (size_t i = 0; i < 3; ++i)
			{
				const size_t k = 4 * id + i;
				const uint64 u0 = mod_mul(adc(x[k + 0 * n_2], width[k + 0 * n_2], c0), w[k + 0 * n_2]); 
				const uint64 u1 = mod_mul(adc(x[k + 1 * n_2], width[k + 1 * n_2], c1), w[k + 1 * n_2]);
				x[k + 0 * n_2] = mod_add(u0, u1); x[k + 1 * n_2] = mod_sub(u0, u1);
			}
			const size_t k = 4 * id + 3;
			const uint64 u0 = mod_mul(x[k + 0 * n_2] + c0, w[k + 0 * n_2]); 
			const uint64 u1 = mod_mul(x[k + 1 * n_2] + c1, w[k + 1 * n_2]);
			x[k + 0 * n_2] = mod_add(u0, u1); x[k + 1 * n_2] = mod_sub(u0, u1);
		}
	}

public:
	engine_cpu(const uint32_t q) : engine(q)
	{
		const size_t n = get_size();

		_reg = new uint64[3 * n];	// allocate 3 registers
		_root = new uint64[2 * n];
		_weight = new uint64[3 * n];
		_digit_width = new uint8[n];
		_carry = new uint64[n / 4];

		roots(_root);
		weights_widths(q, _weight, _digit_width);
	}

	virtual ~engine_cpu()
	{
		delete[] _reg;
		delete[] _root;
		delete[] _weight;
		delete[] _digit_width;
		delete[] _carry;
	}

	void set(const Reg dst, const uint64 a) const override
	{
		const size_t n = get_size();
		uint64 * const x = &_reg[size_t(dst) * n];

		x[0] = a;	// digit_weight[0] = 1
		for (size_t k = 1; k < n; ++k) x[k] = 0;

		// radix-2
		if (!get_even()) x[n / 2] = x[0];
	}

	void get(uint64 * const d, const Reg src) const override
	{
		const size_t n = get_size();
		const uint64 * const x = &_reg[size_t(src) * n];
		const uint64 * const wi = &_weight[2 * n];
		const uint8 * const width = _digit_width;

		for (size_t k = 0; k < n; ++k) d[k] = x[k];

		if (!get_even())
		{
			// inverse radix-2
			for (size_t k = 0; k < n / 2; ++k)
			{
				const uint64 u0 = d[k + 0 * n / 2], u1 = d[k + 1 * n / 2];
				const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));
				d[k + 0 * n / 2] = v0; d[k + 1 * n / 2] = v1;
			}
		}

		// unweight, carry (strong)
		uint64 c = 0;
		for (size_t k = 0; k < n; ++k) d[k] = adc(mod_mul(d[k], wi[k]), width[k], c);

		while (c != 0)
		{
			for (size_t k = 0; k < n; ++k)
			{
				d[k] = adc(d[k], width[k], c);
				if (c == 0) break;
			}
		}

		// encode
		for (size_t k = 0; k < n; ++k) d[k] = uint32(d[k]) | (uint64(width[k]) << 32);
	}

	void copy(const Reg dst, const Reg src) const override
	{
		const size_t n = get_size();
		const uint64 * const x = &_reg[size_t(src) * n];
		uint64 * const y = &_reg[size_t(dst) * n];

		for (size_t k = 0; k < n; ++k) y[k] = x[k];
	}

	bool is_equal(const Reg src1, const Reg src2) const override
	{
		const size_t n = get_size();
		const uint64 * const x = &_reg[size_t(src1) * n];
		const uint64 * const y = &_reg[size_t(src2) * n];

		for (size_t k = 0; k < n; ++k) if (y[k] != x[k]) return false;
		return true;
	}

	void square_mul(const Reg src, const uint32 a = 1) const override
	{
		const size_t n = get_size();
		uint64 * const x = &_reg[size_t(src) * n];

		if (n % 5 == 0) { forward5(x); sqr10(x); backward5(x); }
		else { forward4(x); sqr4(x); backward4(x); }
		if (get_even()) carry_weight_mul(x, a); else carry_weight_mul2(x, a);
	}

	void set_multiplicand(const Reg dst, const Reg src) const override
	{
		if (src != dst) copy(dst, src);

		const size_t n = get_size();
		uint64 * const y = &_reg[size_t(dst) * n];

		if (n % 5 == 0) { forward5(y); forward_mul10(y); }
		else { forward4(y); forward_mul4(y);}
	}

	void mul(const Reg dst, const Reg src) const override
	{
		const size_t n = get_size();
		uint64 * const x = &_reg[size_t(dst) * n];
		const uint64 * const y = &_reg[size_t(src) * n];

		if (n % 5 == 0) { forward5(x); mul10(x, y); backward5(x); }
		else { forward4(x); mul4(x, y); backward4(x); }
		if (get_even()) carry_weight_mul(x); else carry_weight_mul2(x);
	}

	void sub(const Reg src, const uint32 a) const override
	{
		const size_t n = get_size();
		uint64 * const x = &_reg[size_t(src) * n];
		const uint64 * const w = &_weight[0];
		const uint64 * const wi = &_weight[2 * n];
		const uint8 * const width = _digit_width;

		uint32 c = a;
		while (c != 0)
		{
			if (get_even())
			{
				// Unweight, sub with carry, weight
				for (size_t k = 0; k < n; ++k)
				{
					x[k] = mod_mul(sbc(mod_mul(x[k], wi[k]), width[k], c), w[k]);
					if (c == 0) return;
				}
			}
			else
			{
				// Inverse radix-2, unweight, sub with carry, weight, radix-2
				const size_t n_2 = n / 2;
				for (size_t k = 0; k < n_2; ++k)
				{
					const uint64 u0 = x[k + 0 * n_2], u1 = x[k + 1 * n_2];
					const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));
					const uint64 v0n = mod_mul(sbc(mod_mul(v0, wi[k + 0 * n_2]), width[k + 0 * n_2], c), w[k + 0 * n_2]);
					x[k + 0 * n_2] = mod_add(v0n, v1); x[k + 1 * n_2] = mod_sub(v0n, v1);
					if (c == 0) return;
				}
				for (size_t k = 0; k < n_2; ++k)
				{
					const uint64 u0 = x[k + 0 * n_2], u1 = x[k + 1 * n_2];
					const uint64 v0 = mod_half(mod_add(u0, u1)), v1 = mod_half(mod_sub(u0, u1));
					const uint64 v1n = mod_mul(sbc(mod_mul(v1, wi[k + 1 * n_2]), width[k + 1 * n_2], c), w[k + 1 * n_2]);
					x[k + 0 * n_2] = mod_add(v0, v1n); x[k + 1 * n_2] = mod_sub(v0, v1n);
					if (c == 0) return;
				}
			}
		}
	}

	void error() const override
	{
		uint64 * const x = &_reg[0];
		x[get_size() / 2] += 1;
	}

	size_t get_checkpoint_size() const override { return 3 * get_size() * sizeof(uint64); }

	bool get_checkpoint(std::vector<char> & data) const override
	{
		const size_t size = get_checkpoint_size();
		if (data.size() != size) return false;
		std::memcpy(data.data(), _reg, size);
		return true;
	}

	bool set_checkpoint(const std::vector<char> & data) const override
	{
		const size_t size = get_checkpoint_size();
		if (data.size() != size) return false;
		std::memcpy(_reg, data.data(), size);
		return true;
	}
};
