/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <algorithm>

#include "arith.h"

class ibdwt
{
public:
	static constexpr size_t transform_size(const uint32_t exponent)
	{
		// Make sure the transform is long enough so that each 'digit' can't overflow after the convolution.
		uint32_t w = 0, log2_n = 1, log2_n5 = 2;
		do
		{
			++log2_n;
			// digit-width is w or w + 1
			w = exponent >> log2_n;
		// The condition is n * (2^{w + 1} - 1)^2 < 2^64 - 2^32 + 1.
		// If (w + 1) * 2 + log2(n) = 63 then n * (2^{w + 1} - 1)^2 < n * (2^{w + 1})^2 = 2^63 < 2^64 - 2^32 + 1.
		} while ((w + 1) * 2 + log2_n >= 64);

		do
		{
			++log2_n5;
			w = exponent / (5u << log2_n5);
		// log2(5) ~ 2.3219 < 2.4
		} while ((w + 1) * 2 + (log2_n5 + 2.4) >= 64);

		return std::min(size_t(1) << log2_n, size_t(5) << log2_n5);	// must be >= 4
	}

	static constexpr bool is_even(const size_t n)
	{
		size_t m = (n % 5 == 0) ? n / 5 : n;
		for (; m > 1; m /= 4);
		return (m == 1);
	}

	// Bit-reversal permutation
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// Digit-reversal permutation (n = 2^e * 5^f)
	static constexpr size_t reversal(const size_t i, const size_t n)
	{
		size_t r = 0, k = n, j = i;
		while (k % 2 == 0) { r = 2 * r + j % 2; k /= 2; j /= 2; }
		while (k % 5 == 0) { r = 5 * r + j % 5; k /= 5; j /= 5; }
		return r;
	}

	// Inverse digit-reversal permutation (n = 2^e * 5^f)
	static constexpr size_t inv_reversal(const size_t i, const size_t n)
	{
		size_t r = 0, k = n, j = i;
		while (k % 5 == 0) { r = 5 * r + j % 5; k /= 5; j /= 5; }
		while (k % 2 == 0) { r = 2 * r + j % 2; k /= 2; j /= 2; }
		return r;
	}

	// Init roots, radix-5 is the last stage of the transform
	static void roots45(const size_t n, uint64 * const root)
	{
		const size_t n5 = (n % 5 == 0) ? n / 5 : n;

		// n mod 5 != 0 => n roots
		// n mod 5 != 0 => 2 * n / 5 roots

		uint64 * const r2 = &root[0];
		uint64 * const r2i = &root[n];

		const uint64 r = mod_root_nth(2 * n5 / 2), ri = mod_invert(r);
		uint64 r_j = 1, ri_j = 1;
		for (size_t j = 0; j < n5 / 2; ++j)
		{
			const size_t jr = bitrev(j, n5 / 2);
			r2[jr] = r_j; r2i[jr] = ri_j;
			r_j = mod_mul(r_j, r); ri_j = mod_mul(ri_j, ri);
		}

		uint64 * const r4 = &root[n5 / 2];
		uint64 * const r4i = &root[n + n5 / 2];

		for (size_t j = 0; j < n5 / 4; ++j)
		{
			r4[2 * j + 0] = r2[2 * j]; r4i[2 * j + 0] = r2i[2 * j];
			r4[2 * j + 1] = mod_mul(r2[j], r2[2 * j]); r4i[2 * j + 1] = mod_mul(r2i[j], r2i[2 * j]);
		}

		if (n % 5 == 0)
		{
			uint64 * const r5 = &root[n5];
			uint64 * const r5i = &root[n + n5];

			const uint64 r = mod_root_nth(5 * n5), ri = mod_invert(r);
			uint64 r_j = 1, ri_j = 1;
			for (size_t j = 0; j < n5; ++j)
			{
				const size_t jr = bitrev(j, n5);
				r5[jr] = r_j; r5i[jr] = ri_j;
				r_j = mod_mul(r_j, r); ri_j = mod_mul(ri_j, ri);
			}
		}
	}

	// Init roots, radix-5 is the first stage of the transform
	static void roots54(const size_t n, uint64 * const root)
	{
		uint64 * const r2 = &root[0];
		uint64 * const r2i = &root[n / 2];

		for (size_t s = (n % 5 == 0) ? 5 : 1; s <= n / 4; s *= 2)
		{
			const uint64 rs = mod_root_nth(2 * s), rsi = mod_invert(rs);
			uint64 rsj = 1, rsji = 1;
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = inv_reversal(j, s);
				r2[s + jr] = rsj; r2i[s + jr] = rsji;
				rsj = mod_mul(rsj, rs); rsji = mod_mul(rsji, rsi);
			}
		}

		uint64 * const r4 = &root[n];
		uint64 * const r4i = &root[n + n];

		for (size_t s = (n % 5 == 0) ? 5 : 1; s <= n / 4; s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t sj = s + j;
				r4[2 * sj + 0] = r2[2 * sj]; r4i[2 * sj + 0] = r2i[2 * sj];
				r4[2 * sj + 1] = mod_mul(r2[sj], r2[2 * sj]); r4i[2 * sj + 1] = mod_mul(r2i[sj], r2i[2 * sj]);
			}
		}
	}

	// Init weights and digit widths
	static void weights_widths(const size_t n, const uint32_t q, uint64 * const weight, uint8 * const width)
	{
		uint64 * const w = &weight[0];

		// n-th root of two
		const uint64 nr2 = mod_pow(554, (MOD_P - 1) / 192 / n);

		const uint32 q_n = q / uint32(n);

		w[2 * 0 + 0] = 1; w[2 * 0 + 1] = 1;

		uint32 ceil_qjm1_n = 0;
		for (size_t j = 1; j <= n; ++j)
		{
			const uint64 qj = q * uint64(j);
			// ceil(a / b) = floor((a - 1) / b) + 1
			const uint32 ceil_qj_n = uint32((qj - 1) / n + 1);

			// bit position for digit[i] is ceil(qj / n)
			const uint32 c = ceil_qj_n - ceil_qjm1_n;
			if ((c != q_n) && (c != q_n + 1)) throw;
			width[j - 1] = uint8(c);

			if (j == n) break;

			// weight is 2^[ceil(qj / n) - qj / n]
			// e = (ceil(qj / n).n - qj) / n
			// qj = k * n => e = 0
			// qj = k * n + r, r > 0 => ((k + 1).n - k.n + r) / n = (n - r) / n
			const uint32 r = uint32(qj % n);
			const uint64 nr2r = (r != 0) ? mod_pow(nr2, n - r) : 1;
			const size_t i = (j % 4) * (n / 4) + (j / 4);
			w[2 * i + 0] = nr2r; w[2 * i + 1] = mod_invert(nr2r);

			ceil_qjm1_n = ceil_qj_n;
		}
	}
};

