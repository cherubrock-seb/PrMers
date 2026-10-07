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
		// Goldilocks validity: n must divide (MOD_P-1)/192 = 2^26*5*17*257*65537.
		// Therefore pure 2^27 is invalid; MM31 must choose 5*2^25 instead.
		uint32_t w = 0, log2_n = 1, log2_n5 = 2;
		do
		{
			++log2_n;
			// digit-width is w or w + 1
			w = exponent >> log2_n;
		// Digits are non-negative and less than 2^{w + 1}. The weighted convolution coefficient is a sum of n products
		// a_i b_j 2^e, where the IBDWT weight contributes e = 0 or 1 (ceil(qi/n) + ceil(qj/n) - ceil(qk/n) is 0 or 1,
		// and the wrap-around is exact modulo 2^q - 1). Hence the coefficient is at most n * 2 * (2^{w + 1} - 1)^2
		// and must be < 2^64 - 2^32 + 1.
		// If (w + 1) * 2 + log2(n) = 63 then 2 * n * (2^{w + 1} - 1)^2 < 2^64 * (1 - 2^{-w}) < 2^64 - 2^32 + 1
		// (w < 32), so the power-of-two condition below is sufficient.
		} while ((w + 1) * 2 + log2_n >= 64);

		do
		{
			++log2_n5;
			w = exponent / (5u << log2_n5);
		// n = 5 * 2^k: the coefficient is at most 5 * 2^k * 2 * (2^{w + 1} - 1)^2 < 2^{2 * (w + 1) + k + 1 + log2(5)}.
		// log2(5) ~ 2.3219 < 2.4, hence the condition 2 * (w + 1) + k + 1 + 2.4 < 64 (that is 2 * (w + 1) + k <= 60).
		} while ((w + 1) * 2 + (log2_n5 + 3.4) >= 64);

		const size_t invalid = size_t(-1);
		const size_t n2 = (log2_n <= 26) ? (size_t(1) << log2_n) : invalid;
		const size_t n5 = (log2_n5 <= 26) ? (size_t(5) << log2_n5) : invalid;
		return std::min(n2, n5);	// must be >= 4 and divide (MOD_P-1)/192
	}

	// Largest multiplier a for which the carry kernels (carry_weight_mul_p1 and friends: square_mul(r, a), mul(.., a))
	// are exact with an n-point transform for the exponent q.
	//
	// adc_mul() multiplies each unweighted coefficient u by a and propagates the carry in 64-bit registers. The
	// coefficient is at most Lmax = 2 * n * (2^{w + 1} - 1)^2 (see transform_size) with w = floor(q / n); the digit
	// width is at least w, so a digit that receives a carry c_in sends on at most
	//   c_out <= (u * a + c_in) / 2^w   =>   c_out <= Lmax * a / (2^w - 1).
	// The carry must stay below 2^64: a * c, the (u >> width) * a term, is added to a 64-bit carry and the carries
	// are stored in 64-bit words between work-groups. Hence
	//   a <= (2^64 - 1) * (2^w - 1) / Lmax.
	// (If w = 0 there is no decay, the carry can accumulate over the n digits: a <= (2^64 - 1) / (n * Lmax).)
	// The result is capped at 2^32 - 1, the range of the kernel argument.
	static constexpr uint32_t max_small_multiplier(const size_t n, const uint32_t q)
	{
		constexpr uint64_t u64max = ~uint64_t(0), u32max = 0xffffffffull;
		const uint64_t w = (n != 0) ? q / n : 0;
		if (n == 0 || w >= 30) return 0;	// not a valid transform
		const uint64_t d = (uint64_t(2) << w) - 1;	// 2^{w + 1} - 1
		if (d * d > u64max / (2 * uint64_t(n))) return 0;	// coefficient would exceed 2^64: not a valid transform
		const uint64_t lmax = 2 * uint64_t(n) * d * d;
		if (w == 0 && lmax > u64max / n) return 0;
		const uint64_t decay = (w == 0) ? 1 : (uint64_t(1) << w) - 1;	// 2^w - 1
		const uint64_t denom = (w == 0) ? lmax * uint64_t(n) : lmax;
#ifdef __SIZEOF_INT128__
		const unsigned __int128 cap = (unsigned __int128)u64max * decay / denom;
		return cap > u32max ? uint32_t(u32max) : uint32_t(cap);
#else
		// floor(u64max / denom) * decay never exceeds the exact bound u64max * decay / denom
		const uint64_t q1 = u64max / denom;
		if (q1 >= u32max) return uint32_t(u32max);
		const uint64_t cap = q1 * decay;
		return cap > u32max ? uint32_t(u32max) : uint32_t(cap);
#endif
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

	// Init roots, radix-5 is the first stage of the transform
	static void roots(const size_t n, uint64 * const root)
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