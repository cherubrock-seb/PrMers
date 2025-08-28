/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <algorithm>
#include <vector>

#include "arith.h"

class engine
{
private:
	const size_t _n;
	const bool _even;
	const int _ln_max;

private:
	static constexpr bool is_even(const size_t n)
	{
		size_t m = (n % 5 == 0) ? n / 10 : n;
		for (; m > 1; m /= 4);
		return (m == 1);
	}

protected:
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

	// Bit-reversal permutation
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// Init roots
	void roots(uint64 * const root) const
	{
		const size_t n = _n, n5 = (n % 5 == 0) ? n / 5 : n;

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

	// Init weights and digit widths
	void weights_widths(const uint32_t q, uint64 * const weight, uint8 * const width) const
	{
		const size_t n = _n;

		uint64 * const w = &weight[0];
		uint64 * const wi_n = &weight[n];
		uint64 * const wi = &weight[2 * n];

		// n-th root of two
		const uint64 nr2 = mod_pow(554, (MOD_P - 1) / 192 / n);

		const uint32 q_n = q / uint32(n);

		const uint64 inv_n = mod_invert((n % 5 == 0) ? n : n / 2);
		w[0] = 1; wi_n[0] = inv_n; wi[0] = 1;

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
			const uint64 nr2ri = mod_invert(nr2r);
			w[j] = nr2r; wi_n[j] = mod_mul(nr2ri, inv_n); wi[j] = nr2ri;
			ceil_qjm1_n = ceil_qj_n;
		}
	}

public:
	engine(const uint32_t q) : _n(transform_size(q)), _even(is_even(_n)), _ln_max(ilog2_32(uint32_t((_n % 5 == 0) ? _n / 5 : _n)) - (_even ? 2 : 3)) {}
	virtual ~engine() {}

	size_t get_size() const { return _n; }
	bool get_even() const { return _even; }
	int get_ln_max() const { return _ln_max; }

	bool is_one(const std::vector<uint64> & d, uint64_t & res64) const
	{
		const size_t n = get_size();
		const uint64 * const x = d.data();

		uint64_t r64 = 0; uint8 s = 0;
		for (size_t k = 0; k < n; ++k)
		{
			const uint64_t x_k = x[k], u = uint32(x_k);
			const uint8 width = uint8(x_k >> 32);
			r64 += u << s;
			s += width;
			if (s >= 64) break;
		}
		res64 = r64;

		if (uint32(x[0]) != 1) return false;
		for (size_t k = 1; k < n; ++k) if (uint32(x[k]) != 0) return false;
		return true;
	}

	// Interface

	typedef size_t Reg;

	virtual void set(const Reg dst, const uint64 a) const = 0;
	virtual void get(uint64 * const d, const Reg src) const = 0;
	virtual void copy(const Reg dst, const Reg src) const = 0;
	virtual bool is_equal(const Reg src1, const Reg src2) const = 0;
	virtual void square_mul(const Reg src, const uint32_t a = 1) const = 0;
	virtual void set_multiplicand(const Reg dst, const Reg src) const = 0;
	virtual void mul(const Reg dst, const Reg src) const = 0;
	virtual void error() const = 0;

	virtual size_t get_checkpoint_size() const = 0;
	virtual bool get_checkpoint(std::vector<char> & data) const = 0;
	virtual bool set_checkpoint(const std::vector<char> & data) const = 0;

	static engine * create_cpu(const uint32_t q);
	static engine * create_gpu(const uint32_t q, const size_t reg_count, const size_t device, const bool verbose);
};

class engine_info : public engine
{
public:
	engine_info(const uint32_t q) : engine(q) {}
	virtual ~engine_info() {}

	void set(const Reg, const uint64) const override {}
	void get(uint64 *, const Reg) const override  {}
	void copy(const Reg, const Reg) const override {}
	bool is_equal(const Reg, const Reg) const override { return false; }
	void square_mul(const Reg, const uint32_t) const override {}
	void set_multiplicand(const Reg, const Reg) const override {}
	void mul(const Reg, const Reg) const override {}
	void error() const override {}
	size_t get_checkpoint_size() const override { return 0; }
	bool get_checkpoint(std::vector<char> &) const override { return false; }
	bool set_checkpoint(const std::vector<char> &) const override { return false; }
};
