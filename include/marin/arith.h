/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <bit>
#include <stdexcept>

static constexpr int ilog2(const size_t n) { return int(std::bit_width(n)) - 1; }

#define INLINE	static inline

typedef uint8_t		uint8;
typedef uint32_t	uint32;
typedef uint64_t	uint64;

// The prime finite field with p = 2^64 - 2^32 + 1

#define	MOD_P		((((1ull << 32) - 1) << 32) + 1)
#define	MOD_MP64	uint32(-1)	// -p mod (2^64) = 2^32 - 1

INLINE uint64 mod_add(const uint64 lhs, const uint64 rhs) { return lhs + rhs + ((lhs >= MOD_P - rhs) ? MOD_MP64 : 0); }
INLINE uint64 mod_sub(const uint64 lhs, const uint64 rhs) { return lhs - rhs - ((lhs < rhs) ? MOD_MP64 : 0); }

// t modulo p. We must have t < p^2.
INLINE uint64 reduce(const uint64 lo, const uint64 hi)
{
	// hih * 2^96 + hil * 2^64 + lo = lo + hil * 2^32 - (hih + hil)
	const uint64 r = (lo >= MOD_P) ? lo - MOD_P : lo;	// lhs * rhs < p^2 => hi * 2^32 < p^2 / 2^32 < p.
	return mod_sub(mod_add(r, (hi << 32) - uint32(hi)), hi >> 32);
}

INLINE uint64 mod_mul(const uint64 lhs, const uint64 rhs)
{
	uint64 lo, hi;
#ifdef _MSC_VER
	lo = _umul128(lhs, rhs, &hi);
#else
	const __uint128_t t = lhs * __uint128_t(rhs);
	lo = uint64(t); hi = uint64(t >> 64);
#endif
	return reduce(lo, hi);
}

INLINE uint64 mod_sqr(const uint64 lhs) { return mod_mul(lhs, lhs); }

INLINE uint64 mod_muli(const uint64 lhs) { return reduce(lhs << 48, lhs >> (64 - 48)); }

INLINE uint64 mod_half(const uint64 lhs) { return ((lhs % 2 == 0) ? lhs / 2 : ((lhs - 1) / 2 + (MOD_P + 1) / 2)); }

INLINE uint64 mod_pow(const uint64 lhs, const uint64 e)
{
	if (e == 0) return 1;

	uint64 r = 1, y = lhs;
	for (uint64 i = e; i != 1; i /= 2)
	{
		if (i % 2 != 0) r = mod_mul(r, y);
		y = mod_mul(y, y);
	}

	return mod_mul(r, y);
}

// lhs must be in [1, p). Zero has no inverse (the power would silently return 0) and a value >= p is not a
// field element. Used while building the root/weight tables, never in a transform loop.
INLINE uint64 mod_invert(const uint64 lhs)
{
	if (lhs == 0 || lhs >= MOD_P) throw std::invalid_argument("mod_invert: argument must be in [1, p)");
	return mod_pow(lhs, MOD_P - 2);
}

// A primitive n-th root of unity. n must divide p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537: for any other n the
// integer division below would silently give a root of the wrong order (or divide by zero for n == 0).
// Used while building the root tables, never in a transform loop.
INLINE uint64 mod_root_nth(const uint64 n)
{
	if (n == 0 || (MOD_P - 1) % n != 0) throw std::invalid_argument("mod_root_nth: n must divide p - 1");
	return mod_pow(7, (MOD_P - 1) / n);
}

// Add a carry onto the number and return the carry of the first width bits.
// width must be in [1, 31]: the digit mask is built in 32 bits and the carry is shifted by 64 - width, so
// width == 0 or width >= 32 is undefined behaviour. The check costs one compare per digit and adc() is only
// called from the (non-transform) final conversion of a residue.
INLINE uint32 adc(const uint64 lhs, const uint8 width, uint64 & carry)
{
	if (width == 0 || width >= 32) throw std::invalid_argument("adc: digit width must be in [1, 31]");
	const uint64 s = lhs + carry;
	const uint64 c = (s < lhs) ? 1 : 0;
	carry = (s >> width) + (c << (64 - width));
	return uint32(s) & ((1u << width) - 1);
}

// Add carry and mul
INLINE uint32 adc_mul(const uint64 lhs, const uint32 a, const uint8 width, uint64 & carry)
{
	uint64 c = 0;
	const uint32 d = adc(lhs, width, c);
	const uint32 r = adc(uint64(d) * a, width, carry);
	carry += a * c;
	return r;
}

// There is no host sbc(): the host never subtracts digit by digit (the GPU kernels do, with the exact
// multi-unit-borrow sbc_reg() in kernels/marin.cl). A single-unit-borrow version used to live here, unused;
// it was wrong whenever the carry exceeded 2^width, so it was removed rather than kept as a trap.
