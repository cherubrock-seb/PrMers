/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static constexpr int ilog2_32(const uint32_t n) { return 31 - __builtin_clz(n); }

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
	return mod_sub(mod_add(r, (hi << 32) - uint32_t(hi)), hi >> 32);
}

INLINE uint64 mod_mul(const uint64 lhs, const uint64 rhs)
{
	const __uint128_t t = lhs * __uint128_t(rhs);
	return reduce(uint64(t), uint64(t >> 64));
}

INLINE uint64 mod_sqr(const uint64 lhs) { return mod_mul(lhs, lhs); }

INLINE uint64 mod_muli(const uint64 lhs)
{
	const __uint128_t t = __uint128_t(lhs) << 48;	// sqrt(-1) = 2^48 (mod p)
	return reduce(uint64(t), uint64(t >> 64));
}

INLINE uint64 mod_half(const uint64 lhs) { return ((lhs % 2 == 0) ? lhs / 2 : ((lhs - 1) / 2 + (MOD_P + 1) / 2)); }

static uint64 mod_pow(const uint64 lhs, const uint64 e)
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

static uint64 mod_invert(const uint64 lhs) { return mod_pow(lhs, MOD_P - 2); }

static uint64 mod_root_nth(const uint64 n) { return mod_pow(7, (MOD_P - 1) / n); }

// Add a carry onto the number and return the carry of the first width bits
INLINE uint32 adc(const uint64 lhs, const uint8 width, uint64 & carry)
{
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
